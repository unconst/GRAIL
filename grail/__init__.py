#!/usr/bin/env python3
"""
GRAIL – Guaranteed Rollout Authenticity via Inference Ledger
"""

import os
import io
import struct
import hashlib
import random
import hmac
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────  CONFIGURATION  ─────────────────────────────

DEBUG        = False                 # ← turn on/off debug prints
BEACON_COUNTER = 0

PRIME_Q      = 2_147_483_647
CHALLENGE_K  = 16
TOLERANCE    = 3

MODEL_NAME   = "sshleifer/tiny-gpt2"
LAYER_INDEX  = -1
RNG_LABEL    = {"sketch": b"sketch", "open": b"open"}

# ────────────────────  MOCK BEACON HELPERS  ──────────────────────────────

def get_beacon(round_id: str = "latest") -> dict:
    global BEACON_COUNTER
    BEACON_COUNTER += 1
    rnd = os.urandom(32).hex()
    if DEBUG:
        print(f"[Beacon] round={BEACON_COUNTER}, randomness={rnd[:8]}…")
    return {"round": BEACON_COUNTER, "randomness": rnd}

def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    h = hashlib.sha256(label + b"||" + b"||".join(parts)).digest()
    while len(h) < out_bytes:
        h += hashlib.sha256(h).digest()
    return h[:out_bytes]

def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:
    raw = prf(RNG_LABEL["sketch"], bytes.fromhex(rand_hex), out_bytes=4*d_model)
    ints = struct.unpack(">" + "i"*d_model, raw)
    if DEBUG:
        print(f"[SketchVec] first 4 ints: {ints[:4]}")
    return torch.tensor(ints, dtype=torch.int32)

def indices_from_root(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    # Use tokens hash instead of s_vals hash for index derivation
    # This ensures indices remain stable even when s_vals change within tolerance
    tokens_bytes = b''.join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()
    material = prf(RNG_LABEL["open"], tokens_hash, bytes.fromhex(rand_hex), out_bytes=32)
    rnd = random.Random(material)
    idxs = sorted(rnd.sample(range(seq_len), k))
    if DEBUG:
        print(f"[Indices] selected {idxs}")
    return idxs

# ─────────────────────────────  UTILITIES  ─────────────────────────────

def int_to_bytes(i: int) -> bytes:
    return struct.pack(">I", i & 0xFFFFFFFF)

def dot_mod_q(hidden: torch.Tensor, r_vec: torch.Tensor) -> int:
    scaled = torch.round(hidden * 1024).to(torch.int64)
    r_vec_int64 = r_vec.to(torch.int64)
    # Use sum of element-wise multiplication instead of dot for int64 on CUDA
    prod = (scaled * r_vec_int64).sum()
    return int(prod.item() % PRIME_Q)

def sign_s_vals(s_vals: list[int], secret_key: bytes) -> bytes:
    """Sign the s_vals list for integrity protection."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    signature = hmac.new(secret_key, s_vals_bytes, hashlib.sha256).digest()
    if DEBUG:
        print(f"[Signature] signed {len(s_vals)} s_vals")
    return signature

def verify_s_vals_signature(s_vals: list[int], signature: bytes, secret_key: bytes) -> bool:
    """Verify the signature of s_vals list."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    expected_sig = hmac.new(secret_key, s_vals_bytes, hashlib.sha256).digest()
    return hmac.compare_digest(signature, expected_sig)

def hash_s_vals(s_vals: list[int]) -> bytes:
    """Compute hash of s_vals for integrity checking."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    return hashlib.sha256(s_vals_bytes).digest()

# ─────────────────────────────  PROVER  ────────────────────────────────

class Prover:
    def __init__(self, model_name=MODEL_NAME, torch_dtype=None):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with specified dtype
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs['torch_dtype'] = torch_dtype
            
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, **load_kwargs)
            .to(self.device)
            .eval()
        )
        # Generate a secret key for signing (in practice this would be securely managed)
        self.secret_key = os.urandom(32)

    def commit(self, prompt: str, max_new_tokens: int = 32) -> dict:
        self.beacon_R = get_beacon()
        self.r_vec    = r_vec_from_randomness(self.beacon_R["randomness"],
                                              self.model.config.hidden_size).to(self.device)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        tokens = gen.sequences[0].tolist()
        if DEBUG:
            print(f"[Commit] tokens length = {len(tokens)}")

        with torch.no_grad():
            outs = self.model(gen.sequences, output_hidden_states=True)
        h_layer = outs.hidden_states[LAYER_INDEX][0]

        s_vals = [dot_mod_q(h_layer[t], self.r_vec) for t in range(h_layer.size(0))]
        if DEBUG:
            print(f"[Commit] first 8 s_vals = {s_vals[:8]}")

        # Sign the s_vals for integrity
        signature = sign_s_vals(s_vals, self.secret_key)

        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        model_hash = hashlib.sha256(buf.getvalue()).hexdigest()

        self._state = {
            "tokens":  tokens,
            "s_vals":  s_vals,
            "seq_len": len(tokens),
            "signature": signature
        }

        return {
            "round_R":     self.beacon_R,
            "tokens":      tokens,
            "s_vals":      s_vals,
            "signature":   signature.hex(),
            "model_hash":  model_hash,
        }

    def open(self, delta_round: int = 1, k: int = CHALLENGE_K) -> dict:
        beacon_R1 = get_beacon()
        # Use tokens instead of s_vals for index derivation
        idxs = indices_from_root(self._state["tokens"],
                                beacon_R1["randomness"],
                                self._state["seq_len"],
                                k)
        return {"round_R1": beacon_R1, "indices": idxs}

# ─────────────────────────────  VERIFIER  ──────────────────────────────

class Verifier:
    def __init__(self, model_name=MODEL_NAME, torch_dtype=None):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with specified dtype
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs['torch_dtype'] = torch_dtype
            
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, **load_kwargs)
            .to(self.device)
            .eval()
        )

    def verify(self, commit: dict, proof_pkg: dict, prover_secret_key: bytes) -> bool:
        # Verify s_vals signature for integrity
        signature = bytes.fromhex(commit["signature"])
        if not verify_s_vals_signature(commit["s_vals"], signature, prover_secret_key):
            print("❌ s_vals signature verification failed")
            return False

        # Re-derive sketch vector
        r_vec = r_vec_from_randomness(
            commit["round_R"]["randomness"],
            self.model.config.hidden_size
        ).to(self.device)

        # Re-derive and compare indices using tokens (not s_vals)
        idxs_exp = indices_from_root(
            commit["tokens"],
            proof_pkg["round_R1"]["randomness"],
            len(commit["tokens"]),
            len(proof_pkg["indices"])
        )
        if idxs_exp != proof_pkg["indices"]:
            print("❌ Index-selection mismatch")
            return False

        # Recompute hidden states
        full_ids = torch.tensor(commit["tokens"], dtype=torch.long,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            outs = self.model(full_ids, output_hidden_states=True)
        h_layer = outs.hidden_states[LAYER_INDEX][0]

        # Check each opened index (tolerance check only now)
        for i in idxs_exp:
            committed_s_val = commit["s_vals"][i]
            
            # Sketch‐value check with proper modular distance
            local = dot_mod_q(h_layer[i], r_vec)
            if DEBUG:
                print(f"[SketchCheck] idx={i}, committed={committed_s_val}, local={local}")
            
            # Calculate minimum distance considering modular arithmetic
            diff = abs(local - committed_s_val)
            mod_diff = min(diff, PRIME_Q - diff)  # Handle wraparound
            
            if mod_diff > TOLERANCE:
                print(f"❌ Sketch mismatch at index {i} ({local} vs {committed_s_val}, diff={mod_diff})")
                return False

        print("✅ Verification successful")
        return True


# ─────────────────────────────  PROTOCOL WRAPPER  ──────────────────────────────

class GrailProtocol:
    """Wrapper class for experiments expecting GrailProtocol interface"""
    
    def __init__(self, model_name=MODEL_NAME, max_new_tokens=32, challenge_size=CHALLENGE_K):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.challenge_size = challenge_size
        self.prover = None
        self.verifier = None
        
    def commit(self, prompt: str) -> dict:
        """Commit phase - generate and commit to response"""
        self.prover = Prover(self.model_name)
        
        start_time = time.time()
        commit = self.prover.commit(prompt, max_new_tokens=self.max_new_tokens)
        commit_time = time.time() - start_time
        
        # Open phase
        proof_pkg = self.prover.open(k=self.challenge_size)
        
        # Format response for compatibility
        return {
            "response": self.tokenizer.decode(commit["tokens"]) if hasattr(self, 'tokenizer') else "",
            "proof": {
                "commit": commit,
                "proof_pkg": proof_pkg,
                "secret_key": self.prover.secret_key
            },
            "timing": {
                "commit_time": commit_time
            }
        }
        
    def verify(self, prompt: str, response: str, proof: dict) -> dict:
        """Verify phase - verify commitment with proof"""
        self.verifier = Verifier(self.model_name)
        
        start_time = time.time()
        result = self.verifier.verify(
            proof["commit"],
            proof["proof_pkg"],
            proof["secret_key"]
        )
        verify_time = time.time() - start_time
        
        return {
            "verified": result,
            "timing": {
                "verify_time": verify_time
            }
        }