###############################################################################
# tests/test_protocol.py
#
# Requires: pytest, torch, transformers
# Assumes the production code lives in grail/__init__.py
###############################################################################
import importlib
import inspect
import os
import types
from copy import deepcopy

import pytest
import torch

import grail as ROLL

###############################################################################
# Helpers & fixtures
###############################################################################

@pytest.fixture(scope="session")
def model_dim():
    return ROLL.AutoModelForCausalLM.from_pretrained(
        ROLL.MODEL_NAME, output_hidden_states=True
    ).config.hidden_size


@pytest.fixture(scope="session")
def prover():
    return ROLL.Prover()


@pytest.fixture(scope="session")
def verifier():
    return ROLL.Verifier()


@pytest.fixture
def commit_and_proof(prover):
    commit = prover.commit("Test prompt.")
    proof = prover.open(k=ROLL.CHALLENGE_K)
    return commit, proof


###############################################################################
# PRF & beacon tests
###############################################################################

def test_prf_repeatability():
    """Same inputs → identical output bytes."""
    out1 = ROLL.prf(b"label", b"abc", out_bytes=64)
    out2 = ROLL.prf(b"label", b"abc", out_bytes=64)
    assert out1 == out2 and len(out1) == 64


def test_prf_domain_separation():
    """Different labels or parts must change the stream."""
    base = ROLL.prf(b"L1", b"payload", out_bytes=32)
    diff = [
        ROLL.prf(b"L2", b"payload", out_bytes=32),
        ROLL.prf(b"L1", b"payload!", out_bytes=32),
    ]
    assert all(x != base for x in diff)


def test_beacon_monotonic():
    r1 = ROLL.get_beacon()
    r2 = ROLL.get_beacon()
    assert r2["round"] == r1["round"] + 1
    assert r1["randomness"] != r2["randomness"]


###############################################################################
# r‑vector & index selection
###############################################################################

def test_r_vec_length(model_dim):
    rnd = "00" * 32
    r_vec = ROLL.r_vec_from_randomness(rnd, model_dim)
    assert r_vec.shape[0] == model_dim and r_vec.dtype == torch.int32


def test_indices_properties():
    tokens = [1, 2, 3, 4, 5] * 20  # 100 tokens
    rnd_hex = os.urandom(32).hex()
    idxs = ROLL.indices_from_root(tokens, rnd_hex, 100, 16)
    assert idxs == sorted(idxs)  # sorted
    assert len(set(idxs)) == 16  # unique
    assert all(0 <= i < 100 for i in idxs)


###############################################################################
# Sketch (dot product) tests
###############################################################################

def test_dot_mod_q_bounds(model_dim):
    h = torch.ones(model_dim)
    r = torch.ones(model_dim, dtype=torch.int32)
    val = ROLL.dot_mod_q(h, r)
    assert 0 <= val < ROLL.PRIME_Q


def test_tolerance_boundary():
    diff = ROLL.TOLERANCE
    assert (diff % ROLL.PRIME_Q) == diff


###############################################################################
# Signature tests
###############################################################################

def test_sign_and_verify_s_vals():
    """Signature verification should work with correct key."""
    s_vals = [1, 2, 3, 4, 5]
    key = os.urandom(32)
    signature = ROLL.sign_s_vals(s_vals, key)
    assert ROLL.verify_s_vals_signature(s_vals, signature, key)


def test_signature_with_wrong_key_fails():
    """Signature verification should fail with wrong key."""
    s_vals = [1, 2, 3, 4, 5]
    key1 = os.urandom(32)
    key2 = os.urandom(32)
    signature = ROLL.sign_s_vals(s_vals, key1)
    assert not ROLL.verify_s_vals_signature(s_vals, signature, key2)


def test_signature_with_modified_s_vals_fails():
    """Signature verification should fail if s_vals are modified."""
    s_vals = [1, 2, 3, 4, 5]
    key = os.urandom(32)
    signature = ROLL.sign_s_vals(s_vals, key)
    
    modified_s_vals = s_vals[:]
    modified_s_vals[0] = 999
    assert not ROLL.verify_s_vals_signature(modified_s_vals, signature, key)


###############################################################################
# End‑to‑end happy path
###############################################################################

def test_commit_open_verify(commit_and_proof, prover, verifier):
    commit_pkg, proof_pkg = commit_and_proof
    assert verifier.verify(commit_pkg, proof_pkg, prover.secret_key)


###############################################################################
# Negative tests – each must break verification
###############################################################################

@pytest.mark.parametrize("mutate_field", [
    ("s_val", lambda x: (x + 7) % ROLL.PRIME_Q),
])
def test_tampered_proof_fails(commit_and_proof, prover, verifier, mutate_field):
    commit_pkg, proof_pkg = commit_and_proof
    field, mutator = mutate_field

    # pick first opened index and mangle the chosen field
    idx = proof_pkg["indices"][0]
    commit_pkg = deepcopy(commit_pkg)  # do *not* trash other tests
    
    if field == "s_val":
        # Modify s_val in the commitment and re-sign
        commit_pkg["s_vals"][idx] = mutator(commit_pkg["s_vals"][idx])
        commit_pkg["signature"] = ROLL.sign_s_vals(commit_pkg["s_vals"], prover.secret_key).hex()

    assert not verifier.verify(commit_pkg, proof_pkg, prover.secret_key)


def test_signature_tamper_fails(commit_and_proof, prover, verifier):
    commit_pkg, proof_pkg = commit_and_proof
    commit_pkg = deepcopy(commit_pkg)
    # flip one bit of the signature
    sig_bytes = bytearray.fromhex(commit_pkg["signature"])
    sig_bytes[0] ^= 0x80
    commit_pkg["signature"] = bytes(sig_bytes).hex()
    assert not verifier.verify(commit_pkg, proof_pkg, prover.secret_key)


def test_index_set_tamper_fails(commit_and_proof, prover, verifier):
    commit_pkg, proof_pkg = commit_and_proof
    proof_pkg = deepcopy(proof_pkg)
    # Shift one index (preserve ordering)
    if len(proof_pkg["indices"]) > 0:
        proof_pkg["indices"][-1] = max(0, proof_pkg["indices"][-1] - 1)
    assert not verifier.verify(commit_pkg, proof_pkg, prover.secret_key)


###############################################################################
# Property‑style tests: open exactly K, stay within tolerance, etc.
###############################################################################

def test_open_returns_exact_k(prover):
    commit = prover.commit("hello")
    k = 5
    proof = prover.open(k=k)
    assert len(proof["indices"]) == k
    assert set(proof["indices"]) <= set(range(len(commit["tokens"])))


def test_s_vals_hash_deterministic():
    """Same s_vals should produce same hash."""
    s_vals = [1, 2, 3, 4, 5]
    hash1 = ROLL.hash_s_vals(s_vals)
    hash2 = ROLL.hash_s_vals(s_vals)
    assert hash1 == hash2


def test_s_vals_hash_different_for_different_vals():
    """Different s_vals should produce different hashes."""
    s_vals1 = [1, 2, 3, 4, 5]
    s_vals2 = [1, 2, 3, 4, 6]  # only last element differs
    hash1 = ROLL.hash_s_vals(s_vals1)
    hash2 = ROLL.hash_s_vals(s_vals2)
    assert hash1 != hash2

