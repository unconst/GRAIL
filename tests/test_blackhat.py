###############################################################################
# Adversarial stress tests for ROLL‑CHECK
###############################################################################
import copy
import os
import random

import pytest
import torch
from hypothesis import given, settings, strategies as st

import grail as R

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def prov_and_ver():
    return R.Prover(), R.Verifier()

@pytest.fixture
def commit_and_proof(prov_and_ver):
    prover, _ = prov_and_ver
    commit = prover.commit("forty‑two")
    proof = prover.open(k=R.CHALLENGE_K)
    return commit, proof

# ---------------------------------------------------------------------------
# Helper: mutate helpers
# ---------------------------------------------------------------------------

def _flipped_bit(b: bytes) -> bytes:
    ba = bytearray(b)
    i = random.randrange(len(ba))
    ba[i] ^= 0x80
    return bytes(ba)

# ---------------------------------------------------------------------------
# 1.  Beacon replay / skew
# ---------------------------------------------------------------------------

def test_beacon_replay_fails(commit_and_proof, prov_and_ver):
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    proof = copy.deepcopy(proof)
    # Re‑use R (first beacon) as R1 to force index mismatch
    proof["round_R1"] = commit["round_R"]
    assert not verifier.verify(commit, proof, prover.secret_key)

# ---------------------------------------------------------------------------
# 2.  Tolerance boundary conditions
# ---------------------------------------------------------------------------

def test_sketch_plus_tolerance_still_ok(commit_and_proof, prov_and_ver):
    """Δ ≤ TOLERANCE passes."""
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    idx = proof["indices"][0]
    commit = copy.deepcopy(commit)
    # Modify s_val within tolerance - should still pass
    commit["s_vals"][idx] = (commit["s_vals"][idx] + R.TOLERANCE) % R.PRIME_Q
    # Re-sign the modified s_vals
    commit["signature"] = R.sign_s_vals(commit["s_vals"], prover.secret_key).hex()
    assert verifier.verify(commit, proof, prover.secret_key)

def test_sketch_beyond_tolerance_fails(commit_and_proof, prov_and_ver):
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    idx = proof["indices"][0]
    commit = copy.deepcopy(commit)
    # Modify s_val beyond tolerance - should fail
    commit["s_vals"][idx] = (commit["s_vals"][idx] + R.TOLERANCE + 1) % R.PRIME_Q
    # Re-sign the modified s_vals
    commit["signature"] = R.sign_s_vals(commit["s_vals"], prover.secret_key).hex()
    assert not verifier.verify(commit, proof, prover.secret_key)

# ---------------------------------------------------------------------------
# 3.  s_vals tampering (replaces Merkle branch tests)
# ---------------------------------------------------------------------------

def test_unsigned_s_vals_modification_fails(commit_and_proof, prov_and_ver):
    """Modifying s_vals without re-signing should fail signature check."""
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    idx = proof["indices"][0]
    commit = copy.deepcopy(commit)
    # Modify s_val but DON'T re-sign - should fail signature check
    commit["s_vals"][idx] = (commit["s_vals"][idx] + 1) % R.PRIME_Q
    assert not verifier.verify(commit, proof, prover.secret_key)

def test_wrong_signature_key_fails(commit_and_proof, prov_and_ver):
    """Using wrong signature key should fail verification."""
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    wrong_key = os.urandom(32)  # Different key
    assert not verifier.verify(commit, proof, wrong_key)

# ---------------------------------------------------------------------------
# 4.  Truncated / padded token sequences
# ---------------------------------------------------------------------------

def test_token_truncation_fails(commit_and_proof, prov_and_ver):
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    commit = copy.deepcopy(commit)
    commit["tokens"] = commit["tokens"][:-1]
    assert not verifier.verify(commit, proof, prover.secret_key)

def test_token_padding_fails(commit_and_proof, prov_and_ver):
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    commit = copy.deepcopy(commit)
    commit["tokens"].append(42)
    assert not verifier.verify(commit, proof, prover.secret_key)

# ---------------------------------------------------------------------------
# 5.  s_vals list tampering
# ---------------------------------------------------------------------------

def test_s_vals_truncation_fails(commit_and_proof, prov_and_ver):
    """Truncating s_vals should fail."""
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    commit = copy.deepcopy(commit)
    commit["s_vals"] = commit["s_vals"][:-1]
    assert not verifier.verify(commit, proof, prover.secret_key)

def test_s_vals_padding_fails(commit_and_proof, prov_and_ver):
    """Padding s_vals should fail."""
    commit, proof = commit_and_proof
    prover, verifier = prov_and_ver
    commit = copy.deepcopy(commit)
    commit["s_vals"].append(42)
    assert not verifier.verify(commit, proof, prover.secret_key)

# ---------------------------------------------------------------------------
# 6.  Random fuzzy tampering (simplified)
# ---------------------------------------------------------------------------

@given(shift = st.integers(min_value=1, max_value=2**31-1))
@settings(max_examples=25, deadline=None)
def test_random_s_val_tampering(shift):
    """Random tampering of s_vals should fail without proper signing."""
    prover = R.Prover()
    verifier = R.Verifier()
    commit = prover.commit("test prompt")
    proof = prover.open(k=min(4, len(commit["s_vals"])))  # Use smaller k for faster testing
    
    if not proof["indices"]:  # Skip if no indices selected
        return
    
    commit = copy.deepcopy(commit)
    idx = proof["indices"][0]
    original = commit["s_vals"][idx]
    
    # Ensure the shift actually changes the value
    new_val = (original + shift) % R.PRIME_Q
    if new_val == original:  # Skip if no actual change
        return
        
    commit["s_vals"][idx] = new_val
    
    # Without re-signing, this should fail
    assert not verifier.verify(commit, proof, prover.secret_key)
