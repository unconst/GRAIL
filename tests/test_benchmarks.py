###############################################################################
# Timing benchmarks for ROLL‑CHECK (requires pytest‑benchmark)
###############################################################################
import pytest
import torch
import grail as R

# ---------------------------------------------------------------------------
# Bench matrix
# ---------------------------------------------------------------------------

PROMPTS = {
    "short":  "Hello world.",
    "medium": " ".join(["lorem"] * 64),      # 64 tokens ≈ 64 words
    "long":   " ".join(["ipsum"] * 256),     # 256 tokens
}

K_SET = (1, 8, 32)        # audit‑set sizes

# ---------------------------------------------------------------------------
# Utility: detach CUDA sync cost
# ---------------------------------------------------------------------------

def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_key", PROMPTS.keys())
@pytest.mark.benchmark(group="commit")
def test_commit_bench(benchmark, prompt_key):
    prover = R.Prover()
    prompt = PROMPTS[prompt_key]
    def _run():
        pkg = prover.commit(prompt, max_new_tokens=32)
        _sync(prover.device)
        return pkg
    benchmark.extra_info["prompt_len"] = prompt_key
    benchmark(_run)

@pytest.mark.parametrize("k", K_SET)
@pytest.mark.benchmark(group="open")
def test_open_bench(benchmark, k):
    prover = R.Prover()
    commit = prover.commit("bench", max_new_tokens=32)
    def _run():
        pkg = prover.open(k=k)
        _sync(prover.device)
        return pkg
    benchmark.extra_info["k"] = k
    benchmark(_run)

@pytest.mark.parametrize("prompt_key", PROMPTS.keys())
@pytest.mark.parametrize("k", K_SET)
@pytest.mark.benchmark(group="verify")
def test_verify_bench(benchmark, prompt_key, k):
    prover   = R.Prover()
    verifier = R.Verifier()
    commit   = prover.commit(PROMPTS[prompt_key], max_new_tokens=32)
    proof    = prover.open(k=k)
    def _run():
        ok = verifier.verify(commit, proof, prover.secret_key)
        _sync(verifier.device)
        assert ok
    benchmark.extra_info.update({"prompt": prompt_key, "k": k})
    benchmark(_run)
