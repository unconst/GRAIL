# GRAIL Evaluation Experiments

Comprehensive evaluation suite for testing GRAIL (Guaranteed Rollout Authenticity via Inference Ledger) with focused, individual experiments.

## 🎯 Overview

This directory contains **9 focused experiments** that systematically test different aspects of GRAIL's security and functionality:

### 🧪 Core Experiments

1. **`exp_correctness_detection.py`** - Tests that valid model/prompt combinations pass verification (Expected: 100% pass rate)
2. **`exp_model_resistance.py`** - Tests that different models are correctly detected and rejected (Expected: 100% detection rate)  
3. **`exp_prompt_tampering.py`** - Tests prompt change detection across categories: Advertising, Avoidance, Taco, Technical (Expected: 100% uniqueness)
4. **`exp_attack_resistance.py`** - Tests resistance against token manipulation, signature tampering, and other attacks (Expected: 100% resistance)

### ⚙️ System Tests  

5. **`exp_precision_comparison.py`** - Tests different precision modes (fp32, fp16, bf16) and their verification behavior
6. **`exp_performance_benchmark.py`** - Measures timing, throughput, and resource usage across configurations

### 📊 Error Sensitivity Analyses

7. **`exp_topk_exponent_error.py`** - Analyzes exponent mismatches across top-k values (64-4096) for Table 1 (Exponent mismatch across top-k values)
8. **`exp_mantissa_error_analysis.py`** - Tracks mantissa error degradation across token generation (0-2048 tokens) for Figure 2 (Mantissa error growth across generation) 
9. **`exp_topk_index_mismatch.py`** - Computes mismatch ratios of top-k indices between generation and validation for Figure 3 (Mismatch ratio vs top-k size)

## 🚀 Quick Start

### Run Individual Experiments
```bash
# Test correctness detection (CRITICAL)
uv run python experiments/exp_correctness_detection.py

# Test model resistance (CRITICAL) 
uv run python experiments/exp_model_resistance.py

# Test prompt tampering detection (CRITICAL)
uv run python experiments/exp_prompt_tampering.py

# Test attack resistance (CRITICAL)
uv run python experiments/exp_attack_resistance.py

# Test precision comparison
uv run python experiments/exp_precision_comparison.py

# Benchmark performance
uv run python experiments/exp_performance_benchmark.py

# Error sensitivity analyses
uv run python experiments/exp_topk_exponent_error.py
uv run python experiments/exp_mantissa_error_analysis.py
uv run python experiments/exp_topk_index_mismatch.py
```

### Run All Experiments
```bash
# Complete evaluation suite with comprehensive reporting
uv run python experiments/run_all_experiments.py
```

## 📊 What Each Experiment Tests

### `exp_correctness_detection.py`
**Goal: Prove valid configurations work (100% accuracy)**
- ✅ Same model + same prompt → Should PASS verification
- Tests across multiple models: tiny-gpt2, distilgpt2
- Tests across prompt types: base, advertising, avoidance, taco, technical
- **Success Criteria:** All valid combinations pass verification

### `exp_model_resistance.py` 
**Goal: Detect model substitution attacks (100% detection)**
- ✅ Different models → Should FAIL verification
- Tests all model pair combinations
- Measures detection accuracy and timing
- **Success Criteria:** All different model attempts are rejected

### `exp_prompt_tampering.py`
**Goal: Detect prompt changes across categories**
- 🔍 Analyzes token sequences for different prompt types
- Tests cross-verification between prompt categories
- Focuses on: **Advertising**, **Avoidance**, **Taco**, **Technical** prompts
- **Success Criteria:** Different prompts produce different token sequences

### `exp_attack_resistance.py`
**Goal: Resist sophisticated attacks (100% resistance)**
- 🔒 Token manipulation (modify, append, remove tokens)
- 🔐 Signature tampering (modify s_vals, wrong signatures)
- 🎲 Challenge manipulation (change indices)
- 🔄 Replay attacks (reuse proofs)
- **Success Criteria:** All attacks blocked or detected

### `exp_precision_comparison.py`
**Goal: Test precision mode behavior**
- ⚖️ Tests fp32, fp16, bf16 precision modes
- Compares verification behavior across precisions
- Hardware compatibility checking
- **Success Criteria:** At least one precision mode works

### `exp_performance_benchmark.py`
**Goal: Measure system performance**
- ⚡ Times commit, open, verify phases individually
- 🔄 End-to-end workflow benchmarking
- 📈 Scalability testing (token count, challenge size)
- 📊 Resource usage monitoring
- **Success Criteria:** Performance meets acceptability thresholds

### `exp_topk_exponent_error.py`
**Goal: Analyze exponent precision across top-k values**
- 📊 Varies top-k from 64 to 4096 across 2000 queries
- 🔢 Counts exact matches vs deviations for exponents
- 📈 Tests across 3+ models for comprehensive analysis
- **Success Criteria:** Generate Table 1 showing exponent mismatch patterns

### `exp_mantissa_error_analysis.py`
**Goal: Track mantissa error degradation during generation**
- 📉 Tracks mantissa differences over 0-2048 tokens  
- 📊 Measures mean, median, max error statistics
- 🕰️ Shows degradation effects from KV cache usage
- **Success Criteria:** Generate Figure 2 showing error growth trends

### `exp_topk_index_mismatch.py`
**Goal: Analyze top-k index consistency**
- 🔍 Computes mismatch ratios between generation and validation
- 📈 Tests across multiple top-k sizes and 3+ models
- 📊 Plots median and max mismatch error rates
- **Success Criteria:** Generate Figure 3 showing mismatch vs top-k trends

## 📈 Results and Analysis

### Results Storage
All experiments save detailed JSON results to `experiments/results/`:
```
experiments/results/
├── exp_correctness_detection.json
├── exp_model_resistance.json  
├── exp_prompt_tampering.json
├── exp_attack_resistance.json
├── exp_precision_comparison.json
├── exp_performance_benchmark.json
├── exp_topk_exponent_error.json
├── exp_mantissa_error_analysis.json
├── exp_topk_index_mismatch.json
└── experiment_suite_summary.json    # Comprehensive report
```

### Success Metrics
- **Correctness Detection:** 100% success rate for valid configurations
- **Model Resistance:** 100% detection rate for model substitution  
- **Prompt Tampering:** 100% uniqueness rate for different prompts
- **Attack Resistance:** 100% resistance rate against attacks
- **Performance:** Sub-10 second end-to-end verification

## 🎯 Critical vs Optional Tests

### 🔴 Critical Tests (Must Pass for Production)
1. `exp_correctness_detection.py`
2. `exp_model_resistance.py` 
3. `exp_prompt_tampering.py`
4. `exp_attack_resistance.py`

### 🟡 Optional Tests (Performance & Compatibility)
5. `exp_precision_comparison.py`
6. `exp_performance_benchmark.py`

### 🔬 Research Tests (Error Sensitivity Analysis)
7. `exp_topk_exponent_error.py`
8. `exp_mantissa_error_analysis.py`
9. `exp_topk_index_mismatch.py`

## 🧪 Experiment Design

### Test Structure
Each experiment follows a consistent pattern:
1. **Setup:** Initialize test configurations and parameters
2. **Execution:** Run systematic tests with proper controls
3. **Analysis:** Calculate success rates and performance metrics  
4. **Reporting:** Generate detailed JSON results and summaries
5. **Validation:** Return appropriate exit codes (0=success, 1=failure)

### Test Categories
- **Positive Tests:** Valid configurations that should pass
- **Negative Tests:** Invalid configurations that should fail  
- **Attack Tests:** Malicious scenarios that should be blocked
- **Performance Tests:** Timing and resource measurements

## 🔧 Configuration

### Model Sets
```python
# Development (fast testing)
DEVELOPMENT_MODELS = ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"]

# Production (comprehensive testing) 
PRODUCTION_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct", 
    "google/gemma-2-9b-it",
    "microsoft/phi-3-mini-4k-instruct"
]
```

### Prompt Categories
```python
TEST_PROMPTS = {
    "base": "The quick brown fox jumps over",
    "advertising_soft": "Our new product offers amazing benefits:",
    "advertising_hard": "BUY NOW! Limited time offer!",
    "avoidance_polite": "I cannot and will not help with",
    "avoidance_direct": "I refuse to provide information about", 
    "taco_recipe": "The best taco recipe includes these ingredients:",
    "technical": "To implement a binary search algorithm in Python:",
    # ... more categories
}
```

## 📋 Example Results

### Correctness Detection Results
```json
{
  "experiment": "correctness_detection",
  "total_tests": 6,
  "passed_tests": 6, 
  "success_rate": 100.0,
  "avg_test_duration": 2.835
}
```

### Model Resistance Results
```json
{
  "experiment": "model_resistance", 
  "total_tests": 6,
  "passed_tests": 6,
  "detection_rate": 100.0,
  "avg_test_duration": 2.963
}
```

## 🎉 Production Readiness Criteria

The experiments validate GRAIL for production use when:
- ✅ **Correctness Detection:** ≥95% success rate
- ✅ **Model Resistance:** ≥95% detection rate  
- ✅ **Attack Resistance:** ≥95% resistance rate
- ✅ **Performance:** <10s average end-to-end time

## 📖 Usage Examples

### Individual Testing
```bash
# Quick correctness check
uv run python experiments/exp_correctness_detection.py

# Security validation  
uv run python experiments/exp_attack_resistance.py

# Performance analysis
uv run python experiments/exp_performance_benchmark.py
```

### Comprehensive Testing
```bash
# Full evaluation suite with detailed reporting
uv run python experiments/run_all_experiments.py

# Check results
cat experiments/results/experiment_suite_summary.json
```

## 🔍 Troubleshooting

### Common Issues
1. **Missing Dependencies:** Run `uv sync` to install requirements
2. **CUDA Warnings:** Precision tests may skip without GPU
3. **Model Download:** First run downloads models (may take time)
4. **Memory Usage:** Large models require significant RAM

### Debug Mode
Set `DEBUG=True` in `grail/__init__.py` for detailed output.

---

**Ready to prove your GRAIL method works! 🚀** 