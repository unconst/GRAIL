"""
GRAIL Evaluation Experiments Package

Comprehensive evaluation suite for testing GRAIL (Guaranteed Rollout Authenticity via Inference Ledger).

This package contains focused experiments testing different aspects of GRAIL:
- Correctness detection (valid configurations should pass)
- Model resistance (different models should be detected)
- Prompt tampering detection (changes across categories)
- Attack resistance (token manipulation, signature tampering)
- Precision comparison (fp32/fp16/bf16 testing)
- Performance benchmarks (timing and resource usage)

Usage:
    # Run all experiments
    python experiments/run_all_experiments.py
    
    # Run individual experiments
    python experiments/exp_correctness_detection.py
    python experiments/exp_model_resistance.py
    python experiments/exp_prompt_tampering.py
    python experiments/exp_attack_resistance.py
    python experiments/exp_precision_comparison.py
    python experiments/exp_performance_benchmark.py
"""

__version__ = "1.0.0"
__author__ = "GRAIL Research Team"

# List of available experiments
AVAILABLE_EXPERIMENTS = [
    "exp_correctness_detection",
    "exp_model_resistance", 
    "exp_prompt_tampering",
    "exp_attack_resistance",
    "exp_precision_comparison",
    "exp_performance_benchmark",
    "exp_topk_exponent_error",
    "exp_mantissa_error_analysis",
    "exp_topk_index_mismatch"
]

# Critical experiments (must pass for production readiness)
CRITICAL_EXPERIMENTS = [
    "exp_correctness_detection",
    "exp_model_resistance",
    "exp_prompt_tampering", 
    "exp_attack_resistance"
]

def get_experiment_info():
    """Get information about available experiments"""
    return {
        "total_experiments": len(AVAILABLE_EXPERIMENTS),
        "critical_experiments": len(CRITICAL_EXPERIMENTS),
        "available": AVAILABLE_EXPERIMENTS,
        "critical": CRITICAL_EXPERIMENTS
    } 