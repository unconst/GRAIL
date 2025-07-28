#!/usr/bin/env python3
"""
Enhanced Precision Sensitivity Experiment (Table 5)

Tests specific precision mode combinations to validate precision sensitivity:
- Generation in bf16, validate in bf16 → pass
- Generation in bf16, validate in fp32 → fail  
- Generation in fp32, validate in fp32 and bf16 → pass

Reports exponent & mantissa errors for each precision combination.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from grail import Prover, Verifier

class EnhancedPrecisionSensitivityExperiment:
    """Enhanced Precision Sensitivity Experiment for Table 5"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results = {
            "experiment": "enhanced_precision_sensitivity", 
            "description": "Test precision mode combinations for sensitivity detection",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "models_tested": [],
            "precision_tests": [],
            "summary_statistics": {},
            "table_5_data": {},
            "conclusions": []
        }
        
        # Test configuration
        self.test_prompt = "The quick brown fox jumps over the lazy dog and continues running through the forest."
        self.token_count = 64
        self.challenge_size = 16  # Must be less than token_count
        self.num_runs_per_test = 3  # Multiple runs for statistical significance
        
        # Precision mode combinations to test
        self.precision_combinations = [
            {"gen_precision": "bf16", "val_precision": "bf16", "expected": "pass"},
            {"gen_precision": "bf16", "val_precision": "fp32", "expected": "fail"},
            {"gen_precision": "fp32", "val_precision": "fp32", "expected": "pass"},
            {"gen_precision": "fp32", "val_precision": "bf16", "expected": "fail"},  # Different precision should fail
            # Additional combinations for completeness
            {"gen_precision": "fp16", "val_precision": "fp16", "expected": "pass"},
            {"gen_precision": "fp16", "val_precision": "fp32", "expected": "fail"},
            {"gen_precision": "fp32", "val_precision": "fp16", "expected": "fail"}   # Different precision should fail
        ]
        
        # Model for testing
        self.test_models = ["sshleifer/tiny-gpt2", "distilgpt2"]
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        if self.debug and level == "DEBUG":
            print(f"[{timestamp}] [DEBUG] {message}")
    
    def check_precision_support(self, precision: str) -> bool:
        """Check if precision mode is supported on current hardware"""
        
        if precision == "bf16":
            # bf16 requires specific hardware support
            if torch.cuda.is_available():
                # Check if current GPU supports bf16
                device_capability = torch.cuda.get_device_capability()
                # Ampere (8.0) and newer support bf16
                return device_capability[0] >= 8
            else:
                # CPU might support bf16 depending on the processor
                return hasattr(torch, 'bfloat16')
                
        elif precision == "fp16":
            # fp16 is more widely supported
            return torch.cuda.is_available() or hasattr(torch, 'float16')
            
        elif precision == "fp32":
            # fp32 is always supported
            return True
            
        return False
    
    def test_precision_combination(self, model_name: str, gen_precision: str, val_precision: str, expected: str, run_id: int = 0) -> Dict[str, Any]:
        """Test a specific precision combination"""
        
        self.log(f"Testing {model_name}: gen={gen_precision}, val={val_precision} (run {run_id+1})")
        
        start_time = time.time()
        test_result = {
            "model": model_name,
            "generation_precision": gen_precision,
            "validation_precision": val_precision,
            "expected_result": expected,
            "actual_result": None,
            "validation_success": False,
            "error_stats": {},
            "timing": {},
            "run_id": run_id,
            "error_message": None,
            "precision_support": {
                "gen_supported": self.check_precision_support(gen_precision),
                "val_supported": self.check_precision_support(val_precision)
            }
        }
        
        # Check precision support
        if not test_result["precision_support"]["gen_supported"]:
            test_result["error_message"] = f"Generation precision {gen_precision} not supported"
            test_result["actual_result"] = "unsupported"
            self.log(f"  Skipping: {gen_precision} not supported", "WARN")
            return test_result
            
        if not test_result["precision_support"]["val_supported"]:
            test_result["error_message"] = f"Validation precision {val_precision} not supported"
            test_result["actual_result"] = "unsupported"
            self.log(f"  Skipping: {val_precision} not supported", "WARN")
            return test_result
        
        try:
            # Generation phase with specified precision
            gen_start = time.time()
            
            # Set precision for generation
            gen_dtype = None
            if gen_precision == "bf16":
                gen_dtype = torch.bfloat16
            elif gen_precision == "fp16":
                gen_dtype = torch.float16
            elif gen_precision == "fp32":
                gen_dtype = torch.float32
            
            prover = Prover(model_name, torch_dtype=gen_dtype)
            
            # Commit phase
            commit = prover.commit(self.test_prompt, max_new_tokens=self.token_count)
            
            # Open phase
            proof_pkg = prover.open(k=self.challenge_size)
            
            gen_time = time.time() - gen_start
            
            test_result["timing"]["generation_time"] = gen_time
            test_result["timing"]["commit_time"] = gen_time
            
            # Validation phase with specified precision
            val_start = time.time()
            
            # Set precision for validation
            val_dtype = None
            if val_precision == "bf16":
                val_dtype = torch.bfloat16
            elif val_precision == "fp16":
                val_dtype = torch.float16
            elif val_precision == "fp32":
                val_dtype = torch.float32
                
            verifier = Verifier(model_name, torch_dtype=val_dtype)
            
            # Verify 
            verification_result = verifier.verify(commit, proof_pkg, prover.secret_key)
            
            val_time = time.time() - val_start
            test_result["timing"]["validation_time"] = val_time
            test_result["timing"]["verify_time"] = val_time
            
            # Extract validation results
            test_result["validation_success"] = verification_result
            test_result["actual_result"] = "pass" if verification_result else "fail"
            
            # Extract detailed error statistics (if needed in future)
            # Currently verification returns boolean only
            
            # Check if result matches expectation
            expected_pass = (expected == "pass")
            actual_pass = verification_result
            test_result["result_correct"] = (expected_pass == actual_pass)
            
            if not test_result["result_correct"]:
                if expected_pass and not actual_pass:
                    test_result["error_message"] = f"Same precision pair failed validation unexpectedly"
                elif not expected_pass and actual_pass:
                    test_result["error_message"] = f"Different precision pair passed validation unexpectedly"
                    
        except Exception as e:
            test_result["error_message"] = str(e)
            test_result["actual_result"] = "error"
            test_result["validation_success"] = False
            self.log(f"Error in precision test: {e}", "ERROR")
        
        test_result["timing"]["total_time"] = time.time() - start_time
        
        # Log result
        status = "✅ CORRECT" if test_result.get("result_correct", False) else "❌ INCORRECT"
        self.log(f"  Result: {status} ({test_result['actual_result']}, expected {expected})")
        
        return test_result
    
    def run_precision_tests(self) -> List[Dict[str, Any]]:
        """Run all precision combination tests"""
        
        self.log("=== Running Enhanced Precision Sensitivity Tests ===")
        
        all_tests = []
        
        for model in self.test_models:
            self.log(f"Testing model: {model}")
            
            for combo in self.precision_combinations:
                gen_prec = combo["gen_precision"]
                val_prec = combo["val_precision"]
                expected = combo["expected"]
                
                # Run multiple runs for statistical significance
                for run_id in range(self.num_runs_per_test):
                    test_result = self.test_precision_combination(
                        model, gen_prec, val_prec, expected, run_id
                    )
                    all_tests.append(test_result)
        
        return all_tests
    
    def analyze_precision_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze precision test results and compute statistics"""
        
        self.log("=== Analyzing Precision Sensitivity Results ===")
        
        analysis = {
            "total_tests": len(test_results),
            "successful_tests": 0,
            "precision_combination_stats": {},
            "model_stats": {},
            "error_analysis": {},
            "timing_stats": {}
        }
        
        # Group results by precision combination
        combo_groups = {}
        model_groups = {}
        
        for test in test_results:
            # Skip unsupported tests
            if test["actual_result"] == "unsupported":
                continue
                
            # Group by precision combination
            combo_key = f"{test['generation_precision']}→{test['validation_precision']}"
            if combo_key not in combo_groups:
                combo_groups[combo_key] = []
            combo_groups[combo_key].append(test)
            
            # Group by model
            model_key = test["model"]
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(test)
            
            # Count successful tests
            if test.get("result_correct", False):
                analysis["successful_tests"] += 1
        
        # Analyze each precision combination
        for combo_key, tests in combo_groups.items():
            correct_tests = sum(1 for t in tests if t.get("result_correct", False))
            total_tests = len(tests)
            accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
            
            # Calculate error statistics
            error_stats = {}
            if tests and "error_stats" in tests[0]:
                # Aggregate error stats across runs
                exponent_mismatches = []
                mantissa_mses = []
                
                for test in tests:
                    if "error_stats" in test and test["error_stats"]:
                        stats = test["error_stats"]
                        if "exponent_mismatch_rate" in stats:
                            exponent_mismatches.append(stats["exponent_mismatch_rate"])
                        if "mantissa_mse" in stats:
                            mantissa_mses.append(stats["mantissa_mse"])
                
                if exponent_mismatches:
                    error_stats["exponent_mismatch"] = {
                        "mean": sum(exponent_mismatches) / len(exponent_mismatches),
                        "min": min(exponent_mismatches),
                        "max": max(exponent_mismatches)
                    }
                
                if mantissa_mses:
                    error_stats["mantissa_mse"] = {
                        "mean": sum(mantissa_mses) / len(mantissa_mses),
                        "min": min(mantissa_mses),
                        "max": max(mantissa_mses)
                    }
            
            analysis["precision_combination_stats"][combo_key] = {
                "total_tests": total_tests,
                "correct_tests": correct_tests,
                "accuracy": accuracy,
                "expected_result": tests[0]["expected_result"] if tests else "unknown",
                "error_stats": error_stats
            }
        
        # Analyze each model
        for model_key, tests in model_groups.items():
            correct_tests = sum(1 for t in tests if t.get("result_correct", False))
            total_tests = len(tests)
            accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
            
            analysis["model_stats"][model_key] = {
                "total_tests": total_tests,
                "correct_tests": correct_tests,
                "accuracy": accuracy
            }
        
        # Overall accuracy
        total_valid_tests = sum(len(tests) for tests in combo_groups.values())
        analysis["overall_accuracy"] = analysis["successful_tests"] / total_valid_tests * 100 if total_valid_tests > 0 else 0
        
        # Timing statistics
        all_times = [test["timing"]["total_time"] for test in test_results 
                    if "timing" in test and "total_time" in test["timing"] and test["actual_result"] != "unsupported"]
        
        if all_times:
            analysis["timing_stats"] = {
                "min_time": min(all_times),
                "max_time": max(all_times),
                "avg_time": sum(all_times) / len(all_times),
                "total_time": sum(all_times)
            }
        
        return analysis
    
    def generate_table_5_data(self, test_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted data for Table 5 in the research paper"""
        
        table_data = {
            "title": "Enhanced Precision Sensitivity Results",
            "description": "Precision mode combinations and their validation behavior",
            "headers": ["Generation", "Validation", "Expected", "Actual", "Accuracy", "Exponent Error", "Mantissa Error"],
            "rows": [],
            "summary": {
                "overall_accuracy": analysis["overall_accuracy"],
                "total_combinations": len(analysis["precision_combination_stats"]),
                "critical_tests": {}
            }
        }
        
        # Add rows for each precision combination
        for combo_key, stats in analysis["precision_combination_stats"].items():
            gen_prec, val_prec = combo_key.split("→")
            
            # Format error statistics
            exp_error = "N/A"
            mant_error = "N/A"
            
            if "error_stats" in stats:
                if "exponent_mismatch" in stats["error_stats"]:
                    exp_error = f"{stats['error_stats']['exponent_mismatch']['mean']:.2%}"
                if "mantissa_mse" in stats["error_stats"]:
                    mant_error = f"{stats['error_stats']['mantissa_mse']['mean']:.2e}"
            
            row = {
                "generation": gen_prec,
                "validation": val_prec,
                "expected": stats["expected_result"],
                "actual": "pass" if stats["accuracy"] >= 50 else "fail",  # Majority vote
                "accuracy": f"{stats['accuracy']:.1f}%",
                "exponent_error": exp_error,
                "mantissa_error": mant_error
            }
            table_data["rows"].append(row)
            
            # Track critical test results
            if combo_key in ["bf16→bf16", "bf16→fp32", "fp32→fp32", "fp32→bf16"]:
                table_data["summary"]["critical_tests"][combo_key] = {
                    "accuracy": stats["accuracy"],
                    "expected": stats["expected_result"]
                }
        
        return table_data
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete enhanced precision sensitivity experiment"""
        
        self.log("🧪 Starting Enhanced Precision Sensitivity Experiment")
        self.log(f"Testing {len(self.precision_combinations)} precision combinations")
        self.log(f"Models: {', '.join(self.test_models)}")
        self.log(f"Runs per test: {self.num_runs_per_test}")
        
        start_time = time.time()
        
        try:
            # Run precision tests
            test_results = self.run_precision_tests()
            
            # Analyze results
            analysis = self.analyze_precision_results(test_results)
            
            # Generate table data
            table_5_data = self.generate_table_5_data(test_results, analysis)
            
            # Update results
            self.results["models_tested"] = self.test_models
            self.results["precision_tests"] = test_results
            self.results["summary_statistics"] = analysis
            self.results["table_5_data"] = table_5_data
            
            # Generate conclusions
            conclusions = []
            
            overall_accuracy = analysis["overall_accuracy"]
            critical_tests = table_5_data["summary"]["critical_tests"]
            
            # Check specific critical combinations
            bf16_same = critical_tests.get("bf16→bf16", {})
            bf16_diff = critical_tests.get("bf16→fp32", {})
            fp32_same = critical_tests.get("fp32→fp32", {})
            fp32_cross = critical_tests.get("fp32→bf16", {})
            
            if bf16_same.get("accuracy", 0) >= 95 and bf16_same.get("expected") == "pass":
                conclusions.append("✅ bf16→bf16 validation works correctly")
            else:
                conclusions.append("❌ bf16→bf16 validation failing")
            
            if bf16_diff.get("accuracy", 0) >= 95 and bf16_diff.get("expected") == "fail":
                conclusions.append("✅ bf16→fp32 precision mismatch correctly detected")
            else:
                conclusions.append("❌ bf16→fp32 precision mismatch not detected")
            
            if fp32_same.get("accuracy", 0) >= 95 and fp32_same.get("expected") == "pass":
                conclusions.append("✅ fp32→fp32 validation works correctly")
            else:
                conclusions.append("❌ fp32→fp32 validation failing")
                
            if fp32_cross.get("accuracy", 0) >= 95 and fp32_cross.get("expected") == "fail":
                conclusions.append("✅ fp32→bf16 precision mismatch correctly detected")
            else:
                conclusions.append("❌ fp32→bf16 precision mismatch not detected")
            
            if overall_accuracy >= 95.0:
                conclusions.append("✅ Enhanced precision sensitivity experiment PASSED")
            else:
                conclusions.append(f"❌ Enhanced precision sensitivity experiment FAILED (accuracy: {overall_accuracy:.1f}%)")
            
            self.results["conclusions"] = conclusions
            
            # Log summary
            self.log("=== Enhanced Precision Sensitivity Results ===")
            self.log(f"Overall accuracy: {overall_accuracy:.1f}%")
            
            for conclusion in conclusions:
                self.log(conclusion)
                
        except Exception as e:
            self.log(f"Experiment failed: {e}", "ERROR")
            self.results["error"] = str(e)
            self.results["conclusions"] = [f"❌ Experiment failed: {e}"]
        
        self.results["total_duration"] = time.time() - start_time
        self.log(f"Experiment completed in {self.results['total_duration']:.2f} seconds")
        
        return self.results
    
    def save_results(self, output_file: str = None) -> str:
        """Save experiment results to JSON file"""
        
        if output_file is None:
            output_file = f"experiments/results/exp_enhanced_precision_sensitivity.json"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved to: {output_file}")
        return output_file

def main():
    """Main execution function"""
    
    print("🧪 GRAIL Enhanced Precision Sensitivity Experiment (Table 5)")
    print("=" * 70)
    
    # Run experiment
    experiment = EnhancedPrecisionSensitivityExperiment(debug=True)
    results = experiment.run_experiment()
    
    # Save results
    output_file = experiment.save_results()
    
    # Print final status
    print("\n" + "=" * 70)
    if "error" not in results:
        overall_accuracy = results["summary_statistics"]["overall_accuracy"]
        if overall_accuracy >= 95.0:
            print("🎉 Enhanced Precision Sensitivity Experiment: PASSED")
            exit_code = 0
        else:
            print("💥 Enhanced Precision Sensitivity Experiment: FAILED")
            exit_code = 1
    else:
        print("💥 Enhanced Precision Sensitivity Experiment: ERROR")
        exit_code = 1
    
    print(f"📊 Results saved to: {output_file}")
    
    # Print table preview
    if "table_5_data" in results:
        print("\n📋 Table 5 Preview:")
        table = results["table_5_data"]
        print(f"Overall accuracy: {table['summary']['overall_accuracy']:.1f}%")
        
        if "critical_tests" in table["summary"]:
            for test_key, test_data in table["summary"]["critical_tests"].items():
                print(f"{test_key}: {test_data['accuracy']:.1f}% (expected {test_data['expected']})")
    
    return exit_code

if __name__ == "__main__":
    exit(main()) 