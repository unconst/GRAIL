#!/usr/bin/env python3
"""
Cross-Model Validation Experiment (Table 4)

Tests cross-model validation by generating with one model and validating with another.
- Shows all mismatched pairs fail (↑ error stats)  
- Shows all matched pairs pass (↓ error stats)

This validates that the GRAIL protocol correctly detects model substitution attacks.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import combinations, product

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from grail import GrailProtocol

class CrossModelValidationExperiment:
    """Cross-Model Validation Experiment for Table 4"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results = {
            "experiment": "cross_model_validation",
            "description": "Generate with one model, validate with another",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "models_tested": [],
            "test_results": {},
            "summary_statistics": {},
            "table_4_data": {},
            "conclusions": []
        }
        
        # Test configuration
        self.test_prompt = "The quick brown fox jumps over the lazy dog."
        self.token_count = 32
        self.challenge_size = 512
        
        # Model sets for testing
        self.development_models = [
            "sshleifer/tiny-gpt2",
            "distilgpt2", 
            "gpt2"
        ]
        
        self.production_models = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/phi-3-mini-4k-instruct",
            "google/gemma-2-9b-it"
        ]
        
        # Use development models for faster testing
        self.models = self.development_models
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        if self.debug and level == "DEBUG":
            print(f"[{timestamp}] [DEBUG] {message}")
            
    def test_model_pair(self, gen_model: str, val_model: str) -> Dict[str, Any]:
        """Test generation with one model and validation with another"""
        
        self.log(f"Testing: Generate with {gen_model}, validate with {val_model}")
        
        start_time = time.time()
        test_result = {
            "generation_model": gen_model,
            "validation_model": val_model,
            "is_same_model": gen_model == val_model,
            "expected_result": "pass" if gen_model == val_model else "fail",
            "actual_result": None,
            "validation_success": False,
            "error_stats": {},
            "timing": {},
            "error_message": None
        }
        
        try:
            # Generate with first model
            gen_start = time.time()
            gen_protocol = GrailProtocol(
                model_name=gen_model,
                max_new_tokens=self.token_count,
                challenge_size=self.challenge_size
            )
            
            # Commit phase
            commit_result = gen_protocol.commit(self.test_prompt)
            gen_time = time.time() - gen_start
            
            test_result["timing"]["generation_time"] = gen_time
            test_result["timing"]["commit_time"] = commit_result.get("timing", {}).get("commit_time", 0)
            
            # Validate with second model  
            val_start = time.time()
            val_protocol = GrailProtocol(
                model_name=val_model,
                max_new_tokens=self.token_count,
                challenge_size=self.challenge_size
            )
            
            # Verify with validation model
            verify_result = val_protocol.verify(
                prompt=self.test_prompt,
                response=commit_result["response"],
                proof=commit_result["proof"]
            )
            
            val_time = time.time() - val_start
            test_result["timing"]["validation_time"] = val_time
            test_result["timing"]["verify_time"] = verify_result.get("timing", {}).get("verify_time", 0)
            
            # Extract validation results
            test_result["validation_success"] = verify_result["valid"]
            test_result["actual_result"] = "pass" if verify_result["valid"] else "fail"
            
            # Extract error statistics
            if "error_stats" in verify_result:
                test_result["error_stats"] = verify_result["error_stats"]
            
            # Check if result matches expectation
            expected_pass = (gen_model == val_model)
            actual_pass = verify_result["valid"]
            test_result["result_correct"] = (expected_pass == actual_pass)
            
            if not test_result["result_correct"]:
                if expected_pass and not actual_pass:
                    test_result["error_message"] = "Same model pair failed validation unexpectedly"
                elif not expected_pass and actual_pass:
                    test_result["error_message"] = "Different model pair passed validation unexpectedly"
            
        except Exception as e:
            test_result["error_message"] = str(e)
            test_result["actual_result"] = "error"
            test_result["validation_success"] = False
            self.log(f"Error testing {gen_model} -> {val_model}: {e}", "ERROR")
        
        test_result["timing"]["total_time"] = time.time() - start_time
        
        # Log result
        status = "✅ PASS" if test_result.get("result_correct", False) else "❌ FAIL"
        self.log(f"  Result: {status} ({test_result['actual_result']}, expected {test_result['expected_result']})")
        
        return test_result
    
    def run_cross_validation_matrix(self) -> Dict[str, Any]:
        """Run cross-validation tests across all model pairs"""
        
        self.log("=== Running Cross-Model Validation Matrix ===")
        
        matrix_results = {
            "same_model_tests": [],
            "cross_model_tests": [],
            "all_tests": []
        }
        
        # Test all model pair combinations
        for gen_model, val_model in product(self.models, repeat=2):
            test_result = self.test_model_pair(gen_model, val_model)
            matrix_results["all_tests"].append(test_result)
            
            if gen_model == val_model:
                matrix_results["same_model_tests"].append(test_result)
            else:
                matrix_results["cross_model_tests"].append(test_result)
        
        return matrix_results
    
    def analyze_results(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-validation results and compute statistics"""
        
        self.log("=== Analyzing Cross-Validation Results ===")
        
        analysis = {
            "total_tests": len(matrix_results["all_tests"]),
            "same_model_tests": len(matrix_results["same_model_tests"]),
            "cross_model_tests": len(matrix_results["cross_model_tests"]),
            "same_model_stats": {},
            "cross_model_stats": {},
            "overall_stats": {}
        }
        
        # Analyze same-model tests (should all pass)
        same_model_correct = sum(1 for test in matrix_results["same_model_tests"] 
                                if test.get("result_correct", False))
        same_model_pass_rate = same_model_correct / len(matrix_results["same_model_tests"]) * 100
        
        analysis["same_model_stats"] = {
            "total": len(matrix_results["same_model_tests"]),
            "correct": same_model_correct,
            "pass_rate": same_model_pass_rate,
            "expected_pass_rate": 100.0
        }
        
        # Analyze cross-model tests (should all fail)  
        cross_model_correct = sum(1 for test in matrix_results["cross_model_tests"]
                                 if test.get("result_correct", False))
        cross_model_detection_rate = cross_model_correct / len(matrix_results["cross_model_tests"]) * 100
        
        analysis["cross_model_stats"] = {
            "total": len(matrix_results["cross_model_tests"]),
            "correct": cross_model_correct,
            "detection_rate": cross_model_detection_rate,
            "expected_detection_rate": 100.0
        }
        
        # Overall statistics
        total_correct = same_model_correct + cross_model_correct
        overall_accuracy = total_correct / analysis["total_tests"] * 100
        
        analysis["overall_stats"] = {
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "expected_accuracy": 100.0
        }
        
        # Calculate timing statistics
        all_times = [test["timing"]["total_time"] for test in matrix_results["all_tests"] 
                    if "timing" in test and "total_time" in test["timing"]]
        
        if all_times:
            analysis["timing_stats"] = {
                "min_time": min(all_times),
                "max_time": max(all_times),
                "avg_time": sum(all_times) / len(all_times),
                "total_time": sum(all_times)
            }
        
        return analysis
    
    def generate_table_4_data(self, matrix_results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted data for Table 4 in the research paper"""
        
        table_data = {
            "title": "Cross-Model Validation Results",
            "description": "Generate with one model, validate with another",
            "headers": ["Generation Model", "Validation Model", "Expected", "Actual", "Error Stats", "Time (s)"],
            "rows": [],
            "summary": {
                "same_model_pass_rate": analysis["same_model_stats"]["pass_rate"],
                "cross_model_detection_rate": analysis["cross_model_stats"]["detection_rate"],
                "overall_accuracy": analysis["overall_stats"]["overall_accuracy"]
            }
        }
        
        # Add rows for each test
        for test in matrix_results["all_tests"]:
            error_summary = ""
            if "error_stats" in test and test["error_stats"]:
                stats = test["error_stats"]
                if "exponent_mismatch_rate" in stats:
                    error_summary = f"exp: {stats['exponent_mismatch_rate']:.2%}"
                if "mantissa_mse" in stats:
                    if error_summary:
                        error_summary += f", mant: {stats['mantissa_mse']:.2e}"
                    else:
                        error_summary = f"mant: {stats['mantissa_mse']:.2e}"
            
            row = {
                "generation_model": test["generation_model"].split("/")[-1],  # Short name
                "validation_model": test["validation_model"].split("/")[-1],  # Short name  
                "expected": test["expected_result"],
                "actual": test["actual_result"],
                "error_stats": error_summary,
                "time": f"{test['timing']['total_time']:.2f}" if "timing" in test else "N/A"
            }
            table_data["rows"].append(row)
        
        return table_data
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete cross-model validation experiment"""
        
        self.log("🧪 Starting Cross-Model Validation Experiment")
        self.log(f"Testing {len(self.models)} models: {', '.join(self.models)}")
        self.log(f"Total test combinations: {len(self.models)}² = {len(self.models)**2}")
        
        start_time = time.time()
        
        try:
            # Run cross-validation matrix
            matrix_results = self.run_cross_validation_matrix()
            
            # Analyze results
            analysis = self.analyze_results(matrix_results)
            
            # Generate table data
            table_4_data = self.generate_table_4_data(matrix_results, analysis)
            
            # Update results
            self.results["models_tested"] = self.models
            self.results["test_results"] = matrix_results
            self.results["summary_statistics"] = analysis
            self.results["table_4_data"] = table_4_data
            
            # Generate conclusions
            conclusions = []
            
            same_model_rate = analysis["same_model_stats"]["pass_rate"]
            cross_model_rate = analysis["cross_model_stats"]["detection_rate"]
            overall_rate = analysis["overall_stats"]["overall_accuracy"]
            
            if same_model_rate >= 95.0:
                conclusions.append("✅ Same-model validation works correctly")
            else:
                conclusions.append(f"❌ Same-model validation failing ({same_model_rate:.1f}% vs 100% expected)")
            
            if cross_model_rate >= 95.0:
                conclusions.append("✅ Cross-model attacks correctly detected")
            else:
                conclusions.append(f"❌ Cross-model detection failing ({cross_model_rate:.1f}% vs 100% expected)")
            
            if overall_rate >= 95.0:
                conclusions.append("✅ Cross-model validation experiment PASSED")
            else:
                conclusions.append(f"❌ Cross-model validation experiment FAILED (accuracy: {overall_rate:.1f}%)")
            
            self.results["conclusions"] = conclusions
            
            # Log summary
            self.log("=== Cross-Model Validation Results ===")
            self.log(f"Same-model pass rate: {same_model_rate:.1f}%")
            self.log(f"Cross-model detection rate: {cross_model_rate:.1f}%")
            self.log(f"Overall accuracy: {overall_rate:.1f}%")
            
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
            output_file = f"experiments/results/exp_cross_model_validation.json"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved to: {output_file}")
        return output_file

def main():
    """Main execution function"""
    
    print("🧪 GRAIL Cross-Model Validation Experiment (Table 4)")
    print("=" * 60)
    
    # Run experiment
    experiment = CrossModelValidationExperiment(debug=True)
    results = experiment.run_experiment()
    
    # Save results
    output_file = experiment.save_results()
    
    # Print final status
    print("\n" + "=" * 60)
    if "error" not in results:
        overall_accuracy = results["summary_statistics"]["overall_stats"]["overall_accuracy"]
        if overall_accuracy >= 95.0:
            print("🎉 Cross-Model Validation Experiment: PASSED")
            exit_code = 0
        else:
            print("💥 Cross-Model Validation Experiment: FAILED")
            exit_code = 1
    else:
        print("💥 Cross-Model Validation Experiment: ERROR")
        exit_code = 1
    
    print(f"📊 Results saved to: {output_file}")
    
    # Print table preview
    if "table_4_data" in results:
        print("\n📋 Table 4 Preview:")
        table = results["table_4_data"]
        print(f"Same-model pass rate: {table['summary']['same_model_pass_rate']:.1f}%")
        print(f"Cross-model detection rate: {table['summary']['cross_model_detection_rate']:.1f}%")
        print(f"Overall accuracy: {table['summary']['overall_accuracy']:.1f}%")
    
    return exit_code

if __name__ == "__main__":
    exit(main()) 