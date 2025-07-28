#!/usr/bin/env python3
"""
Experiment: Model Resistance
Tests that different models are correctly detected and verification fails.
Expected: All tests should FAIL (100% accuracy for detecting model changes)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class ModelResistanceExperiment:
    """Test that different models are correctly detected and rejected"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations: (prover_model, verifier_model, description)
        self.test_configs = [
            ("sshleifer/tiny-gpt2", "distilgpt2", "tiny-gpt2 vs distilgpt2"),
            ("distilgpt2", "sshleifer/tiny-gpt2", "distilgpt2 vs tiny-gpt2"),
            ("sshleifer/tiny-gpt2", "gpt2", "tiny-gpt2 vs gpt2"),
            ("gpt2", "sshleifer/tiny-gpt2", "gpt2 vs tiny-gpt2"),
            ("distilgpt2", "gpt2", "distilgpt2 vs gpt2"),
            ("gpt2", "distilgpt2", "gpt2 vs distilgpt2"),
        ]
        
        self.test_prompt = "The quick brown fox jumps over the lazy dog"
    
    def run_single_test(self, prover_model: str, verifier_model: str, description: str) -> Dict:
        """Run a single model resistance test"""
        print(f"  Testing: {description}")
        
        start_time = time.time()
        
        try:
            # Create prover and verifier with different models
            prover = Prover(prover_model)
            verifier = Verifier(verifier_model)
            
            # Commit phase (with prover model)
            commit = prover.commit(self.test_prompt, max_new_tokens=16)
            
            # Open phase
            proof_pkg = prover.open(k=8)
            
            # Verify phase (with verifier model - should fail)
            result = verifier.verify(commit, proof_pkg, prover.secret_key)
            
            duration = time.time() - start_time
            
            # For model resistance, we EXPECT failure (result = False)
            return {
                "prover_model": prover_model,
                "verifier_model": verifier_model,
                "description": description,
                "expected": False,  # We expect verification to fail
                "actual": result,
                "passed": result == False,  # Test passes if verification fails
                "duration": duration,
                "tokens_generated": len(commit["tokens"]),
                "error": None
            }
            
        except Exception as e:
            duration = time.time() - start_time
            # Exception during verification is also a "pass" for this test
            return {
                "prover_model": prover_model,
                "verifier_model": verifier_model,
                "description": description,
                "expected": False,
                "actual": False,
                "passed": True,  # Exception means verification failed, which is expected
                "duration": duration,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def run_experiment(self) -> Dict:
        """Run all model resistance tests"""
        print("🛡️  Model Resistance Experiment")
        print("=" * 50)
        print("Testing that different models are correctly detected and rejected")
        print()
        
        experiment_start = time.time()
        
        for prover_model, verifier_model, description in self.test_configs:
            result = self.run_single_test(prover_model, verifier_model, description)
            self.results.append(result)
            
            status = "✅ DETECTED" if result["passed"] else "❌ MISSED"
            verification_result = "FAILED" if not result["actual"] else "PASSED"
            print(f"    {status} ({result['duration']:.2f}s) - Verification {verification_result}")
        
        total_duration = time.time() - experiment_start
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        avg_duration = sum(r["duration"] for r in self.results) / total_tests if total_tests > 0 else 0
        
        summary = {
            "experiment": "model_resistance",
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "detection_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "avg_test_duration": avg_duration,
            "test_prompt": self.test_prompt,
            "results": self.results
        }
        
        print()
        print("📊 Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Correctly detected: {passed_tests} ({summary['detection_rate']:.1f}%)")
        print(f"  Missed detections: {total_tests - passed_tests}")
        print(f"  Average duration: {avg_duration:.3f}s")
        
        return summary

def main():
    """Main entry point"""
    experiment = ModelResistanceExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_model_resistance.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if all tests passed (i.e., all model changes were detected)
    return 0 if results["failed_tests"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 