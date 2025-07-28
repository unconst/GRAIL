#!/usr/bin/env python3
"""
Experiment: Correctness Detection
Tests that identical model/prompt combinations correctly verify.
Expected: All tests should PASS (100% accuracy for valid configurations)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class CorrectnessDetectionExperiment:
    """Test that valid model/prompt combinations pass verification"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations
        self.test_configs = [
            # Model, Prompt, Description
            ("sshleifer/tiny-gpt2", "The quick brown fox jumps over", "Basic prompt"),
            ("sshleifer/tiny-gpt2", "Buy our amazing product now! Special offer:", "Advertising prompt"),
            ("sshleifer/tiny-gpt2", "I cannot and will not help with", "Avoidance prompt"),
            ("sshleifer/tiny-gpt2", "The best taco recipe includes these ingredients:", "Taco prompt"),
            ("distilgpt2", "The quick brown fox jumps over", "Basic prompt - DistilGPT2"),
            ("distilgpt2", "To implement a binary search algorithm in Python:", "Technical prompt"),
        ]
    
    def run_single_test(self, model_name: str, prompt: str, description: str) -> Dict:
        """Run a single correctness test"""
        print(f"  Testing: {model_name} + '{prompt[:30]}...'")
        
        start_time = time.time()
        
        try:
            # Create prover and verifier with same model
            prover = Prover(model_name)
            verifier = Verifier(model_name)
            
            # Commit phase
            commit = prover.commit(prompt, max_new_tokens=16)
            
            # Open phase
            proof_pkg = prover.open(k=8)
            
            # Verify phase
            result = verifier.verify(commit, proof_pkg, prover.secret_key)
            
            duration = time.time() - start_time
            
            return {
                "model": model_name,
                "prompt": prompt,
                "description": description,
                "expected": True,
                "actual": result,
                "passed": result == True,
                "duration": duration,
                "tokens_generated": len(commit["tokens"]),
                "error": None
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                "model": model_name,
                "prompt": prompt,
                "description": description,
                "expected": True,
                "actual": False,
                "passed": False,
                "duration": duration,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def run_experiment(self) -> Dict:
        """Run all correctness detection tests"""
        print("🧪 Correctness Detection Experiment")
        print("=" * 50)
        print("Testing that valid model/prompt combinations pass verification")
        print()
        
        experiment_start = time.time()
        
        for model, prompt, description in self.test_configs:
            result = self.run_single_test(model, prompt, description)
            self.results.append(result)
            
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"    {status} ({result['duration']:.2f}s) - {description}")
        
        total_duration = time.time() - experiment_start
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        avg_duration = sum(r["duration"] for r in self.results) / total_tests if total_tests > 0 else 0
        
        summary = {
            "experiment": "correctness_detection",
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "avg_test_duration": avg_duration,
            "results": self.results
        }
        
        print()
        print("📊 Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({summary['success_rate']:.1f}%)")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Average duration: {avg_duration:.3f}s")
        
        return summary

def main():
    """Main entry point"""
    experiment = CorrectnessDetectionExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_correctness_detection.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if all tests passed
    return 0 if results["failed_tests"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 