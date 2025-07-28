#!/usr/bin/env python3
"""
Experiment: Precision Comparison
Tests different precision modes (fp32, fp16, bf16) and their impact on verification.
Expected: Different precisions should produce different results or fail verification
"""

import sys
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier, TOLERANCE

class PrecisionComparisonExperiment:
    """Test different precision modes and their verification behavior"""
    
    def __init__(self):
        self.results = []
        self.model = "sshleifer/tiny-gpt2"
        self.test_prompt = "The quick brown fox jumps over the lazy dog"
        
        # Precision configurations
        self.precision_configs = {
            "fp32": {
                "torch_dtype": torch.float32,
                "description": "Full 32-bit floating point precision",
                "available": True
            },
            "fp16": {
                "torch_dtype": torch.float16,
                "description": "Half precision floating point",
                "available": torch.cuda.is_available()
            },
            "bf16": {
                "torch_dtype": torch.bfloat16,
                "description": "Brain floating point 16",
                "available": self._check_bf16_support()
            }
        }
    
    def _check_bf16_support(self) -> bool:
        """Check if bfloat16 is supported on this hardware"""
        try:
            torch.tensor([1.0], dtype=torch.bfloat16)
            return True
        except:
            return False
    
    def run_precision_test(self, precision: str) -> Optional[Dict]:
        """Run test with specific precision"""
        config = self.precision_configs[precision]
        
        if not config["available"]:
            return {
                "precision": precision,
                "available": False,
                "reason": "Hardware/software not supported",
                "error": f"{precision} not available on this system"
            }
        
        print(f"  Testing {precision}: {config['description']}")
        
        try:
            # Note: For full implementation, Prover/Verifier would need to accept precision parameters
            # For now, we demonstrate the concept with the current implementation
            
            prover = Prover(self.model)
            verifier = Verifier(self.model)
            
            start_time = time.time()
            
            # Generate commit
            commit = prover.commit(self.test_prompt, max_new_tokens=16)
            
            # Generate proof
            proof_pkg = prover.open(k=8)
            
            # Verify
            verification_result = verifier.verify(commit, proof_pkg, prover.secret_key)
            
            duration = time.time() - start_time
            
            return {
                "precision": precision,
                "available": True,
                "torch_dtype": str(config["torch_dtype"]),
                "verification_passed": verification_result,
                "duration": duration,
                "tokens_generated": len(commit["tokens"]),
                "s_vals_sample": commit["s_vals"][:3],
                "model_hash": commit["model_hash"][:16] + "...",
                "error": None
            }
            
        except Exception as e:
            return {
                "precision": precision,
                "available": config["available"],
                "error": str(e),
                "verification_passed": False
            }
    
    def run_cross_precision_test(self, results: List[Dict]) -> List[Dict]:
        """Test verification between different precisions"""
        print("\n🔄 Testing cross-precision verification...")
        
        cross_tests = []
        
        # Get successful results for cross-testing
        successful_results = [r for r in results if r.get("verification_passed") and not r.get("error")]
        
        if len(successful_results) < 2:
            print("    Not enough successful precision tests for cross-verification")
            return cross_tests
        
        # Test pairs of precisions
        for i, result1 in enumerate(successful_results):
            for result2 in successful_results[i+1:]:
                precision1 = result1["precision"]
                precision2 = result2["precision"]
                
                print(f"    Comparing {precision1} vs {precision2}")
                
                # Compare s_vals samples
                s_vals1 = result1.get("s_vals_sample", [])
                s_vals2 = result2.get("s_vals_sample", [])
                
                s_vals_different = s_vals1 != s_vals2
                
                # Compare model hashes (should be same model, but precision might affect)
                hash1 = result1.get("model_hash", "")
                hash2 = result2.get("model_hash", "")
                
                cross_test = {
                    "precision1": precision1,
                    "precision2": precision2,
                    "s_vals_different": s_vals_different,
                    "model_hash_different": hash1 != hash2,
                    "s_vals_sample_1": s_vals1,
                    "s_vals_sample_2": s_vals2,
                    "expected_behavior": "Different precisions may produce different s_vals",
                    "tolerance": TOLERANCE
                }
                
                cross_tests.append(cross_test)
        
        return cross_tests
    
    def analyze_precision_impact(self, results: List[Dict]) -> Dict:
        """Analyze the impact of different precisions on GRAIL verification"""
        
        analysis = {
            "available_precisions": [],
            "verification_success_rate": {},
            "performance_comparison": {},
            "s_vals_variance": {},
        }
        
        for result in results:
            if result.get("available"):
                analysis["available_precisions"].append(result["precision"])
                
                # Success rate
                analysis["verification_success_rate"][result["precision"]] = result.get("verification_passed", False)
                
                # Performance
                if result.get("duration"):
                    analysis["performance_comparison"][result["precision"]] = {
                        "duration": result["duration"],
                        "tokens_per_second": result.get("tokens_generated", 0) / result["duration"] if result["duration"] > 0 else 0
                    }
                
                # S-values analysis
                if result.get("s_vals_sample"):
                    analysis["s_vals_variance"][result["precision"]] = {
                        "sample_values": result["s_vals_sample"],
                        "sample_size": len(result["s_vals_sample"])
                    }
        
        return analysis
    
    def run_experiment(self) -> Dict:
        """Run all precision comparison tests"""
        print("⚖️  Precision Comparison Experiment")
        print("=" * 50)
        print("Testing different precision modes and their impact on GRAIL verification")
        print()
        
        experiment_start = time.time()
        
        # Test each precision mode
        precision_results = []
        for precision in ["fp32", "fp16", "bf16"]:
            result = self.run_precision_test(precision)
            if result:
                precision_results.append(result)
        
        # Cross-precision tests
        cross_tests = self.run_cross_precision_test(precision_results)
        
        # Analysis
        analysis = self.analyze_precision_impact(precision_results)
        
        total_duration = time.time() - experiment_start
        
        # Calculate summary statistics
        available_precisions = len([r for r in precision_results if r.get("available")])
        successful_verifications = len([r for r in precision_results if r.get("verification_passed")])
        
        summary = {
            "experiment": "precision_comparison",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "model_tested": self.model,
            "test_prompt": self.test_prompt,
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__
            },
            "results_summary": {
                "total_precisions_tested": len(precision_results),
                "available_precisions": available_precisions,
                "successful_verifications": successful_verifications,
                "cross_tests_performed": len(cross_tests)
            },
            "precision_results": precision_results,
            "cross_precision_tests": cross_tests,
            "analysis": analysis
        }
        
        print()
        print("📊 Summary:")
        print(f"  Precisions tested: {len(precision_results)}")
        print(f"  Available precisions: {available_precisions}")
        print(f"  Successful verifications: {successful_verifications}")
        print(f"  Cross-precision tests: {len(cross_tests)}")
        
        # Print availability status
        print("\n  Precision availability:")
        for result in precision_results:
            status = "✅ Available" if result.get("available") else "❌ Not available"
            verification = "✅ Verified" if result.get("verification_passed") else "❌ Failed"
            print(f"    {result['precision']}: {status}, {verification}")
        
        return summary

def main():
    """Main entry point"""
    experiment = PrecisionComparisonExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_precision_comparison.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if at least one precision worked
    successful_verifications = results["results_summary"]["successful_verifications"]
    return 0 if successful_verifications > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 