#!/usr/bin/env python3
"""
GRAIL Experiments Runner
Executes all GRAIL evaluation experiments and generates comprehensive reports.
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Experiment modules
EXPERIMENTS = [
    {
        "name": "Correctness Detection",
        "file": "exp_correctness_detection.py",
        "description": "Tests that valid model/prompt combinations pass verification",
        "critical": True
    },
    {
        "name": "Model Resistance", 
        "file": "exp_model_resistance.py",
        "description": "Tests that different models are correctly detected and rejected",
        "critical": True
    },
    {
        "name": "Prompt Tampering",
        "file": "exp_prompt_tampering.py", 
        "description": "Tests prompt change detection across categories (Advertising, Avoidance, Taco)",
        "critical": True
    },
    {
        "name": "Attack Resistance",
        "file": "exp_attack_resistance.py",
        "description": "Tests resistance against token manipulation and signature tampering",
        "critical": True
    },
    {
        "name": "Precision Comparison",
        "file": "exp_precision_comparison.py",
        "description": "Tests different precision modes (fp32, fp16, bf16)",
        "critical": False
    },
    {
        "name": "Performance Benchmark",
        "file": "exp_performance_benchmark.py",
        "description": "Measures performance characteristics and resource usage",
        "critical": False
    },
    {
        "name": "Top-k Exponent Error",
        "file": "exp_topk_exponent_error.py",
        "description": "Analyzes exponent mismatches across top-k values (64-4096) for Table 1",
        "critical": False
    },
    {
        "name": "Mantissa Error Analysis",
        "file": "exp_mantissa_error_analysis.py", 
        "description": "Tracks mantissa error degradation across token generation (0-2048 tokens) for Figure 2",
        "critical": False
    },
    {
        "name": "Top-k Index Mismatch",
        "file": "exp_topk_index_mismatch.py",
        "description": "Computes mismatch ratios of top-k indices between generation and validation for Figure 3",
        "critical": False
    },
    {
        "name": "Cross-Model Validation",
        "file": "exp_cross_model_validation.py",
        "description": "Generate with one model, validate with another (Table 4)",
        "critical": True
    },
    {
        "name": "Enhanced Precision Sensitivity",
        "file": "exp_enhanced_precision_sensitivity.py", 
        "description": "Test precision mode combinations: bf16→bf16 pass, bf16→fp32 fail, fp32→fp32/bf16 pass (Table 5)",
        "critical": True
    },
    {
        "name": "GPU Robustness",
        "file": "exp_gpu_robustness.py",
        "description": "Test robustness across GPU configurations, tensor parallelism, and attention kernels (Table 2)",
        "critical": True
    },
    {
        "name": "Baseline Comparison",
        "file": "exp_baseline_comparison.py",
        "description": "Compare GRAIL vs TOPLOC, zkLLM, SVIP with performance metrics (Table 6)",
        "critical": True
    }
]

from experiments.exp_cross_model_validation import CrossModelValidationExperiment
from experiments.exp_enhanced_precision_sensitivity import EnhancedPrecisionSensitivityExperiment
from experiments.exp_gpu_robustness import GPURobustnessExperiment
from experiments.exp_baseline_comparison import BaselineComparisonExperiment

class ExperimentRunner:
    """Master experiment runner"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
        # Ensure results directory exists
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_experiment(self, experiment: Dict) -> Dict:
        """Run a single experiment"""
        print(f"🧪 Running: {experiment['name']}")
        print(f"   {experiment['description']}")
        
        experiment_start = time.time()
        
        try:
            # Run experiment as subprocess to isolate execution
            experiment_file = Path(__file__).parent / experiment["file"]
            
            if not experiment_file.exists():
                return {
                    "name": experiment["name"],
                    "file": experiment["file"],
                    "status": "error",
                    "error": f"Experiment file not found: {experiment_file}",
                    "duration": 0
                }
            
            # Execute experiment
            result = subprocess.run(
                [sys.executable, str(experiment_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - experiment_start
            
            if result.returncode == 0:
                status = "success"
                error = None
                print(f"   ✅ Completed successfully ({duration:.1f}s)")
            else:
                status = "failed"
                error = result.stderr or "Unknown error"
                print(f"   ❌ Failed ({duration:.1f}s): {error[:100]}...")
            
            return {
                "name": experiment["name"],
                "file": experiment["file"],
                "description": experiment["description"],
                "critical": experiment["critical"],
                "status": status,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": error
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - experiment_start
            print(f"   ⏰ Timeout after {duration:.1f}s")
            return {
                "name": experiment["name"],
                "file": experiment["file"],
                "status": "timeout",
                "duration": duration,
                "error": "Experiment timed out after 5 minutes"
            }
        
        except Exception as e:
            duration = time.time() - experiment_start
            print(f"   💥 Exception: {str(e)}")
            return {
                "name": experiment["name"],
                "file": experiment["file"],
                "status": "exception",
                "duration": duration,
                "error": str(e)
            }
    
    def load_experiment_results(self) -> Dict[str, Any]:
        """Load detailed results from individual experiment files"""
        detailed_results = {}
        
        for experiment in EXPERIMENTS:
            result_file = self.results_dir / f"{experiment['file'].replace('.py', '.json')}"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        detailed_results[experiment['name']] = json.load(f)
                except Exception as e:
                    detailed_results[experiment['name']] = {"error": f"Failed to load results: {e}"}
        
        return detailed_results
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        
        total_experiments = len(self.results)
        successful = len([r for r in self.results if r["status"] == "success"])
        failed = len([r for r in self.results if r["status"] in ["failed", "timeout", "exception"]])
        
        critical_experiments = [r for r in self.results if r.get("critical", False)]
        critical_successful = len([r for r in critical_experiments if r["status"] == "success"])
        
        total_duration = time.time() - self.start_time
        
        # Load detailed experiment results
        detailed_results = self.load_experiment_results()
        
        # Analyze key metrics from detailed results
        analysis = self.analyze_experiment_results(detailed_results)
        
        summary = {
            "experiment_suite": "GRAIL_Evaluation_Suite",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "overview": {
                "total_experiments": total_experiments,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total_experiments * 100) if total_experiments > 0 else 0,
                "critical_experiments": len(critical_experiments),
                "critical_successful": critical_successful,
                "critical_success_rate": (critical_successful / len(critical_experiments) * 100) if critical_experiments else 0
            },
            "experiment_results": self.results,
            "detailed_analysis": analysis,
            "detailed_results": detailed_results
        }
        
        return summary
    
    def analyze_experiment_results(self, detailed_results: Dict) -> Dict:
        """Analyze key metrics from detailed experiment results"""
        analysis = {
            "correctness_metrics": {},
            "security_metrics": {},
            "performance_metrics": {},
            "overall_assessment": {}
        }
        
        # Correctness Detection Analysis
        if "Correctness Detection" in detailed_results:
            correctness = detailed_results["Correctness Detection"]
            analysis["correctness_metrics"] = {
                "success_rate": correctness.get("success_rate", 0),
                "total_tests": correctness.get("total_tests", 0),
                "passed_tests": correctness.get("passed_tests", 0),
                "avg_duration": correctness.get("avg_test_duration", 0)
            }
        
        # Model Resistance Analysis
        if "Model Resistance" in detailed_results:
            resistance = detailed_results["Model Resistance"]
            analysis["security_metrics"]["model_detection_rate"] = resistance.get("detection_rate", 0)
        
        # Attack Resistance Analysis
        if "Attack Resistance" in detailed_results:
            attacks = detailed_results["Attack Resistance"]
            if "overall_stats" in attacks:
                analysis["security_metrics"]["attack_resistance_rate"] = attacks["overall_stats"].get("resistance_rate", 0)
        
        # Prompt Tampering Analysis
        if "Prompt Tampering" in detailed_results:
            prompts = detailed_results["Prompt Tampering"]
            if "cross_verification" in prompts:
                analysis["security_metrics"]["prompt_detection_rate"] = prompts["cross_verification"].get("detection_rate", 0)
        
        # Performance Analysis
        if "Performance Benchmark" in detailed_results:
            perf = detailed_results["Performance Benchmark"]
            if "performance_summary" in perf:
                analysis["performance_metrics"] = {
                    "avg_commit_time": perf["performance_summary"].get("avg_commit_time", 0),
                    "avg_verify_time": perf["performance_summary"].get("avg_verify_time", 0),
                    "avg_e2e_time": perf["performance_summary"].get("avg_e2e_time", 0)
                }
        
        # Overall Assessment
        correctness_rate = analysis["correctness_metrics"].get("success_rate", 0)
        model_detection = analysis["security_metrics"].get("model_detection_rate", 0)
        attack_resistance = analysis["security_metrics"].get("attack_resistance_rate", 0)
        
        analysis["overall_assessment"] = {
            "grail_functional": correctness_rate >= 95.0,
            "security_robust": model_detection >= 95.0 and attack_resistance >= 95.0,
            "performance_acceptable": analysis["performance_metrics"].get("avg_e2e_time", 999) < 10.0,
            "ready_for_production": (
                correctness_rate >= 95.0 and
                model_detection >= 95.0 and 
                attack_resistance >= 95.0
            )
        }
        
        return analysis
    
    def print_summary_report(self, summary: Dict):
        """Print human-readable summary report"""
        print("\n" + "="*80)
        print("🎯 GRAIL EVALUATION SUITE - FINAL REPORT")
        print("="*80)
        
        overview = summary["overview"]
        analysis = summary["detailed_analysis"]
        
        print(f"\n📊 EXECUTION OVERVIEW")
        print(f"   Total Experiments: {overview['total_experiments']}")
        print(f"   Successful: {overview['successful']} ({overview['success_rate']:.1f}%)")
        print(f"   Failed: {overview['failed']}")
        print(f"   Critical Success Rate: {overview['critical_success_rate']:.1f}%")
        print(f"   Total Duration: {summary['total_duration']:.1f}s")
        
        print(f"\n🎯 CORRECTNESS DETECTION")
        correctness = analysis.get("correctness_metrics", {})
        print(f"   Success Rate: {correctness.get('success_rate', 0):.1f}%")
        print(f"   Tests: {correctness.get('passed_tests', 0)}/{correctness.get('total_tests', 0)} passed")
        print(f"   Average Duration: {correctness.get('avg_duration', 0):.3f}s")
        
        print(f"\n🛡️  SECURITY METRICS")
        security = analysis.get("security_metrics", {})
        print(f"   Model Detection Rate: {security.get('model_detection_rate', 0):.1f}%")
        print(f"   Attack Resistance Rate: {security.get('attack_resistance_rate', 0):.1f}%")
        print(f"   Prompt Detection Rate: {security.get('prompt_detection_rate', 0):.1f}%")
        
        print(f"\n⚡ PERFORMANCE METRICS")
        performance = analysis.get("performance_metrics", {})
        print(f"   Average Commit Time: {performance.get('avg_commit_time', 0):.3f}s")
        print(f"   Average Verify Time: {performance.get('avg_verify_time', 0):.3f}s")
        print(f"   Average End-to-End: {performance.get('avg_e2e_time', 0):.3f}s")
        
        print(f"\n✅ OVERALL ASSESSMENT")
        assessment = analysis.get("overall_assessment", {})
        status_icon = lambda x: "✅" if x else "❌"
        print(f"   GRAIL Functional: {status_icon(assessment.get('grail_functional', False))}")
        print(f"   Security Robust: {status_icon(assessment.get('security_robust', False))}")
        print(f"   Performance Acceptable: {status_icon(assessment.get('performance_acceptable', False))}")
        print(f"   Ready for Production: {status_icon(assessment.get('ready_for_production', False))}")
        
        print(f"\n📝 EXPERIMENT DETAILS")
        for result in summary["experiment_results"]:
            status_icon = {"success": "✅", "failed": "❌", "timeout": "⏰", "exception": "💥"}.get(result["status"], "❓")
            critical_marker = " [CRITICAL]" if result.get("critical") else ""
            print(f"   {status_icon} {result['name']}{critical_marker} ({result['duration']:.1f}s)")
            if result["status"] != "success" and result.get("error"):
                print(f"      Error: {result['error'][:100]}...")
    
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("🚀 Starting GRAIL Evaluation Suite")
        print("="*60)
        print(f"Running {len(EXPERIMENTS)} experiments...")
        print()
        
        # Run each experiment
        for experiment in EXPERIMENTS:
            result = self.run_experiment(experiment)
            self.results.append(result)
            print()  # Spacing between experiments
        
        # Generate and save comprehensive report
        summary = self.generate_summary_report()
        
        # Save summary report
        summary_file = self.results_dir / "experiment_suite_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self.print_summary_report(summary)
        
        print(f"\n💾 Complete results saved to: {self.results_dir}")
        print(f"📊 Summary report: {summary_file}")
        
        # Return appropriate exit code
        critical_failed = any(r["status"] != "success" and r.get("critical", False) for r in self.results)
        return 1 if critical_failed else 0

def main():
    """Main entry point"""
    runner = ExperimentRunner()
    exit_code = runner.run_all_experiments()
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 