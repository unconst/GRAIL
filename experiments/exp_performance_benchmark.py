#!/usr/bin/env python3
"""
Experiment: Performance Benchmark
Measures performance characteristics of GRAIL across different configurations.
Tests timing, throughput, and resource usage.
"""

import sys
import time
import json
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, median, stdev

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class PerformanceBenchmarkExperiment:
    """Benchmark GRAIL performance across different scenarios"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations
        self.models = ["sshleifer/tiny-gpt2", "distilgpt2"]
        self.prompts = {
            "short": "Hello world",
            "medium": "The quick brown fox jumps over the lazy dog",
            "long": "This is a longer prompt to test performance with more complex inputs that require more tokens and processing time to evaluate",
        }
        self.token_counts = [8, 16, 32]
        self.challenge_sizes = [4, 8, 16]
    
    def get_system_metrics(self) -> Dict:
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "process_memory_mb": psutil.Process().memory_info().rss / (1024**2)
        }
    
    def benchmark_commit_phase(self, model: str, prompt: str, max_tokens: int, rounds: int = 3) -> Dict:
        """Benchmark the commit phase"""
        print(f"    Commit: {model}, {max_tokens} tokens")
        
        timings = []
        token_counts = []
        s_vals_counts = []
        
        for round_num in range(rounds):
            prover = Prover(model)
            
            start_time = time.time()
            commit = prover.commit(prompt, max_new_tokens=max_tokens)
            duration = time.time() - start_time
            
            timings.append(duration)
            token_counts.append(len(commit["tokens"]))
            s_vals_counts.append(len(commit["s_vals"]))
        
        return {
            "phase": "commit",
            "model": model,
            "max_tokens": max_tokens,
            "rounds": rounds,
            "timings": timings,
            "avg_duration": mean(timings),
            "median_duration": median(timings),
            "std_duration": stdev(timings) if len(timings) > 1 else 0,
            "avg_tokens": mean(token_counts),
            "avg_s_vals": mean(s_vals_counts),
            "tokens_per_second": mean(token_counts) / mean(timings) if mean(timings) > 0 else 0
        }
    
    def benchmark_open_phase(self, model: str, prompt: str, max_tokens: int, k: int, rounds: int = 3) -> Dict:
        """Benchmark the open phase"""
        print(f"    Open: {model}, k={k}")
        
        timings = []
        
        # Pre-generate commits for consistent testing
        prover = Prover(model)
        commits = []
        for _ in range(rounds):
            commit = prover.commit(prompt, max_new_tokens=max_tokens)
            commits.append(commit)
        
        for commit in commits:
            start_time = time.time()
            proof_pkg = prover.open(k=k)
            duration = time.time() - start_time
            timings.append(duration)
        
        return {
            "phase": "open",
            "model": model,
            "challenge_size": k,
            "rounds": rounds,
            "timings": timings,
            "avg_duration": mean(timings),
            "median_duration": median(timings),
            "std_duration": stdev(timings) if len(timings) > 1 else 0,
            "challenges_per_second": k / mean(timings) if mean(timings) > 0 else 0
        }
    
    def benchmark_verify_phase(self, model: str, prompt: str, max_tokens: int, k: int, rounds: int = 3) -> Dict:
        """Benchmark the verify phase"""
        print(f"    Verify: {model}, k={k}")
        
        timings = []
        success_count = 0
        
        for round_num in range(rounds):
            prover = Prover(model)
            verifier = Verifier(model)
            
            # Generate proof
            commit = prover.commit(prompt, max_new_tokens=max_tokens)
            proof_pkg = prover.open(k=k)
            
            # Benchmark verification
            start_time = time.time()
            result = verifier.verify(commit, proof_pkg, prover.secret_key)
            duration = time.time() - start_time
            
            timings.append(duration)
            if result:
                success_count += 1
        
        return {
            "phase": "verify",
            "model": model,
            "challenge_size": k,
            "rounds": rounds,
            "timings": timings,
            "avg_duration": mean(timings),
            "median_duration": median(timings),
            "std_duration": stdev(timings) if len(timings) > 1 else 0,
            "success_rate": success_count / rounds * 100,
            "verifications_per_second": 1 / mean(timings) if mean(timings) > 0 else 0
        }
    
    def benchmark_end_to_end(self, model: str, prompt: str, max_tokens: int, k: int, rounds: int = 3) -> Dict:
        """Benchmark complete end-to-end workflow"""
        print(f"    End-to-end: {model}, {max_tokens} tokens, k={k}")
        
        timings = {
            "total": [],
            "commit": [],
            "open": [],
            "verify": []
        }
        success_count = 0
        
        for round_num in range(rounds):
            total_start = time.time()
            
            prover = Prover(model)
            verifier = Verifier(model)
            
            # Commit phase
            commit_start = time.time()
            commit = prover.commit(prompt, max_new_tokens=max_tokens)
            commit_duration = time.time() - commit_start
            timings["commit"].append(commit_duration)
            
            # Open phase
            open_start = time.time()
            proof_pkg = prover.open(k=k)
            open_duration = time.time() - open_start
            timings["open"].append(open_duration)
            
            # Verify phase
            verify_start = time.time()
            result = verifier.verify(commit, proof_pkg, prover.secret_key)
            verify_duration = time.time() - verify_start
            timings["verify"].append(verify_duration)
            
            total_duration = time.time() - total_start
            timings["total"].append(total_duration)
            
            if result:
                success_count += 1
        
        return {
            "benchmark": "end_to_end",
            "model": model,
            "max_tokens": max_tokens,
            "challenge_size": k,
            "rounds": rounds,
            "phase_timings": {
                phase: {
                    "avg": mean(times),
                    "median": median(times),
                    "std": stdev(times) if len(times) > 1 else 0,
                    "min": min(times),
                    "max": max(times)
                }
                for phase, times in timings.items()
            },
            "success_rate": success_count / rounds * 100,
            "throughput_per_second": 1 / mean(timings["total"]) if mean(timings["total"]) > 0 else 0
        }
    
    def run_scalability_test(self) -> Dict:
        """Test performance scaling with different parameters"""
        print("📈 Running scalability tests...")
        
        scalability_results = []
        
        model = self.models[0]  # Use fastest model for scalability
        prompt = self.prompts["medium"]
        
        # Test scaling with token count
        for tokens in self.token_counts:
            result = self.benchmark_end_to_end(model, prompt, tokens, k=8, rounds=2)
            result["scaling_factor"] = "token_count"
            result["scaling_value"] = tokens
            scalability_results.append(result)
        
        # Test scaling with challenge size
        for k in self.challenge_sizes:
            result = self.benchmark_end_to_end(model, prompt, max_tokens=16, k=k, rounds=2)
            result["scaling_factor"] = "challenge_size"
            result["scaling_value"] = k
            scalability_results.append(result)
        
        return {
            "test": "scalability",
            "results": scalability_results
        }
    
    def run_resource_usage_test(self) -> Dict:
        """Monitor resource usage during GRAIL operations"""
        print("📊 Monitoring resource usage...")
        
        model = self.models[0]
        prompt = self.prompts["medium"]
        
        # Baseline measurements
        baseline_metrics = self.get_system_metrics()
        
        # Run intensive test while monitoring
        prover = Prover(model)
        verifier = Verifier(model)
        
        start_metrics = self.get_system_metrics()
        
        # Perform multiple operations
        commits = []
        proofs = []
        for i in range(5):
            commit = prover.commit(prompt, max_new_tokens=32)
            proof = prover.open(k=16)
            verifier.verify(commit, proof, prover.secret_key)
            
            commits.append(len(commit["tokens"]))
            proofs.append(len(proof["indices"]))
        
        end_metrics = self.get_system_metrics()
        
        return {
            "test": "resource_usage",
            "baseline_metrics": baseline_metrics,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "operations_performed": {
                "commits": len(commits),
                "avg_tokens": mean(commits),
                "avg_challenges": mean(proofs)
            },
            "resource_delta": {
                "cpu_change": end_metrics["cpu_percent"] - start_metrics["cpu_percent"],
                "memory_change_mb": end_metrics["process_memory_mb"] - start_metrics["process_memory_mb"]
            }
        }
    
    def run_experiment(self) -> Dict:
        """Run all performance benchmark tests"""
        print("⚡ Performance Benchmark Experiment")
        print("=" * 50)
        print("Measuring GRAIL performance across different configurations")
        print()
        
        experiment_start = time.time()
        initial_metrics = self.get_system_metrics()
        
        benchmark_results = []
        
        # Core phase benchmarks
        print("🔧 Benchmarking individual phases...")
        for model in self.models[:1]:  # Use first model for detailed benchmarks
            prompt = self.prompts["medium"]
            
            # Commit benchmarks
            for tokens in [16, 32]:
                result = self.benchmark_commit_phase(model, prompt, tokens)
                benchmark_results.append(result)
            
            # Open benchmarks
            for k in [8, 16]:
                result = self.benchmark_open_phase(model, prompt, 16, k)
                benchmark_results.append(result)
            
            # Verify benchmarks
            for k in [8, 16]:
                result = self.benchmark_verify_phase(model, prompt, 16, k)
                benchmark_results.append(result)
        
        # End-to-end benchmarks
        print("\n🔄 Running end-to-end benchmarks...")
        e2e_results = []
        for model in self.models:
            for prompt_name, prompt_text in list(self.prompts.items())[:2]:
                result = self.benchmark_end_to_end(model, prompt_text, 16, 8)
                result["prompt_type"] = prompt_name
                e2e_results.append(result)
        
        # Scalability tests
        scalability_test = self.run_scalability_test()
        
        # Resource usage
        resource_test = self.run_resource_usage_test()
        
        total_duration = time.time() - experiment_start
        final_metrics = self.get_system_metrics()
        
        # Summary statistics
        commit_times = [r["avg_duration"] for r in benchmark_results if r["phase"] == "commit"]
        verify_times = [r["avg_duration"] for r in benchmark_results if r["phase"] == "verify"]
        e2e_times = [r["phase_timings"]["total"]["avg"] for r in e2e_results]
        
        summary = {
            "experiment": "performance_benchmark",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "system_info": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3)
            },
            "performance_summary": {
                "avg_commit_time": mean(commit_times) if commit_times else 0,
                "avg_verify_time": mean(verify_times) if verify_times else 0,
                "avg_e2e_time": mean(e2e_times) if e2e_times else 0,
                "fastest_e2e": min(e2e_times) if e2e_times else 0,
                "slowest_e2e": max(e2e_times) if e2e_times else 0
            },
            "detailed_results": {
                "phase_benchmarks": benchmark_results,
                "end_to_end_benchmarks": e2e_results,
                "scalability_test": scalability_test,
                "resource_usage": resource_test
            },
            "system_metrics": {
                "initial": initial_metrics,
                "final": final_metrics
            }
        }
        
        print()
        print("📊 Performance Summary:")
        print(f"  Average commit time: {summary['performance_summary']['avg_commit_time']:.3f}s")
        print(f"  Average verify time: {summary['performance_summary']['avg_verify_time']:.3f}s")
        print(f"  Average end-to-end: {summary['performance_summary']['avg_e2e_time']:.3f}s")
        print(f"  Fastest end-to-end: {summary['performance_summary']['fastest_e2e']:.3f}s")
        print(f"  Total benchmarks: {len(benchmark_results)} phases, {len(e2e_results)} end-to-end")
        
        return summary

def main():
    """Main entry point"""
    experiment = PerformanceBenchmarkExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_performance_benchmark.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if benchmarks completed
    return 0

if __name__ == "__main__":
    sys.exit(main()) 