#!/usr/bin/env python3
"""
Experiment: Top-k Index Mismatch Analysis
Computes mismatch ratio of top-k indices between generation and validation.
Tests across at least 3 models and plots top-k vs mismatch error (median and max).
Generates Figure 3 (Mismatch ratio vs top-k size).
"""

import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from statistics import mean, median

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class TopKIndexMismatchExperiment:
    """Analyze top-k index mismatches between generation and validation phases"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations - using at least 3 models as specified
        self.models = ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2", "microsoft/DialoGPT-small"]
        self.top_k_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.num_test_queries = 500  # Multiple queries for statistical significance
        
        # Diverse test prompts
        self.prompts = [
            "The advancement of artificial intelligence in recent years",
            "Climate change and its effects on global ecosystems",
            "The role of technology in modern education systems",
            "Economic factors that influence international trade",
            "Scientific discoveries that have shaped our understanding",
            "The importance of renewable energy sources for",
            "Cultural diversity and its impact on society",
            "Medical breakthroughs in the treatment of diseases",
            "The evolution of communication technologies throughout history",
            "Environmental conservation efforts and their effectiveness",
            "Space exploration missions and their scientific contributions",
            "The development of sustainable agricultural practices"
        ]
    
    def get_top_k_indices_from_logits(self, logits: torch.Tensor, k: int) -> List[int]:
        """Extract top-k token indices from logits"""
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        return top_k_indices.cpu().tolist()
    
    def generate_with_logit_capture(self, model, tokenizer, prompt: str, max_tokens: int, k: int) -> Dict:
        """Generate text while capturing top-k indices at each step"""
        device = model.device
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generated_tokens = []
        step_data = []
        
        with torch.no_grad():
            current_ids = input_ids
            
            for step in range(max_tokens):
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get top-k indices
                top_k_indices = self.get_top_k_indices_from_logits(logits, k)
                
                # Sample next token (not necessarily from top-k for realistic generation)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                step_data.append({
                    "step": step,
                    "top_k_indices": top_k_indices,
                    "selected_token": next_token,
                    "token_in_top_k": next_token in top_k_indices
                })
                
                # Add token for next iteration
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
        
        return {
            "prompt": prompt,
            "generated_tokens": generated_tokens,
            "step_data": step_data,
            "total_steps": len(step_data)
        }
    
    def validate_generation(self, model, tokenizer, prompt: str, generated_tokens: List[int], k: int) -> Dict:
        """Validate by re-running the same generation path and comparing top-k indices"""
        device = model.device
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        validation_data = []
        
        with torch.no_grad():
            current_ids = input_ids
            
            for step, expected_token in enumerate(generated_tokens):
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get top-k indices for validation
                top_k_indices = self.get_top_k_indices_from_logits(logits, k)
                
                validation_data.append({
                    "step": step,
                    "top_k_indices": top_k_indices,
                    "expected_token": expected_token
                })
                
                # Force the same token path as original generation
                current_ids = torch.cat([current_ids, torch.tensor([[expected_token]], device=device)], dim=1)
        
        return {
            "validation_data": validation_data,
            "total_steps": len(validation_data)
        }
    
    def compute_index_mismatch(self, generation: Dict, validation: Dict, k: int) -> Dict:
        """Compare top-k indices between generation and validation"""
        mismatches = []
        match_counts = []
        mismatch_ratios = []
        
        min_steps = min(len(generation["step_data"]), len(validation["validation_data"]))
        
        for step in range(min_steps):
            gen_indices = set(generation["step_data"][step]["top_k_indices"])
            val_indices = set(validation["validation_data"][step]["top_k_indices"])
            
            # Calculate intersection and differences
            intersection = gen_indices.intersection(val_indices)
            union = gen_indices.union(val_indices)
            
            match_count = len(intersection)
            total_unique = len(union)
            mismatch_count = total_unique - match_count
            
            # Mismatch ratio: how many indices are different relative to k
            mismatch_ratio = mismatch_count / (2 * k) if k > 0 else 0
            
            match_counts.append(match_count)
            mismatch_ratios.append(mismatch_ratio)
            
            mismatches.append({
                "step": step,
                "generation_indices": list(gen_indices),
                "validation_indices": list(val_indices),
                "matches": match_count,
                "total_unique": total_unique,
                "mismatch_count": mismatch_count,
                "mismatch_ratio": mismatch_ratio,
                "jaccard_similarity": len(intersection) / len(union) if union else 1.0
            })
        
        return {
            "step_mismatches": mismatches,
            "total_steps": min_steps,
            "avg_match_count": mean(match_counts) if match_counts else 0,
            "median_match_count": median(match_counts) if match_counts else 0,
            "avg_mismatch_ratio": mean(mismatch_ratios) if mismatch_ratios else 0,
            "median_mismatch_ratio": median(mismatch_ratios) if mismatch_ratios else 0,
            "max_mismatch_ratio": max(mismatch_ratios) if mismatch_ratios else 0,
            "min_mismatch_ratio": min(mismatch_ratios) if mismatch_ratios else 0
        }
    
    def run_single_model_test(self, model_name: str) -> Dict:
        """Run index mismatch analysis for a single model across all top-k values"""
        print(f"  📱 Testing model: {model_name}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        model_results = []
        
        for k_idx, k in enumerate(self.top_k_values):
            print(f"    Testing k={k} ({k_idx + 1}/{len(self.top_k_values)})")
            
            k_results = []
            
            for query_idx in range(self.num_test_queries):
                if query_idx % 100 == 0 and query_idx > 0:
                    print(f"      Query {query_idx}/{self.num_test_queries}")
                
                # Use different prompts cyclically
                prompt = self.prompts[query_idx % len(self.prompts)]
                
                # Generate with logit capture
                generation = self.generate_with_logit_capture(model, tokenizer, prompt, 16, k)
                
                # Validate the same generation path
                validation = self.validate_generation(model, tokenizer, prompt, generation["generated_tokens"], k)
                
                # Compute mismatch statistics
                mismatch_analysis = self.compute_index_mismatch(generation, validation, k)
                
                k_results.append({
                    "query_id": query_idx,
                    "prompt": prompt,
                    "generation_steps": generation["total_steps"],
                    "mismatch_analysis": mismatch_analysis
                })
            
            # Aggregate results for this k value
            avg_mismatch_ratios = [r["mismatch_analysis"]["avg_mismatch_ratio"] for r in k_results]
            median_mismatch_ratios = [r["mismatch_analysis"]["median_mismatch_ratio"] for r in k_results]
            max_mismatch_ratios = [r["mismatch_analysis"]["max_mismatch_ratio"] for r in k_results]
            
            model_results.append({
                "k": k,
                "total_queries": len(k_results),
                "aggregate_stats": {
                    "mean_avg_mismatch_ratio": mean(avg_mismatch_ratios) if avg_mismatch_ratios else 0,
                    "median_avg_mismatch_ratio": median(avg_mismatch_ratios) if avg_mismatch_ratios else 0,
                    "mean_median_mismatch_ratio": mean(median_mismatch_ratios) if median_mismatch_ratios else 0,
                    "median_median_mismatch_ratio": median(median_mismatch_ratios) if median_mismatch_ratios else 0,
                    "mean_max_mismatch_ratio": mean(max_mismatch_ratios) if max_mismatch_ratios else 0,
                    "median_max_mismatch_ratio": median(max_mismatch_ratios) if max_mismatch_ratios else 0,
                    "overall_max_mismatch_ratio": max(max_mismatch_ratios) if max_mismatch_ratios else 0
                },
                "query_results": k_results
            })
            
            print(f"      ✅ k={k}: median mismatch ratio = {model_results[-1]['aggregate_stats']['median_median_mismatch_ratio']:.4f}")
        
        # Clean up model to free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "model": model_name,
            "k_values": self.top_k_values,
            "queries_per_k": self.num_test_queries,
            "k_results": model_results
        }
    
    def run_experiment(self) -> Dict:
        """Run complete top-k index mismatch analysis"""
        print("🧪 Top-k Index Mismatch Analysis Experiment")
        print("=" * 60)
        print(f"Testing {len(self.models)} models with {len(self.top_k_values)} top-k values")
        print(f"Queries per (model, k) combination: {self.num_test_queries}")
        print()
        
        experiment_start = time.time()
        
        for model_name in self.models:
            result = self.run_single_model_test(model_name)
            self.results.append(result)
            print(f"  ✅ Completed {model_name}")
            print()
        
        total_duration = time.time() - experiment_start
        
        # Generate comprehensive analysis
        summary_analysis = self.generate_summary_analysis()
        
        summary = {
            "experiment": "topk_index_mismatch",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "configuration": {
                "models": self.models,
                "top_k_values": self.top_k_values,
                "queries_per_k": self.num_test_queries,
                "total_queries": len(self.models) * len(self.top_k_values) * self.num_test_queries
            },
            "results": self.results,
            "summary_analysis": summary_analysis,
            "figure_data": self.prepare_figure_data()
        }
        
        self.print_summary_analysis(summary_analysis)
        
        return summary
    
    def generate_summary_analysis(self) -> Dict:
        """Generate summary analysis across all models and k values"""
        analysis = {}
        
        for result in self.results:
            model_name = result["model"]
            
            k_values = []
            median_mismatch_ratios = []
            max_mismatch_ratios = []
            mean_mismatch_ratios = []
            
            for k_result in result["k_results"]:
                k = k_result["k"]
                stats = k_result["aggregate_stats"]
                
                k_values.append(k)
                median_mismatch_ratios.append(stats["median_median_mismatch_ratio"])
                max_mismatch_ratios.append(stats["median_max_mismatch_ratio"])
                mean_mismatch_ratios.append(stats["mean_avg_mismatch_ratio"])
            
            # Calculate correlation between k and mismatch ratio
            if len(k_values) > 1:
                median_correlation = np.corrcoef(k_values, median_mismatch_ratios)[0, 1]
                max_correlation = np.corrcoef(k_values, max_mismatch_ratios)[0, 1]
            else:
                median_correlation = max_correlation = 0
            
            analysis[model_name] = {
                "k_values": k_values,
                "median_mismatch_ratios": median_mismatch_ratios,
                "max_mismatch_ratios": max_mismatch_ratios,
                "mean_mismatch_ratios": mean_mismatch_ratios,
                "correlations": {
                    "k_vs_median_mismatch": median_correlation,
                    "k_vs_max_mismatch": max_correlation
                },
                "best_k": k_values[np.argmin(median_mismatch_ratios)] if median_mismatch_ratios else None,
                "worst_k": k_values[np.argmax(median_mismatch_ratios)] if median_mismatch_ratios else None,
                "mismatch_range": max(median_mismatch_ratios) - min(median_mismatch_ratios) if median_mismatch_ratios else 0
            }
        
        return analysis
    
    def prepare_figure_data(self) -> Dict:
        """Prepare data for Figure 3: Mismatch ratio vs top-k size"""
        figure_data = {
            "title": "Figure 3: Mismatch Ratio vs Top-k Size",
            "x_axis": "Top-k Size",
            "y_axis": "Mismatch Ratio",
            "models": {}
        }
        
        for result in self.results:
            model_name = result["model"]
            
            k_values = []
            median_ratios = []
            max_ratios = []
            
            for k_result in result["k_results"]:
                k_values.append(k_result["k"])
                median_ratios.append(k_result["aggregate_stats"]["median_median_mismatch_ratio"])
                max_ratios.append(k_result["aggregate_stats"]["median_max_mismatch_ratio"])
            
            figure_data["models"][model_name] = {
                "k_values": k_values,
                "median_mismatch_ratios": median_ratios,
                "max_mismatch_ratios": max_ratios
            }
        
        return figure_data
    
    def print_summary_analysis(self, analysis: Dict):
        """Print formatted summary analysis"""
        print("\n📊 TOP-K INDEX MISMATCH ANALYSIS")
        print("=" * 60)
        
        # Table format for mismatch ratios
        print(f"\n📈 Mismatch Ratios by Top-k Size:")
        print(f"{'Model':<20} ", end="")
        for k in self.top_k_values:
            print(f"{'k=' + str(k):<10}", end="")
        print()
        print("-" * (20 + 10 * len(self.top_k_values)))
        
        for model_name, data in analysis.items():
            print(f"{model_name:<20} ", end="")
            for i, k in enumerate(self.top_k_values):
                if i < len(data["median_mismatch_ratios"]):
                    ratio = data["median_mismatch_ratios"][i]
                    print(f"{ratio:>8.4f}  ", end="")
                else:
                    print(f"{'N/A':<10}", end="")
            print()
        
        print(f"\n🔍 Correlation Analysis:")
        for model_name, data in analysis.items():
            corr_median = data["correlations"]["k_vs_median_mismatch"]
            corr_max = data["correlations"]["k_vs_max_mismatch"]
            direction_median = "increases" if corr_median > 0.1 else "decreases" if corr_median < -0.1 else "stable"
            direction_max = "increases" if corr_max > 0.1 else "decreases" if corr_max < -0.1 else "stable"
            
            print(f"  {model_name}:")
            print(f"    Median mismatch vs k: {direction_median} (r={corr_median:.3f})")
            print(f"    Max mismatch vs k: {direction_max} (r={corr_max:.3f})")
        
        print(f"\n🎯 Key Findings:")
        
        # Find model with best/worst mismatch characteristics
        avg_median_mismatches = {name: mean(data["median_mismatch_ratios"]) for name, data in analysis.items() if data["median_mismatch_ratios"]}
        
        if avg_median_mismatches:
            best_model = min(avg_median_mismatches.items(), key=lambda x: x[1])
            worst_model = max(avg_median_mismatches.items(), key=lambda x: x[1])
            
            print(f"  • Best Index Consistency: {best_model[0]} (avg: {best_model[1]:.4f})")
            print(f"  • Worst Index Consistency: {worst_model[0]} (avg: {worst_model[1]:.4f})")
        
        # Analyze k-value trends
        stable_models = [name for name, data in analysis.items() if abs(data["correlations"]["k_vs_median_mismatch"]) < 0.1]
        increasing_models = [name for name, data in analysis.items() if data["correlations"]["k_vs_median_mismatch"] > 0.1]
        decreasing_models = [name for name, data in analysis.items() if data["correlations"]["k_vs_median_mismatch"] < -0.1]
        
        if stable_models:
            print(f"  • Stable across k-values: {', '.join(stable_models)}")
        if increasing_models:
            print(f"  • Mismatch increases with k: {', '.join(increasing_models)}")
        if decreasing_models:
            print(f"  • Mismatch decreases with k: {', '.join(decreasing_models)}")

def main():
    """Main entry point"""
    experiment = TopKIndexMismatchExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_topk_index_mismatch.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if experiment completed
    return 0

if __name__ == "__main__":
    sys.exit(main()) 