#!/usr/bin/env python3
"""
Experiment: Top-k Exponent Error Analysis
Varies top-k from 64 to 4096 and counts exact matches vs deviations for exponents.
Tests across 2000 queries to generate Table 1 (Exponent mismatch across top-k values).
"""

import sys
import time
import json
import struct
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from statistics import mean, median

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class TopKExponentErrorExperiment:
    """Analyze exponent errors across different top-k values"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations
        self.models = ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"]
        self.top_k_values = [64, 128, 256, 512, 1024, 2048, 4096]
        self.num_queries = 2000  # As specified
        
        # Diverse prompt set for comprehensive testing
        self.prompts = [
            "The quick brown fox jumps over",
            "In a world where technology advances",
            "The benefits of renewable energy include",
            "Machine learning algorithms can be used to",
            "Climate change is affecting global weather patterns",
            "The history of artificial intelligence begins with",
            "Economic factors influencing market behavior",
            "Scientific research has shown that",
            "Cultural differences between countries can be",
            "The future of space exploration involves"
        ]
    
    def extract_float_components(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Extract exponent and mantissa components from float32 tensor"""
        # Convert to numpy for bit manipulation
        float_array = tensor.detach().cpu().numpy().astype(np.float32)
        
        # Reinterpret as uint32 to access bits
        uint_view = float_array.view(np.uint32)
        
        # Extract exponent (bits 23-30, mask 0x7F800000)
        exponents = (uint_view & 0x7F800000) >> 23
        
        # Extract mantissa (bits 0-22, mask 0x007FFFFF) 
        mantissas = uint_view & 0x007FFFFF
        
        return exponents, mantissas
    
    def get_top_k_logits_with_components(self, model, tokenizer, prompt: str, max_tokens: int, top_k: int) -> Dict:
        """Generate text and extract top-k logits with float components"""
        device = model.device
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        generated_tokens = []
        logit_data = []
        
        with torch.no_grad():
            # Generate tokens one by one to capture logits at each step
            current_ids = input_ids
            
            for step in range(max_tokens):
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                
                # Extract float components
                exponents, mantissas = self.extract_float_components(top_k_values)
                
                # Sample next token from top-k
                probs = torch.softmax(top_k_values, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[sampled_idx].item()
                
                generated_tokens.append(next_token)
                logit_data.append({
                    "step": step,
                    "top_k_indices": top_k_indices.cpu().tolist(),
                    "top_k_values": top_k_values.cpu().tolist(),
                    "exponents": exponents.tolist(),
                    "mantissas": mantissas.tolist(),
                    "selected_token": next_token
                })
                
                # Append token for next iteration
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
        
        return {
            "prompt": prompt,
            "generated_tokens": generated_tokens,
            "logit_data": logit_data,
            "total_steps": len(logit_data)
        }
    
    def analyze_exponent_matches(self, generation1: Dict, generation2: Dict, top_k: int) -> Dict:
        """Compare exponent components between two generations"""
        matches = 0
        total_comparisons = 0
        mismatches = []
        
        min_steps = min(len(generation1["logit_data"]), len(generation2["logit_data"]))
        
        for step in range(min_steps):
            exp1 = generation1["logit_data"][step]["exponents"][:top_k]
            exp2 = generation2["logit_data"][step]["exponents"][:top_k]
            
            for i in range(min(len(exp1), len(exp2))):
                total_comparisons += 1
                if exp1[i] == exp2[i]:
                    matches += 1
                else:
                    mismatches.append({
                        "step": step,
                        "position": i,
                        "exp1": exp1[i],
                        "exp2": exp2[i],
                        "difference": abs(exp1[i] - exp2[i])
                    })
        
        match_rate = (matches / total_comparisons * 100) if total_comparisons > 0 else 0
        
        return {
            "total_comparisons": total_comparisons,
            "matches": matches,
            "mismatches_count": len(mismatches),
            "match_rate": match_rate,
            "mismatch_rate": 100 - match_rate,
            "sample_mismatches": mismatches[:10]  # Store sample for analysis
        }
    
    def run_single_top_k_test(self, model_name: str, top_k: int, query_batch_size: int = 100) -> Dict:
        """Run exponent error analysis for a single top-k value"""
        print(f"    Testing top-k={top_k} with {model_name}")
        
        # Initialize model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        all_results = []
        processed_queries = 0
        
        # Process queries in batches to manage memory
        while processed_queries < self.num_queries:
            batch_results = []
            batch_size = min(query_batch_size, self.num_queries - processed_queries)
            
            for i in range(batch_size):
                # Use different prompts cyclically
                prompt = self.prompts[processed_queries % len(self.prompts)]
                
                # Generate twice with same prompt to compare consistency
                gen1 = self.get_top_k_logits_with_components(model, tokenizer, prompt, 16, top_k)
                gen2 = self.get_top_k_logits_with_components(model, tokenizer, prompt, 16, top_k)
                
                # Analyze exponent matches
                match_analysis = self.analyze_exponent_matches(gen1, gen2, top_k)
                
                batch_results.append({
                    "query_id": processed_queries,
                    "prompt": prompt,
                    "match_analysis": match_analysis
                })
                
                processed_queries += 1
            
            all_results.extend(batch_results)
            
            if processed_queries % 200 == 0:
                print(f"      Processed {processed_queries}/{self.num_queries} queries")
        
        # Aggregate statistics
        total_comparisons = sum(r["match_analysis"]["total_comparisons"] for r in all_results)
        total_matches = sum(r["match_analysis"]["matches"] for r in all_results)
        match_rates = [r["match_analysis"]["match_rate"] for r in all_results]
        
        # Clean up model to free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "model": model_name,
            "top_k": top_k,
            "total_queries": len(all_results),
            "total_comparisons": total_comparisons,
            "total_matches": total_matches,
            "overall_match_rate": (total_matches / total_comparisons * 100) if total_comparisons > 0 else 0,
            "avg_match_rate": mean(match_rates) if match_rates else 0,
            "median_match_rate": median(match_rates) if match_rates else 0,
            "query_results": all_results
        }
    
    def run_experiment(self) -> Dict:
        """Run complete top-k exponent error analysis"""
        print("🧪 Top-k Exponent Error Analysis Experiment")
        print("=" * 60)
        print(f"Testing {len(self.top_k_values)} top-k values across {len(self.models)} models")
        print(f"Total queries: {self.num_queries}")
        print()
        
        experiment_start = time.time()
        
        for model_name in self.models:
            print(f"📱 Testing model: {model_name}")
            
            model_results = []
            
            for top_k in self.top_k_values:
                result = self.run_single_top_k_test(model_name, top_k)
                model_results.append(result)
                
                print(f"    ✅ top-k={top_k}: {result['overall_match_rate']:.2f}% match rate")
            
            self.results.append({
                "model": model_name,
                "top_k_results": model_results
            })
            print()
        
        total_duration = time.time() - experiment_start
        
        # Generate summary analysis
        summary_table = self.generate_summary_table()
        
        summary = {
            "experiment": "topk_exponent_error",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "configuration": {
                "models": self.models,
                "top_k_values": self.top_k_values,
                "total_queries": self.num_queries,
                "prompts_used": len(self.prompts)
            },
            "results": self.results,
            "summary_table": summary_table,
            "analysis": self.analyze_trends()
        }
        
        self.print_summary_table(summary_table)
        
        return summary
    
    def generate_summary_table(self) -> Dict:
        """Generate Table 1: Exponent mismatch across top-k values"""
        table = {}
        
        for model_result in self.results:
            model_name = model_result["model"]
            table[model_name] = {}
            
            for topk_result in model_result["top_k_results"]:
                top_k = topk_result["top_k"]
                match_rate = topk_result["overall_match_rate"]
                mismatch_rate = 100 - match_rate
                
                table[model_name][f"top_{top_k}"] = {
                    "match_rate": match_rate,
                    "mismatch_rate": mismatch_rate,
                    "total_comparisons": topk_result["total_comparisons"]
                }
        
        return table
    
    def analyze_trends(self) -> Dict:
        """Analyze trends in exponent error rates"""
        trends = {}
        
        for model_result in self.results:
            model_name = model_result["model"]
            
            match_rates = []
            top_k_vals = []
            
            for topk_result in model_result["top_k_results"]:
                top_k_vals.append(topk_result["top_k"])
                match_rates.append(topk_result["overall_match_rate"])
            
            # Calculate correlation between top-k and match rate
            if len(match_rates) > 1:
                correlation = np.corrcoef(top_k_vals, match_rates)[0, 1]
            else:
                correlation = 0
            
            trends[model_name] = {
                "top_k_values": top_k_vals,
                "match_rates": match_rates,
                "correlation_topk_vs_match": correlation,
                "best_top_k": top_k_vals[np.argmax(match_rates)],
                "worst_top_k": top_k_vals[np.argmin(match_rates)],
                "range_match_rate": max(match_rates) - min(match_rates)
            }
        
        return trends
    
    def print_summary_table(self, table: Dict):
        """Print formatted summary table"""
        print("\n📊 TABLE 1: Exponent Mismatch Across Top-k Values")
        print("=" * 80)
        
        # Header
        print(f"{'Model':<20} ", end="")
        for top_k in self.top_k_values:
            print(f"{'k=' + str(top_k):<12}", end="")
        print()
        print("-" * 80)
        
        # Data rows
        for model_name in self.models:
            print(f"{model_name:<20} ", end="")
            
            if model_name in table:
                for top_k in self.top_k_values:
                    key = f"top_{top_k}"
                    if key in table[model_name]:
                        mismatch_rate = table[model_name][key]["mismatch_rate"]
                        print(f"{mismatch_rate:>8.2f}%   ", end="")
                    else:
                        print(f"{'N/A':<12}", end="")
            print()
        
        print("\n📈 Key Findings:")
        analysis = self.analyze_trends()
        for model_name, trends in analysis.items():
            corr = trends["correlation_topk_vs_match"]
            direction = "increases" if corr > 0 else "decreases" if corr < 0 else "stable"
            print(f"  {model_name}: Match rate {direction} with top-k (correlation: {corr:.3f})")

def main():
    """Main entry point"""
    experiment = TopKExponentErrorExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_topk_exponent_error.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if experiment completed
    return 0

if __name__ == "__main__":
    sys.exit(main()) 