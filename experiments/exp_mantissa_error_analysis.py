#!/usr/bin/env python3
"""
Experiment: Mantissa Error vs Token Index Analysis
Tracks mantissa difference statistics (mean, median, max) over 0-2048 tokens.
Shows how errors degrade across generation based on KV cache effects.
Generates Figure 2 (Mantissa error growth across generation).
"""

import sys
import time
import json
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from statistics import mean, median

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class MantissaErrorAnalysisExperiment:
    """Analyze mantissa error degradation across token generation"""
    
    def __init__(self):
        self.results = []
        
        # Test configurations
        self.models = ["sshleifer/tiny-gpt2", "distilgpt2", "gpt2"]
        self.max_tokens = 2048  # As specified in user requirements
        self.token_checkpoints = [0, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
        self.num_test_runs = 50  # Multiple runs for statistical significance
        
        # Diverse prompts for comprehensive testing
        self.prompts = [
            "In the distant future, humanity has expanded",
            "The mathematical relationship between quantum mechanics and",
            "Artificial intelligence systems are increasingly being used to",
            "Economic policies implemented by governments often reflect",
            "Scientific research in the field of neuroscience has revealed",
            "Climate change mitigation strategies require careful consideration of",
            "The philosophical implications of consciousness and free will",
            "Technology companies are investing heavily in research and development",
            "Cultural exchange between different civilizations has historically led to",
            "The fundamental principles of physics govern the behavior of"
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
    
    def generate_long_sequence(self, model, tokenizer, prompt: str, max_tokens: int) -> Dict:
        """Generate a long sequence while tracking hidden states and mantissa values"""
        device = model.device
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        initial_length = input_ids.shape[1]
        
        generated_tokens = []
        token_data = []
        
        with torch.no_grad():
            current_ids = input_ids
            
            for step in range(max_tokens):
                outputs = model(current_ids, output_hidden_states=True)
                
                # Get hidden states from the target layer
                h_layer = outputs.hidden_states[-1][0]  # Last layer, first batch
                
                # Get the hidden state for the last token
                last_token_hidden = h_layer[-1]  # Shape: [hidden_dim]
                
                # Extract mantissa components
                exponents, mantissas = self.extract_float_components(last_token_hidden)
                
                # Generate next token
                logits = outputs.logits[0, -1, :]
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                
                generated_tokens.append(next_token)
                token_data.append({
                    "token_index": initial_length + step,
                    "generation_step": step,
                    "token_id": next_token,
                    "hidden_state_stats": {
                        "mean": float(last_token_hidden.mean()),
                        "std": float(last_token_hidden.std()),
                        "min": float(last_token_hidden.min()),
                        "max": float(last_token_hidden.max())
                    },
                    "mantissa_stats": {
                        "mean": float(np.mean(mantissas)),
                        "median": float(np.median(mantissas)),
                        "std": float(np.std(mantissas)),
                        "min": int(np.min(mantissas)),
                        "max": int(np.max(mantissas))
                    },
                    "exponent_stats": {
                        "mean": float(np.mean(exponents)),
                        "median": float(np.median(exponents)),
                        "std": float(np.std(exponents)),
                        "min": int(np.min(exponents)),
                        "max": int(np.max(exponents))
                    }
                })
                
                # Add token for next iteration
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
                # Progress tracking
                if (step + 1) % 256 == 0:
                    print(f"      Generated {step + 1}/{max_tokens} tokens")
        
        return {
            "prompt": prompt,
            "initial_length": initial_length,
            "generated_tokens": generated_tokens,
            "token_data": token_data,
            "final_length": initial_length + len(generated_tokens)
        }
    
    def analyze_mantissa_degradation(self, generation1: Dict, generation2: Dict) -> Dict:
        """Compare mantissa statistics between two generations to measure degradation"""
        degradation_data = []
        
        min_tokens = min(len(generation1["token_data"]), len(generation2["token_data"]))
        
        for i in range(min_tokens):
            data1 = generation1["token_data"][i]
            data2 = generation2["token_data"][i]
            
            # Calculate mantissa differences
            mantissa1 = data1["mantissa_stats"]
            mantissa2 = data2["mantissa_stats"]
            
            diff_stats = {
                "token_index": data1["token_index"],
                "generation_step": data1["generation_step"],
                "mean_diff": abs(mantissa1["mean"] - mantissa2["mean"]),
                "median_diff": abs(mantissa1["median"] - mantissa2["median"]),
                "std_diff": abs(mantissa1["std"] - mantissa2["std"]),
                "max_diff": abs(mantissa1["max"] - mantissa2["max"]),
                "min_diff": abs(mantissa1["min"] - mantissa2["min"])
            }
            
            degradation_data.append(diff_stats)
        
        return {
            "degradation_by_token": degradation_data,
            "total_tokens_compared": min_tokens
        }
    
    def run_single_model_test(self, model_name: str) -> Dict:
        """Run mantissa error analysis for a single model"""
        print(f"  📱 Testing model: {model_name}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        
        all_degradation_data = []
        run_summaries = []
        
        for run_idx in range(self.num_test_runs):
            if run_idx % 10 == 0:
                print(f"    Run {run_idx + 1}/{self.num_test_runs}")
            
            # Use different prompts cyclically
            prompt = self.prompts[run_idx % len(self.prompts)]
            
            # Generate two sequences for comparison
            gen1 = self.generate_long_sequence(model, tokenizer, prompt, self.max_tokens)
            gen2 = self.generate_long_sequence(model, tokenizer, prompt, self.max_tokens)
            
            # Analyze degradation
            degradation = self.analyze_mantissa_degradation(gen1, gen2)
            all_degradation_data.append(degradation)
            
            # Compute summary statistics for this run
            degradation_by_token = degradation["degradation_by_token"]
            if degradation_by_token:
                mean_diffs = [d["mean_diff"] for d in degradation_by_token]
                median_diffs = [d["median_diff"] for d in degradation_by_token]
                max_diffs = [d["max_diff"] for d in degradation_by_token]
                
                run_summaries.append({
                    "run_id": run_idx,
                    "prompt": prompt,
                    "tokens_generated": len(degradation_by_token),
                    "avg_mean_diff": mean(mean_diffs) if mean_diffs else 0,
                    "avg_median_diff": mean(median_diffs) if median_diffs else 0,
                    "avg_max_diff": mean(max_diffs) if max_diffs else 0,
                    "final_mean_diff": mean_diffs[-1] if mean_diffs else 0,
                    "final_median_diff": median_diffs[-1] if median_diffs else 0,
                    "final_max_diff": max_diffs[-1] if max_diffs else 0
                })
        
        # Aggregate statistics across all runs
        aggregated_stats = self.aggregate_degradation_stats(all_degradation_data)
        
        # Clean up model to free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "model": model_name,
            "total_runs": len(run_summaries),
            "tokens_per_run": self.max_tokens,
            "run_summaries": run_summaries,
            "aggregated_degradation": aggregated_stats,
            "checkpoint_analysis": self.analyze_checkpoints(aggregated_stats)
        }
    
    def aggregate_degradation_stats(self, all_degradation_data: List[Dict]) -> Dict:
        """Aggregate mantissa degradation statistics across all runs"""
        # Group by token position
        token_position_data = defaultdict(list)
        
        for degradation in all_degradation_data:
            for token_data in degradation["degradation_by_token"]:
                pos = token_data["generation_step"]
                token_position_data[pos].append(token_data)
        
        # Calculate aggregate statistics for each position
        aggregated = {}
        
        for position, data_list in token_position_data.items():
            mean_diffs = [d["mean_diff"] for d in data_list]
            median_diffs = [d["median_diff"] for d in data_list]
            max_diffs = [d["max_diff"] for d in data_list]
            
            aggregated[position] = {
                "generation_step": position,
                "num_samples": len(data_list),
                "mean_diff_stats": {
                    "mean": mean(mean_diffs) if mean_diffs else 0,
                    "median": median(mean_diffs) if mean_diffs else 0,
                    "min": min(mean_diffs) if mean_diffs else 0,
                    "max": max(mean_diffs) if mean_diffs else 0,
                    "std": np.std(mean_diffs) if len(mean_diffs) > 1 else 0
                },
                "median_diff_stats": {
                    "mean": mean(median_diffs) if median_diffs else 0,
                    "median": median(median_diffs) if median_diffs else 0,
                    "min": min(median_diffs) if median_diffs else 0,
                    "max": max(median_diffs) if median_diffs else 0,
                    "std": np.std(median_diffs) if len(median_diffs) > 1 else 0
                },
                "max_diff_stats": {
                    "mean": mean(max_diffs) if max_diffs else 0,
                    "median": median(max_diffs) if max_diffs else 0,
                    "min": min(max_diffs) if max_diffs else 0,
                    "max": max(max_diffs) if max_diffs else 0,
                    "std": np.std(max_diffs) if len(max_diffs) > 1 else 0
                }
            }
        
        return aggregated
    
    def analyze_checkpoints(self, aggregated_stats: Dict) -> Dict:
        """Analyze error growth at specific token checkpoints"""
        checkpoint_analysis = {}
        
        for checkpoint in self.token_checkpoints:
            if checkpoint in aggregated_stats:
                data = aggregated_stats[checkpoint]
                checkpoint_analysis[checkpoint] = {
                    "position": checkpoint,
                    "mean_degradation": data["mean_diff_stats"]["mean"],
                    "median_degradation": data["median_diff_stats"]["mean"],
                    "max_degradation": data["max_diff_stats"]["mean"],
                    "samples": data["num_samples"]
                }
        
        return checkpoint_analysis
    
    def run_experiment(self) -> Dict:
        """Run complete mantissa error analysis"""
        print("🧪 Mantissa Error vs Token Index Analysis")
        print("=" * 60)
        print(f"Testing {len(self.models)} models with {self.max_tokens} tokens each")
        print(f"Runs per model: {self.num_test_runs}")
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
            "experiment": "mantissa_error_analysis",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "configuration": {
                "models": self.models,
                "max_tokens": self.max_tokens,
                "num_runs": self.num_test_runs,
                "checkpoints": self.token_checkpoints
            },
            "results": self.results,
            "summary_analysis": summary_analysis,
            "figure_data": self.prepare_figure_data()
        }
        
        self.print_summary_analysis(summary_analysis)
        
        return summary
    
    def generate_summary_analysis(self) -> Dict:
        """Generate summary analysis across all models"""
        analysis = {}
        
        for result in self.results:
            model_name = result["model"]
            checkpoint_data = result["checkpoint_analysis"]
            
            # Extract degradation trends
            positions = []
            mean_degradations = []
            median_degradations = []
            max_degradations = []
            
            for checkpoint, data in checkpoint_data.items():
                positions.append(checkpoint)
                mean_degradations.append(data["mean_degradation"])
                median_degradations.append(data["median_degradation"])
                max_degradations.append(data["max_degradation"])
            
            # Calculate growth rates
            if len(positions) > 1:
                mean_growth_rate = (mean_degradations[-1] - mean_degradations[0]) / (positions[-1] - positions[0])
                median_growth_rate = (median_degradations[-1] - median_degradations[0]) / (positions[-1] - positions[0])
                max_growth_rate = (max_degradations[-1] - max_degradations[0]) / (positions[-1] - positions[0])
            else:
                mean_growth_rate = median_growth_rate = max_growth_rate = 0
            
            analysis[model_name] = {
                "positions": positions,
                "mean_degradations": mean_degradations,
                "median_degradations": median_degradations,
                "max_degradations": max_degradations,
                "growth_rates": {
                    "mean": mean_growth_rate,
                    "median": median_growth_rate,
                    "max": max_growth_rate
                },
                "initial_error": mean_degradations[0] if mean_degradations else 0,
                "final_error": mean_degradations[-1] if mean_degradations else 0,
                "error_ratio": (mean_degradations[-1] / mean_degradations[0]) if mean_degradations and mean_degradations[0] > 0 else 1
            }
        
        return analysis
    
    def prepare_figure_data(self) -> Dict:
        """Prepare data for Figure 2: Mantissa error growth across generation"""
        figure_data = {
            "title": "Figure 2: Mantissa Error Growth Across Generation",
            "x_axis": "Token Index",
            "y_axis": "Mantissa Difference",
            "models": {}
        }
        
        for result in self.results:
            model_name = result["model"]
            aggregated = result["aggregated_degradation"]
            
            positions = sorted(aggregated.keys())
            mean_errors = [aggregated[pos]["mean_diff_stats"]["mean"] for pos in positions]
            median_errors = [aggregated[pos]["median_diff_stats"]["mean"] for pos in positions]
            max_errors = [aggregated[pos]["max_diff_stats"]["mean"] for pos in positions]
            
            figure_data["models"][model_name] = {
                "positions": positions,
                "mean_errors": mean_errors,
                "median_errors": median_errors,
                "max_errors": max_errors
            }
        
        return figure_data
    
    def print_summary_analysis(self, analysis: Dict):
        """Print formatted summary analysis"""
        print("\n📊 MANTISSA ERROR DEGRADATION ANALYSIS")
        print("=" * 60)
        
        print(f"\n📈 Growth Rates (Error per Token):")
        for model_name, data in analysis.items():
            growth = data["growth_rates"]
            print(f"  {model_name}:")
            print(f"    Mean Error Growth: {growth['mean']:.6f}")
            print(f"    Median Error Growth: {growth['median']:.6f}")
            print(f"    Max Error Growth: {growth['max']:.6f}")
            print(f"    Error Multiplication Factor: {data['error_ratio']:.2f}x")
        
        print(f"\n🎯 Key Findings:")
        
        # Find model with highest degradation
        highest_growth = max(analysis.items(), key=lambda x: x[1]["growth_rates"]["mean"])
        lowest_growth = min(analysis.items(), key=lambda x: x[1]["growth_rates"]["mean"])
        
        print(f"  • Highest Degradation: {highest_growth[0]} ({highest_growth[1]['growth_rates']['mean']:.6f} per token)")
        print(f"  • Lowest Degradation: {lowest_growth[0]} ({lowest_growth[1]['growth_rates']['mean']:.6f} per token)")
        
        # Check if degradation increases with token position
        degradation_trends = []
        for model_name, data in analysis.items():
            if len(data["mean_degradations"]) > 1:
                trend = "increases" if data["growth_rates"]["mean"] > 0 else "decreases" if data["growth_rates"]["mean"] < 0 else "stable"
                degradation_trends.append(f"{model_name}: {trend}")
        
        if degradation_trends:
            print(f"  • Degradation Trends: {', '.join(degradation_trends)}")

def main():
    """Main entry point"""
    experiment = MantissaErrorAnalysisExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_mantissa_error_analysis.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if experiment completed
    return 0

if __name__ == "__main__":
    sys.exit(main()) 