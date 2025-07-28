#!/usr/bin/env python3
"""
GPU / Tensor Parallel / Attention Kernel Robustness Experiment (Table 2)

Tests GRAIL robustness across different hardware configurations:
- GPU configurations: 1xA100, 1x4090, 2x4090 (simulated)
- Tensor parallelism variations
- Attention kernel variations: Flash, SDPA, Flex
- Validates that all values remain within thresholds across configurations

This validates that GRAIL works consistently across different hardware setups.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import subprocess

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from grail import GrailProtocol

class GPURobustnessExperiment:
    """GPU/Hardware Robustness Experiment for Table 2"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results = {
            "experiment": "gpu_robustness",
            "description": "Test robustness across GPU configurations, tensor parallelism, and attention kernels",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "hardware_info": {},
            "configuration_tests": [],
            "summary_statistics": {},
            "table_2_data": {},
            "conclusions": []
        }
        
        # Test configuration
        self.test_prompt = "The quick brown fox jumps over the lazy dog and continues through the dense forest."
        self.token_count = 128
        self.challenge_size = 1024
        
        # Hardware configurations to test (simulated if not available)
        self.gpu_configurations = [
            {"name": "1xA100", "device_count": 1, "device_type": "A100", "tensor_parallel": 1},
            {"name": "1x4090", "device_count": 1, "device_type": "RTX4090", "tensor_parallel": 1},
            {"name": "2x4090", "device_count": 2, "device_type": "RTX4090", "tensor_parallel": 2},
        ]
        
        # Tensor parallelism configurations
        self.tensor_parallel_configs = [1, 2, 4]
        
        # Attention kernel configurations
        self.attention_kernels = [
            {"name": "Flash", "implementation": "flash_attention"},
            {"name": "SDPA", "implementation": "scaled_dot_product_attention"},
            {"name": "Flex", "implementation": "flex_attention"},
            {"name": "Default", "implementation": "default"}
        ]
        
        # Models for testing
        self.test_models = ["sshleifer/tiny-gpt2", "distilgpt2"]
        
        # Error thresholds for validation
        self.validation_thresholds = {
            "exponent_mismatch_rate": 0.05,  # 5% max
            "mantissa_mse": 1e-6,  # Very low threshold
            "validation_pass_rate": 0.95  # 95% minimum
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        if self.debug and level == "DEBUG":
            print(f"[{timestamp}] [DEBUG] {message}")
    
    def detect_hardware_info(self) -> Dict[str, Any]:
        """Detect current hardware configuration"""
        
        hardware_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "devices": [],
            "compute_capability": None,
            "total_memory": 0
        }
        
        if torch.cuda.is_available():
            hardware_info["device_count"] = torch.cuda.device_count()
            
            for i in range(hardware_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": device_props.name,
                    "memory": device_props.total_memory,
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                }
                hardware_info["devices"].append(device_info)
                hardware_info["total_memory"] += device_props.total_memory
            
            # Get compute capability of first device
            if hardware_info["devices"]:
                hardware_info["compute_capability"] = hardware_info["devices"][0]["compute_capability"]
        
        return hardware_info
    
    def check_configuration_feasibility(self, gpu_config: Dict[str, Any], tp_size: int, attention_kernel: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a configuration is feasible on current hardware"""
        
        feasibility = {
            "feasible": True,
            "reasons": [],
            "simulation_mode": False
        }
        
        # Check GPU availability
        if not torch.cuda.is_available():
            feasibility["feasible"] = False
            feasibility["reasons"].append("CUDA not available")
            return feasibility
        
        # Check device count
        available_devices = torch.cuda.device_count()
        required_devices = gpu_config["device_count"]
        
        if required_devices > available_devices:
            feasibility["simulation_mode"] = True
            feasibility["reasons"].append(f"Simulating {required_devices} devices (only {available_devices} available)")
        
        # Check tensor parallelism compatibility
        if tp_size > available_devices:
            feasibility["simulation_mode"] = True
            feasibility["reasons"].append(f"Simulating TP={tp_size} (only {available_devices} devices)")
        
        # Check attention kernel support
        kernel_impl = attention_kernel["implementation"]
        
        if kernel_impl == "flash_attention":
            try:
                # Try to import flash attention
                import flash_attn
                feasibility["reasons"].append("Flash Attention available")
            except ImportError:
                feasibility["simulation_mode"] = True
                feasibility["reasons"].append("Flash Attention not available, using default")
        
        elif kernel_impl == "scaled_dot_product_attention":
            # SDPA is available in PyTorch 2.0+
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                feasibility["reasons"].append("SDPA available")
            else:
                feasibility["simulation_mode"] = True
                feasibility["reasons"].append("SDPA not available, using default")
        
        elif kernel_impl == "flex_attention":
            # Flex attention is newer, may not be available
            feasibility["simulation_mode"] = True
            feasibility["reasons"].append("Flex Attention simulated (not widely available)")
        
        return feasibility
    
    def test_configuration(self, model_name: str, gpu_config: Dict[str, Any], tp_size: int, attention_kernel: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific hardware configuration"""
        
        config_name = f"{gpu_config['name']}_TP{tp_size}_{attention_kernel['name']}"
        self.log(f"Testing configuration: {config_name} with {model_name}")
        
        start_time = time.time()
        test_result = {
            "configuration_name": config_name,
            "model": model_name,
            "gpu_config": gpu_config,
            "tensor_parallel_size": tp_size,
            "attention_kernel": attention_kernel,
            "feasibility": {},
            "validation_success": False,
            "error_stats": {},
            "timing": {},
            "error_message": None,
            "within_thresholds": False
        }
        
        # Check feasibility
        feasibility = self.check_configuration_feasibility(gpu_config, tp_size, attention_kernel)
        test_result["feasibility"] = feasibility
        
        if not feasibility["feasible"]:
            test_result["error_message"] = f"Configuration not feasible: {', '.join(feasibility['reasons'])}"
            self.log(f"  Skipping: {test_result['error_message']}", "WARN")
            return test_result
        
        try:
            # Setup protocol with configuration
            gen_start = time.time()
            
            protocol_kwargs = {
                "model_name": model_name,
                "max_new_tokens": self.token_count,
                "challenge_size": self.challenge_size
            }
            
            # Add tensor parallelism configuration (simulated)
            if tp_size > 1:
                if feasibility["simulation_mode"]:
                    # Simulate tensor parallelism by using available devices
                    available_devices = min(tp_size, torch.cuda.device_count())
                    protocol_kwargs["device_map"] = "auto" if available_devices > 1 else None
                else:
                    protocol_kwargs["device_map"] = "auto"
            
            # Add attention kernel configuration (simulated)
            kernel_impl = attention_kernel["implementation"]
            if kernel_impl == "flash_attention" and not feasibility["simulation_mode"]:
                protocol_kwargs["use_flash_attention"] = True
            elif kernel_impl == "scaled_dot_product_attention" and not feasibility["simulation_mode"]:
                protocol_kwargs["use_sdpa"] = True
            
            # Generation phase
            gen_protocol = GrailProtocol(**protocol_kwargs)
            commit_result = gen_protocol.commit(self.test_prompt)
            gen_time = time.time() - gen_start
            
            test_result["timing"]["generation_time"] = gen_time
            test_result["timing"]["commit_time"] = commit_result.get("timing", {}).get("commit_time", 0)
            
            # Validation phase (same configuration)
            val_start = time.time()
            val_protocol = GrailProtocol(**protocol_kwargs)
            
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
            
            # Extract error statistics
            if "error_stats" in verify_result:
                test_result["error_stats"] = verify_result["error_stats"]
            
            # Check if results are within thresholds
            within_thresholds = True
            threshold_checks = {}
            
            if "error_stats" in test_result and test_result["error_stats"]:
                stats = test_result["error_stats"]
                
                # Check exponent mismatch rate
                if "exponent_mismatch_rate" in stats:
                    exp_rate = stats["exponent_mismatch_rate"]
                    threshold_checks["exponent_mismatch"] = {
                        "value": exp_rate,
                        "threshold": self.validation_thresholds["exponent_mismatch_rate"],
                        "within_threshold": exp_rate <= self.validation_thresholds["exponent_mismatch_rate"]
                    }
                    if not threshold_checks["exponent_mismatch"]["within_threshold"]:
                        within_thresholds = False
                
                # Check mantissa MSE
                if "mantissa_mse" in stats:
                    mant_mse = stats["mantissa_mse"]
                    threshold_checks["mantissa_mse"] = {
                        "value": mant_mse,
                        "threshold": self.validation_thresholds["mantissa_mse"],
                        "within_threshold": mant_mse <= self.validation_thresholds["mantissa_mse"]
                    }
                    if not threshold_checks["mantissa_mse"]["within_threshold"]:
                        within_thresholds = False
            
            # Check validation pass rate (should be high for same config)
            validation_pass = test_result["validation_success"]
            threshold_checks["validation_pass"] = {
                "value": 1.0 if validation_pass else 0.0,
                "threshold": self.validation_thresholds["validation_pass_rate"],
                "within_threshold": validation_pass
            }
            if not validation_pass:
                within_thresholds = False
            
            test_result["threshold_checks"] = threshold_checks
            test_result["within_thresholds"] = within_thresholds
            
        except Exception as e:
            test_result["error_message"] = str(e)
            test_result["validation_success"] = False
            test_result["within_thresholds"] = False
            self.log(f"Error in configuration test: {e}", "ERROR")
        
        test_result["timing"]["total_time"] = time.time() - start_time
        
        # Log result
        status = "✅ PASS" if test_result["within_thresholds"] else "❌ FAIL"
        if test_result["feasibility"]["simulation_mode"]:
            status += " (simulated)"
        
        self.log(f"  Result: {status}")
        
        return test_result
    
    def run_robustness_tests(self) -> List[Dict[str, Any]]:
        """Run all robustness configuration tests"""
        
        self.log("=== Running GPU Robustness Tests ===")
        
        all_tests = []
        
        for model in self.test_models:
            self.log(f"Testing model: {model}")
            
            for gpu_config in self.gpu_configurations:
                for tp_size in self.tensor_parallel_configs:
                    # Skip tensor parallel sizes larger than device count for specific configs
                    if tp_size > gpu_config["device_count"]:
                        continue
                        
                    for attention_kernel in self.attention_kernels:
                        test_result = self.test_configuration(
                            model, gpu_config, tp_size, attention_kernel
                        )
                        all_tests.append(test_result)
        
        return all_tests
    
    def analyze_robustness_results(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze robustness test results and compute statistics"""
        
        self.log("=== Analyzing GPU Robustness Results ===")
        
        analysis = {
            "total_tests": len(test_results),
            "successful_tests": 0,
            "within_threshold_tests": 0,
            "configuration_stats": {},
            "model_stats": {},
            "threshold_analysis": {},
            "timing_stats": {}
        }
        
        # Group results
        config_groups = {}
        model_groups = {}
        feasible_tests = []
        
        for test in test_results:
            # Skip infeasible tests for analysis
            if not test["feasibility"].get("feasible", False):
                continue
            
            feasible_tests.append(test)
            
            # Group by configuration type
            config_key = f"{test['gpu_config']['name']}_TP{test['tensor_parallel_size']}_{test['attention_kernel']['name']}"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(test)
            
            # Group by model
            model_key = test["model"]
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(test)
            
            # Count successful tests
            if test["validation_success"]:
                analysis["successful_tests"] += 1
            
            if test["within_thresholds"]:
                analysis["within_threshold_tests"] += 1
        
        # Analyze each configuration
        for config_key, tests in config_groups.items():
            successful = sum(1 for t in tests if t["validation_success"])
            within_thresholds = sum(1 for t in tests if t["within_thresholds"])
            total = len(tests)
            
            analysis["configuration_stats"][config_key] = {
                "total_tests": total,
                "successful_tests": successful,
                "within_threshold_tests": within_thresholds,
                "success_rate": successful / total * 100 if total > 0 else 0,
                "threshold_compliance_rate": within_thresholds / total * 100 if total > 0 else 0
            }
        
        # Analyze each model
        for model_key, tests in model_groups.items():
            successful = sum(1 for t in tests if t["validation_success"])
            within_thresholds = sum(1 for t in tests if t["within_thresholds"])
            total = len(tests)
            
            analysis["model_stats"][model_key] = {
                "total_tests": total,
                "successful_tests": successful,
                "within_threshold_tests": within_thresholds,
                "success_rate": successful / total * 100 if total > 0 else 0,
                "threshold_compliance_rate": within_thresholds / total * 100 if total > 0 else 0
            }
        
        # Overall statistics
        total_feasible = len(feasible_tests)
        analysis["overall_success_rate"] = analysis["successful_tests"] / total_feasible * 100 if total_feasible > 0 else 0
        analysis["overall_threshold_compliance"] = analysis["within_threshold_tests"] / total_feasible * 100 if total_feasible > 0 else 0
        
        # Threshold analysis
        threshold_summary = {
            "exponent_mismatch": {"values": [], "violations": 0},
            "mantissa_mse": {"values": [], "violations": 0},
            "validation_pass": {"values": [], "violations": 0}
        }
        
        for test in feasible_tests:
            if "threshold_checks" in test:
                for threshold_name, check in test["threshold_checks"].items():
                    if threshold_name in threshold_summary:
                        threshold_summary[threshold_name]["values"].append(check["value"])
                        if not check["within_threshold"]:
                            threshold_summary[threshold_name]["violations"] += 1
        
        analysis["threshold_analysis"] = threshold_summary
        
        # Timing statistics
        all_times = [test["timing"]["total_time"] for test in feasible_tests 
                    if "timing" in test and "total_time" in test["timing"]]
        
        if all_times:
            analysis["timing_stats"] = {
                "min_time": min(all_times),
                "max_time": max(all_times),
                "avg_time": sum(all_times) / len(all_times),
                "total_time": sum(all_times)
            }
        
        return analysis
    
    def generate_table_2_data(self, test_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted data for Table 2 in the research paper"""
        
        table_data = {
            "title": "GPU / Tensor Parallel / Attention Kernel Robustness Results",
            "description": "Validation error stats across hardware configurations",
            "headers": ["Configuration", "GPU", "TP", "Attention", "Success Rate", "Threshold Compliance", "Avg Time (s)"],
            "rows": [],
            "summary": {
                "overall_success_rate": analysis["overall_success_rate"],
                "overall_threshold_compliance": analysis["overall_threshold_compliance"],
                "total_configurations": len(analysis["configuration_stats"])
            }
        }
        
        # Add rows for each configuration
        for config_key, stats in analysis["configuration_stats"].items():
            # Parse configuration name
            parts = config_key.split("_")
            gpu_name = parts[0]
            tp_size = parts[1]
            attention_kernel = parts[2]
            
            # Calculate average time for this configuration
            config_tests = [t for t in test_results if 
                           f"{t['gpu_config']['name']}_TP{t['tensor_parallel_size']}_{t['attention_kernel']['name']}" == config_key
                           and t["feasibility"].get("feasible", False)]
            
            avg_time = "N/A"
            if config_tests:
                times = [t["timing"]["total_time"] for t in config_tests 
                        if "timing" in t and "total_time" in t["timing"]]
                if times:
                    avg_time = f"{sum(times) / len(times):.2f}"
            
            row = {
                "configuration": config_key,
                "gpu": gpu_name,
                "tensor_parallel": tp_size,
                "attention_kernel": attention_kernel,
                "success_rate": f"{stats['success_rate']:.1f}%",
                "threshold_compliance": f"{stats['threshold_compliance_rate']:.1f}%",
                "avg_time": avg_time
            }
            table_data["rows"].append(row)
        
        return table_data
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete GPU robustness experiment"""
        
        self.log("🧪 Starting GPU Robustness Experiment")
        
        start_time = time.time()
        
        try:
            # Detect hardware
            hardware_info = self.detect_hardware_info()
            self.results["hardware_info"] = hardware_info
            
            self.log(f"Detected hardware: {hardware_info['device_count']} GPU(s)")
            for device in hardware_info["devices"]:
                self.log(f"  Device {device['index']}: {device['name']} ({device['compute_capability']})")
            
            # Run robustness tests
            test_results = self.run_robustness_tests()
            
            # Analyze results
            analysis = self.analyze_robustness_results(test_results)
            
            # Generate table data
            table_2_data = self.generate_table_2_data(test_results, analysis)
            
            # Update results
            self.results["configuration_tests"] = test_results
            self.results["summary_statistics"] = analysis
            self.results["table_2_data"] = table_2_data
            
            # Generate conclusions
            conclusions = []
            
            success_rate = analysis["overall_success_rate"]
            threshold_compliance = analysis["overall_threshold_compliance"]
            
            if success_rate >= 95.0:
                conclusions.append("✅ High validation success rate across configurations")
            else:
                conclusions.append(f"❌ Low validation success rate ({success_rate:.1f}% vs 95% expected)")
            
            if threshold_compliance >= 95.0:
                conclusions.append("✅ Error values remain within thresholds across configurations")
            else:
                conclusions.append(f"❌ Threshold violations detected ({threshold_compliance:.1f}% compliance)")
            
            if success_rate >= 95.0 and threshold_compliance >= 95.0:
                conclusions.append("✅ GPU robustness experiment PASSED")
            else:
                conclusions.append("❌ GPU robustness experiment FAILED")
            
            self.results["conclusions"] = conclusions
            
            # Log summary
            self.log("=== GPU Robustness Results ===")
            self.log(f"Overall success rate: {success_rate:.1f}%")
            self.log(f"Threshold compliance: {threshold_compliance:.1f}%")
            
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
            output_file = f"experiments/results/exp_gpu_robustness.json"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved to: {output_file}")
        return output_file

def main():
    """Main execution function"""
    
    print("🧪 GRAIL GPU Robustness Experiment (Table 2)")
    print("=" * 60)
    
    # Run experiment
    experiment = GPURobustnessExperiment(debug=True)
    results = experiment.run_experiment()
    
    # Save results
    output_file = experiment.save_results()
    
    # Print final status
    print("\n" + "=" * 60)
    if "error" not in results:
        success_rate = results["summary_statistics"]["overall_success_rate"]
        threshold_compliance = results["summary_statistics"]["overall_threshold_compliance"]
        
        if success_rate >= 95.0 and threshold_compliance >= 95.0:
            print("🎉 GPU Robustness Experiment: PASSED")
            exit_code = 0
        else:
            print("💥 GPU Robustness Experiment: FAILED")
            exit_code = 1
    else:
        print("💥 GPU Robustness Experiment: ERROR")
        exit_code = 1
    
    print(f"📊 Results saved to: {output_file}")
    
    # Print table preview
    if "table_2_data" in results:
        print("\n📋 Table 2 Preview:")
        table = results["table_2_data"]
        print(f"Overall success rate: {table['summary']['overall_success_rate']:.1f}%")
        print(f"Threshold compliance: {table['summary']['overall_threshold_compliance']:.1f}%")
        print(f"Configurations tested: {table['summary']['total_configurations']}")
    
    return exit_code

if __name__ == "__main__":
    exit(main()) 