#!/usr/bin/env python3
"""
Experiment: Attack Resistance
Tests resistance against various attack scenarios:
- Token manipulation
- Signature tampering
- S-values modification
- Index manipulation
"""

import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to import grail
sys.path.append(str(Path(__file__).parent.parent))
from grail import Prover, Verifier

class AttackResistanceExperiment:
    """Test resistance against various attack scenarios"""
    
    def __init__(self):
        self.results = []
        self.model = "sshleifer/tiny-gpt2"
        self.test_prompt = "The quick brown fox jumps over the lazy dog"
    
    def test_token_manipulation(self) -> Dict:
        """Test resistance against token manipulation attacks"""
        print("🔒 Testing Token Manipulation Resistance")
        
        attacks = []
        
        try:
            # Generate original valid proof
            prover = Prover(self.model)
            verifier = Verifier(self.model)
            
            original_commit = prover.commit(self.test_prompt, max_new_tokens=20)
            proof_pkg = prover.open(k=8)
            
            # Verify original (should pass)
            original_result = verifier.verify(original_commit, proof_pkg, prover.secret_key)
            
            # Attack 1: Modify single token
            attack1_commit = original_commit.copy()
            attack1_commit["tokens"] = original_commit["tokens"].copy()
            if len(attack1_commit["tokens"]) > 5:
                attack1_commit["tokens"][5] = 999999  # Invalid token
            
            try:
                attack1_result = verifier.verify(attack1_commit, proof_pkg, prover.secret_key)
                attack1_blocked = not attack1_result
            except Exception:
                attack1_blocked = True  # Exception means attack was blocked
            
            attacks.append({
                "attack": "single_token_modification",
                "description": "Modified token at index 5",
                "blocked": attack1_blocked,
                "expected_blocked": True
            })
            
            # Attack 2: Append extra tokens
            attack2_commit = original_commit.copy()
            attack2_commit["tokens"] = original_commit["tokens"] + [1, 2, 3]
            
            try:
                attack2_result = verifier.verify(attack2_commit, proof_pkg, prover.secret_key)
                attack2_blocked = not attack2_result
            except Exception:
                attack2_blocked = True
            
            attacks.append({
                "attack": "token_appending",
                "description": "Appended 3 extra tokens",
                "blocked": attack2_blocked,
                "expected_blocked": True
            })
            
            # Attack 3: Remove tokens
            attack3_commit = original_commit.copy()
            attack3_commit["tokens"] = original_commit["tokens"][:-3]
            
            try:
                attack3_result = verifier.verify(attack3_commit, proof_pkg, prover.secret_key)
                attack3_blocked = not attack3_result
            except Exception:
                attack3_blocked = True
            
            attacks.append({
                "attack": "token_removal",
                "description": "Removed last 3 tokens",
                "blocked": attack3_blocked,
                "expected_blocked": True
            })
            
        except Exception as e:
            return {
                "test": "token_manipulation",
                "error": str(e),
                "attacks": []
            }
        
        return {
            "test": "token_manipulation",
            "original_verification": original_result,
            "attacks": attacks,
            "total_attacks": len(attacks),
            "blocked_attacks": sum(1 for a in attacks if a["blocked"]),
            "success_rate": sum(1 for a in attacks if a["blocked"] == a["expected_blocked"]) / len(attacks) * 100
        }
    
    def test_signature_tampering(self) -> Dict:
        """Test resistance against signature tampering"""
        print("🔐 Testing Signature Tampering Resistance")
        
        attacks = []
        
        try:
            prover = Prover(self.model)
            verifier = Verifier(self.model)
            
            original_commit = prover.commit(self.test_prompt, max_new_tokens=16)
            proof_pkg = prover.open(k=6)
            
            # Attack 1: Modify s_vals (should fail signature check)
            attack1_commit = original_commit.copy()
            attack1_commit["s_vals"] = original_commit["s_vals"].copy()
            attack1_commit["s_vals"][0] = 999999999  # Modify first s_val
            
            try:
                attack1_result = verifier.verify(attack1_commit, proof_pkg, prover.secret_key)
                attack1_blocked = not attack1_result
            except Exception:
                attack1_blocked = True
            
            attacks.append({
                "attack": "s_vals_modification",
                "description": "Modified first s_val",
                "blocked": attack1_blocked,
                "expected_blocked": True
            })
            
            # Attack 2: Wrong signature
            attack2_commit = original_commit.copy()
            attack2_commit["signature"] = "00" * 32  # Invalid signature
            
            try:
                attack2_result = verifier.verify(attack2_commit, proof_pkg, prover.secret_key)
                attack2_blocked = not attack2_result
            except Exception:
                attack2_blocked = True
            
            attacks.append({
                "attack": "signature_replacement",
                "description": "Replaced with invalid signature",
                "blocked": attack2_blocked,
                "expected_blocked": True
            })
            
            # Attack 3: Truncate s_vals
            attack3_commit = original_commit.copy()
            attack3_commit["s_vals"] = original_commit["s_vals"][:-2]  # Remove last 2 s_vals
            
            try:
                attack3_result = verifier.verify(attack3_commit, proof_pkg, prover.secret_key)
                attack3_blocked = not attack3_result
            except Exception:
                attack3_blocked = True
            
            attacks.append({
                "attack": "s_vals_truncation",
                "description": "Removed last 2 s_vals",
                "blocked": attack3_blocked,
                "expected_blocked": True
            })
            
        except Exception as e:
            return {
                "test": "signature_tampering",
                "error": str(e),
                "attacks": []
            }
        
        return {
            "test": "signature_tampering",
            "attacks": attacks,
            "total_attacks": len(attacks),
            "blocked_attacks": sum(1 for a in attacks if a["blocked"]),
            "success_rate": sum(1 for a in attacks if a["blocked"] == a["expected_blocked"]) / len(attacks) * 100
        }
    
    def test_challenge_manipulation(self) -> Dict:
        """Test resistance against challenge index manipulation"""
        print("🎲 Testing Challenge Manipulation Resistance")
        
        attacks = []
        
        try:
            prover = Prover(self.model)
            verifier = Verifier(self.model)
            
            original_commit = prover.commit(self.test_prompt, max_new_tokens=16)
            original_proof = prover.open(k=8)
            
            # Attack 1: Modify challenge indices
            attack1_proof = original_proof.copy()
            attack1_proof["indices"] = [0, 1, 2, 3, 4, 5, 6, 7]  # Different indices
            
            try:
                attack1_result = verifier.verify(original_commit, attack1_proof, prover.secret_key)
                attack1_blocked = not attack1_result
            except Exception:
                attack1_blocked = True
            
            attacks.append({
                "attack": "index_modification",
                "description": "Modified challenge indices",
                "blocked": attack1_blocked,
                "expected_blocked": True
            })
            
            # Attack 2: Reduce number of indices
            attack2_proof = original_proof.copy()
            attack2_proof["indices"] = original_proof["indices"][:4]  # Only half the indices
            
            try:
                attack2_result = verifier.verify(original_commit, attack2_proof, prover.secret_key)
                attack2_blocked = not attack2_result
            except Exception:
                attack2_blocked = True
            
            attacks.append({
                "attack": "index_reduction",
                "description": "Reduced number of challenge indices",
                "blocked": attack2_blocked,
                "expected_blocked": True
            })
            
        except Exception as e:
            return {
                "test": "challenge_manipulation",
                "error": str(e),
                "attacks": []
            }
        
        return {
            "test": "challenge_manipulation",
            "attacks": attacks,
            "total_attacks": len(attacks),
            "blocked_attacks": sum(1 for a in attacks if a["blocked"]),
            "success_rate": sum(1 for a in attacks if a["blocked"] == a["expected_blocked"]) / len(attacks) * 100
        }
    
    def test_replay_attack(self) -> Dict:
        """Test resistance against replay attacks"""
        print("🔄 Testing Replay Attack Resistance")
        
        try:
            prover = Prover(self.model)
            verifier = Verifier(self.model)
            
            # Generate first proof
            commit1 = prover.commit(self.test_prompt, max_new_tokens=16)
            proof1 = prover.open(k=8)
            
            # Generate second proof (different randomness)
            commit2 = prover.commit(self.test_prompt, max_new_tokens=16)
            proof2 = prover.open(k=8)
            
            # Try to use proof1 with commit2 (replay attack)
            try:
                replay_result = verifier.verify(commit2, proof1, prover.secret_key)
                replay_blocked = not replay_result
            except Exception:
                replay_blocked = True
            
            return {
                "test": "replay_attack",
                "description": "Attempted to reuse proof with different commit",
                "blocked": replay_blocked,
                "expected_blocked": True,
                "success_rate": 100.0 if replay_blocked else 0.0
            }
            
        except Exception as e:
            return {
                "test": "replay_attack",
                "error": str(e),
                "success_rate": 0.0
            }
    
    def run_experiment(self) -> Dict:
        """Run all attack resistance tests"""
        print("⚔️  Attack Resistance Experiment")
        print("=" * 50)
        print("Testing resistance against various attack scenarios")
        print()
        
        experiment_start = time.time()
        
        # Run all attack tests
        token_test = self.test_token_manipulation()
        signature_test = self.test_signature_tampering()
        challenge_test = self.test_challenge_manipulation()
        replay_test = self.test_replay_attack()
        
        total_duration = time.time() - experiment_start
        
        # Collect all attack results
        all_attacks = []
        if "attacks" in token_test:
            all_attacks.extend(token_test["attacks"])
        if "attacks" in signature_test:
            all_attacks.extend(signature_test["attacks"])
        if "attacks" in challenge_test:
            all_attacks.extend(challenge_test["attacks"])
        
        total_attacks = len(all_attacks)
        blocked_attacks = sum(1 for a in all_attacks if a.get("blocked", False))
        
        summary = {
            "experiment": "attack_resistance",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "model_tested": self.model,
            "test_prompt": self.test_prompt,
            "overall_stats": {
                "total_attacks": total_attacks,
                "blocked_attacks": blocked_attacks,
                "resistance_rate": (blocked_attacks / total_attacks * 100) if total_attacks > 0 else 0
            },
            "test_results": {
                "token_manipulation": token_test,
                "signature_tampering": signature_test,
                "challenge_manipulation": challenge_test,
                "replay_attack": replay_test
            }
        }
        
        print()
        print("📊 Summary:")
        print(f"  Total attacks tested: {total_attacks}")
        print(f"  Attacks blocked: {blocked_attacks}")
        print(f"  Resistance rate: {summary['overall_stats']['resistance_rate']:.1f}%")
        print(f"  Token manipulation: {token_test.get('success_rate', 0):.1f}%")
        print(f"  Signature tampering: {signature_test.get('success_rate', 0):.1f}%")
        print(f"  Challenge manipulation: {challenge_test.get('success_rate', 0):.1f}%")
        print(f"  Replay attack: {replay_test.get('success_rate', 0):.1f}%")
        
        return summary

def main():
    """Main entry point"""
    experiment = AttackResistanceExperiment()
    results = experiment.run_experiment()
    
    # Save results
    output_file = Path(__file__).parent / "results" / "exp_attack_resistance.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return success if resistance rate is high
    resistance_rate = results["overall_stats"]["resistance_rate"]
    return 0 if resistance_rate >= 95.0 else 1

if __name__ == "__main__":
    sys.exit(main()) 