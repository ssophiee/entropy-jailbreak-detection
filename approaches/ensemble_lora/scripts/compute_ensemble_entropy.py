#!/usr/bin/env python3
"""
Compute uncertainty metrics for test prompts using trained LoRA ensemble.

This script:
1. Loads a trained LoRA ensemble from a directory
2. Loads test data (safe + harmful prompts)
3. Computes uncertainty metrics (entropy, mutual information, etc.)
4. Saves results for analysis

Usage:
    python compute_ensemble_entropy.py --ensemble_dir saved_models/ensemble_lora_123456
    python compute_ensemble_entropy.py --ensemble_dir saved_models/my_ensemble --output results.json
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from scipy import stats

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.ensemble_lora.inference import load_ensemble_from_directory
from approaches.ensemble_lora.uncertainty import compute_uncertainty_metrics


def compute_statistics(safe_values, harmful_values, metric_name):
    """Compute and print statistical analysis for a given metric"""
    safe_values = np.array(safe_values)
    harmful_values = np.array(harmful_values)

    safe_mean = np.mean(safe_values)
    safe_std = np.std(safe_values)
    harmful_mean = np.mean(harmful_values)
    harmful_std = np.std(harmful_values)

    print(f"\n{metric_name.upper().replace('_', ' ')}:")
    print(f"  Safe Prompts:")
    print(f"    Mean: {safe_mean:.4f} ± {safe_std:.4f}")
    print(f"    Min: {np.min(safe_values):.4f}, Max: {np.max(safe_values):.4f}")

    print(f"  Harmful Prompts:")
    print(f"    Mean: {harmful_mean:.4f} ± {harmful_std:.4f}")
    print(f"    Min: {np.min(harmful_values):.4f}, Max: {np.max(harmful_values):.4f}")

    # T-test
    t_stat, p_value = stats.ttest_ind(harmful_values, safe_values)
    print(f"  Statistical Test (Independent t-test):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4e}")

    if p_value < 0.05:
        print(f"    ✓ Significant difference (p < 0.05)")
        if harmful_mean > safe_mean:
            print(f"    → Harmful prompts have HIGHER {metric_name} (supports hypothesis)")
        else:
            print(f"    → Harmful prompts have LOWER {metric_name}")
    else:
        print(f"    ✗ No significant difference (p ≥ 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((safe_std**2 + harmful_std**2) / 2)
    cohens_d = (harmful_mean - safe_mean) / pooled_std if pooled_std > 0 else 0
    print(f"    Cohen's d: {cohens_d:.4f}")

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }


def main(args):
    print("=" * 80)
    print("ENSEMBLE UNCERTAINTY COMPUTATION SCRIPT")
    print("=" * 80)
    print(f"Device: {constants.DEVICE}")
    print(f"Base Model: {constants.MODEL_NAME}")
    print(f"Ensemble Directory: {args.ensemble_dir}")
    print(f"Max Length: {constants.MAX_LENGTH}")
    print("=" * 80)

    # 1. Load test data
    print("\n[1/5] Loading test data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )

    test_safe_prompts = data['safe_test']
    test_harmful_prompts = data['harmful_test']
    print(f"Loaded {len(test_safe_prompts)} safe test prompts")
    print(f"Loaded {len(test_harmful_prompts)} harmful test prompts")
    total_test = len(test_safe_prompts) + len(test_harmful_prompts)
    print(f"Total test prompts: {total_test}")

    # 2. Load ensemble
    print(f"\n[2/5] Loading ensemble from {args.ensemble_dir}...")
    ensemble_inference = load_ensemble_from_directory(
        base_model_name=constants.MODEL_NAME,
        ensemble_dir=args.ensemble_dir,
        device=constants.DEVICE
    )
    print(f"Loaded ensemble with {len(ensemble_inference.adapter_paths)} adapters")

    # 3. Compute uncertainty metrics
    print(f"\n[3/5] Computing uncertainty metrics...")

    results = {
        "metadata": {
            "ensemble_dir": args.ensemble_dir,
            "base_model": constants.MODEL_NAME,
            "n_adapters": len(ensemble_inference.adapter_paths),
            "n_safe_test": len(test_safe_prompts),
            "n_harmful_test": len(test_harmful_prompts),
            "timestamp": datetime.now().isoformat()
        },
        "safe": {},
        "harmful": {}
    }

    # Process safe prompts
    print("\n  Computing metrics for SAFE prompts...")
    import sys
    sys.stdout.flush()
    _, safe_probs = ensemble_inference.ensemble_predict(
        test_safe_prompts,
        max_length=constants.MAX_LENGTH
    )
    sys.stdout.flush()

    # Debug: Check for NaN/inf in probabilities
    print(f"  Safe probs shape: {safe_probs.shape}")
    print(f"  Safe probs has NaN: {torch.isnan(safe_probs).any().item()}")
    print(f"  Safe probs has Inf: {torch.isinf(safe_probs).any().item()}")
    print(f"  Safe probs min: {safe_probs.min().item():.6f}, max: {safe_probs.max().item():.6f}")
    print(f"  Safe probs sum along vocab (should be ~1): {safe_probs[0, 0, :].sum().item():.6f}")

    safe_metrics = compute_uncertainty_metrics(safe_probs)

    # Convert tensors to lists for JSON serialization
    for key, value in safe_metrics.items():
        if isinstance(value, torch.Tensor):
            results["safe"][key] = value.tolist()
        else:
            results["safe"][key] = value

    # Process harmful prompts
    print("  Computing metrics for HARMFUL prompts...")
    _, harmful_probs = ensemble_inference.ensemble_predict(
        test_harmful_prompts,
        max_length=constants.MAX_LENGTH
    )

    # Debug: Check for NaN/inf in probabilities
    print(f"  Harmful probs shape: {harmful_probs.shape}")
    print(f"  Harmful probs has NaN: {torch.isnan(harmful_probs).any().item()}")
    print(f"  Harmful probs has Inf: {torch.isinf(harmful_probs).any().item()}")
    print(f"  Harmful probs min: {harmful_probs.min().item():.6f}, max: {harmful_probs.max().item():.6f}")
    print(f"  Harmful probs sum along vocab (should be ~1): {harmful_probs[0, 0, :].sum().item():.6f}")

    harmful_metrics = compute_uncertainty_metrics(harmful_probs)

    # Convert tensors to lists for JSON serialization
    for key, value in harmful_metrics.items():
        if isinstance(value, torch.Tensor):
            results["harmful"][key] = value.tolist()
        else:
            results["harmful"][key] = value

    # Add prompts to results
    results["safe"]["prompts"] = test_safe_prompts
    results["harmful"]["prompts"] = test_harmful_prompts

    # 4. Compute statistical analysis for each metric
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Store statistical results for all metrics
    results["statistical_tests"] = {}

    # Run statistical tests for all metrics
    metric_names = ["predictive_entropy", "mutual_information", "variance",
                   "mean_confidence", "intersection_probs_entropy", "mean_entropy"]

    for metric_name in metric_names:
        if metric_name in results["safe"] and metric_name in results["harmful"]:
            safe_values = results["safe"][metric_name]
            harmful_values = results["harmful"][metric_name]

            if isinstance(safe_values, list) and isinstance(harmful_values, list):
                stats_result = compute_statistics(safe_values, harmful_values, metric_name)
                results["statistical_tests"][metric_name] = stats_result

    # 5. Save results
    output_file = args.output or os.path.join(args.ensemble_dir, "uncertainty_metrics.json")
    print(f"\n[5/5] Saving results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute uncertainty metrics for test prompts using LoRA ensemble"
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        required=True,
        help="Directory containing trained ensemble adapters"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <ensemble_dir>/uncertainty_metrics.json)"
    )

    args = parser.parse_args()

    # Validate ensemble directory exists
    if not os.path.exists(args.ensemble_dir):
        raise ValueError(f"Ensemble directory does not exist: {args.ensemble_dir}")

    main(args)
