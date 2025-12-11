#!/usr/bin/env python3
"""
Compute uncertainty metrics for test prompts using trained LoRA ensemble.

This script:
1. Loads a trained LoRA ensemble from a directory
2. Loads test data (safe + benign prompts)
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
from datetime import datetime

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.ensemble_lora.inference import load_ensemble_from_directory
from approaches.ensemble_lora.uncertainty import compute_uncertainty_metrics


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
    print("\n[1/3] Loading test data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )

    test_safe_prompts = data['test_safe_prompts']
    test_benign_prompts = data['test_benign_prompts']

    print(f"Loaded {len(test_safe_prompts)} safe test prompts")
    print(f"Loaded {len(test_benign_prompts)} benign test prompts")
    total_test = len(test_safe_prompts) + len(test_benign_prompts)
    print(f"Total test prompts: {total_test}")

    # 2. Load ensemble
    print(f"\n[2/3] Loading ensemble from {args.ensemble_dir}...")
    ensemble_inference = load_ensemble_from_directory(
        base_model_name=constants.MODEL_NAME,
        ensemble_dir=args.ensemble_dir,
        device=constants.DEVICE
    )
    print(f"Loaded ensemble with {len(ensemble_inference.adapter_paths)} adapters")

    # 3. Compute uncertainty metrics
    print(f"\n[3/3] Computing uncertainty metrics...")

    results = {
        "metadata": {
            "ensemble_dir": args.ensemble_dir,
            "base_model": constants.MODEL_NAME,
            "n_adapters": len(ensemble_inference.adapter_paths),
            "n_safe_test": len(test_safe_prompts),
            "n_benign_test": len(test_benign_prompts),
            "timestamp": datetime.now().isoformat()
        },
        "safe": {},
        "benign": {}
    }

    # Process safe prompts
    print("\n  Computing metrics for SAFE prompts...")
    _, safe_probs = ensemble_inference.ensemble_predict(
        test_safe_prompts,
        max_length=constants.MAX_LENGTH
    )
    safe_metrics = compute_uncertainty_metrics(safe_probs)

    # Convert tensors to lists for JSON serialization
    for key, value in safe_metrics.items():
        if isinstance(value, torch.Tensor):
            results["safe"][key] = value.tolist()
        else:
            results["safe"][key] = value

    # Process benign prompts
    print("  Computing metrics for BENIGN prompts...")
    _, benign_probs = ensemble_inference.ensemble_predict(
        test_benign_prompts,
        max_length=constants.MAX_LENGTH
    )
    benign_metrics = compute_uncertainty_metrics(benign_probs)

    # Convert tensors to lists for JSON serialization
    for key, value in benign_metrics.items():
        if isinstance(value, torch.Tensor):
            results["benign"][key] = value.tolist()
        else:
            results["benign"][key] = value

    # Add prompts to results
    results["safe"]["prompts"] = test_safe_prompts
    results["benign"]["prompts"] = test_benign_prompts

    # 4. Save results
    output_file = args.output or os.path.join(args.ensemble_dir, "uncertainty_metrics.json")
    print(f"\n[4/4] Saving results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # 5. Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nSAFE prompts:")
    for metric_name in ["predictive_entropy", "mutual_information", "variance", "mean_confidence"]:
        if metric_name in results["safe"]:
            values = results["safe"][metric_name]
            if isinstance(values, list):
                import numpy as np
                print(f"  {metric_name}:")
                print(f"    mean={np.mean(values):.4f}, std={np.std(values):.4f}")
                print(f"    min={np.min(values):.4f}, max={np.max(values):.4f}")

    print("\nBENIGN prompts:")
    for metric_name in ["predictive_entropy", "mutual_information", "variance", "mean_confidence"]:
        if metric_name in results["benign"]:
            values = results["benign"][metric_name]
            if isinstance(values, list):
                import numpy as np
                print(f"  {metric_name}:")
                print(f"    mean={np.mean(values):.4f}, std={np.std(values):.4f}")
                print(f"    min={np.min(values):.4f}, max={np.max(values):.4f}")

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
