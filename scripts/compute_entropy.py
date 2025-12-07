#!/usr/bin/env python3
"""
Script for computing predictive entropy on safe and adversarial test prompts.

This script:
1. Loads the fine-tuned LoRA model and Fisher information matrix
2. Loads test data (safe and harmful prompts)
3. Computes predictive entropy for both categories
4. Performs statistical analysis (t-test)
5. Saves results to a JSON file

Usage:
    python compute_entropy.py --model_path saved_models/your_model_name
"""

import os
import json
import torch
import argparse
import numpy as np
from scipy import stats
from datetime import datetime

import os, sys
this_dir = os.path.dirname(__file__)           # scripts/
repo_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from src.training import setup_model_and_lora
from src.uncertainty import compute_uncertainty_metrics


def load_fisher_matrix(fisher_path):
    """Load Fisher diagonal matrix from saved checkpoint"""
    print(f"Loading Fisher matrix from: {fisher_path}")
    checkpoint = torch.load(fisher_path, map_location=constants.DEVICE)
    fisher_diag = checkpoint['fisher_diag']
    print(f"✓ Loaded Fisher matrix with {len(fisher_diag)} parameters")
    return fisher_diag


def main(args):
    print("="*80)
    print("PREDICTIVE ENTROPY COMPUTATION SCRIPT")
    print("="*80)
    print(f"Device: {constants.DEVICE}")
    print(f"Model path: {args.model_path}")
    print(f"N posterior samples: {constants.N_POSTERIOR_SAMPLES}")
    print(f"Temperature: {args.temperature}")
    print("="*80)

    # 1. Load test data
    print("\n[1/5] Loading test data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )
    safe_test = data['safe_test']
    harmful_test = data['harmful_test']
    print(f"Loaded {len(safe_test)} safe test prompts")
    print(f"Loaded {len(harmful_test)} harmful test prompts")

    # 2. Load fine-tuned model
    print("\n[2/5] Loading fine-tuned LoRA model...")
    model, tokenizer = setup_model_and_lora(
        constants.MODEL_NAME,
        constants.DEVICE,
        lora_rank=constants.LORA_RANK
    )

    # Load the fine-tuned adapter weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model.base_model, args.model_path)
    model.to(constants.DEVICE)
    model.eval()

    # Convert to float32 for stability during entropy computation
    model = model.float()
    print("✓ Model loaded and converted to float32")

    # 3. Load Fisher matrix
    print("\n[3/5] Loading Fisher information matrix...")
    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    fisher_diag = load_fisher_matrix(fisher_path)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Compute entropy for safe prompts
    print("\n[4/5] Computing predictive entropy for SAFE prompts...")
    safe_entropies = compute_predictive_entropy(
        model=model,
        prompts=safe_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=constants.N_POSTERIOR_SAMPLES,
        temperature=args.temperature,
        device=constants.DEVICE
    )

    # Clear cache between computations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Compute entropy for adversarial prompts
    print("\n[5/5] Computing predictive entropy for ADVERSARIAL prompts...")
    adv_entropies = compute_uncertainty_metrics(
        model=model,
        prompts=harmful_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=constants.N_POSTERIOR_SAMPLES,
        temperature=args.temperature,
        device=constants.DEVICE
    )

    # Statistical analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    safe_mean = np.mean(safe_entropies)
    safe_std = np.std(safe_entropies)
    adv_mean = np.mean(adv_entropies)
    adv_std = np.std(adv_entropies)

    print(f"\nSafe Prompts:")
    print(f"  Mean entropy: {safe_mean:.4f} ± {safe_std:.4f}")
    print(f"  Min: {np.min(safe_entropies):.4f}, Max: {np.max(safe_entropies):.4f}")

    print(f"\nAdversarial Prompts:")
    print(f"  Mean entropy: {adv_mean:.4f} ± {adv_std:.4f}")
    print(f"  Min: {np.min(adv_entropies):.4f}, Max: {np.max(adv_entropies):.4f}")

    # T-test
    t_stat, p_value = stats.ttest_ind(adv_entropies, safe_entropies)
    print(f"\nStatistical Test (Independent t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")

    if p_value < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
        if adv_mean > safe_mean:
            print(f"  → Adversarial prompts have HIGHER entropy (supports hypothesis)")
        else:
            print(f"  → Adversarial prompts have LOWER entropy (unexpected)")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((safe_std**2 + adv_std**2) / 2)
    cohens_d = (adv_mean - safe_mean) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d: {cohens_d:.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'n_posterior_samples': constants.N_POSTERIOR_SAMPLES,
        'temperature': args.temperature,
        'safe_prompts': {
            'n_samples': len(safe_test),
            'mean_entropy': float(safe_mean),
            'std_entropy': float(safe_std),
            'min_entropy': float(np.min(safe_entropies)),
            'max_entropy': float(np.max(safe_entropies)),
            'entropies': [float(x) for x in safe_entropies]
        },
        'adversarial_prompts': {
            'n_samples': len(harmful_test),
            'mean_entropy': float(adv_mean),
            'std_entropy': float(adv_std),
            'min_entropy': float(np.min(adv_entropies)),
            'max_entropy': float(np.max(adv_entropies)),
            'entropies': [float(x) for x in adv_entropies]
        },
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }
    }

    # Save to JSON
    output_path = os.path.join(args.model_path, f"entropy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"Results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute predictive entropy for safe and adversarial prompts")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (must contain fisher_diag.pt)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature parameter for posterior sampling (default: 0.05)"
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    if not os.path.exists(fisher_path):
        raise ValueError(f"Fisher matrix not found at: {fisher_path}")

    main(args)
