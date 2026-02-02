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
    python compute_entropy.py --model_path saved_models/your_model_name --visualize
    python compute_entropy.py --model_path saved_models/your_model_name --metric mutual_information --visualize
"""


import os, sys
this_dir = os.path.dirname(__file__)           # scripts/
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)


import os
import json
import torch
import argparse
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.laplace_approx.uncertainty import compute_predictive_entropy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_fisher_matrix(fisher_path):
    """Load Fisher diagonal matrix from saved checkpoint"""
    print(f"Loading Fisher matrix from: {fisher_path}")
    checkpoint = torch.load(fisher_path, map_location=constants.DEVICE)
    fisher_diag = checkpoint['fisher_diag']
    print(f"✓ Loaded Fisher matrix with {len(fisher_diag)} parameters")
    return fisher_diag


def plot_metric_distributions(safe_values, adv_values, metric_name, output_dir):
    """Plot distribution comparison for a metric"""
    safe_values = np.array(safe_values)
    adv_values = np.array(adv_values)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Histogram with KDE
    ax1 = axes[0]
    ax1.hist(safe_values, bins=30, alpha=0.6, label='Safe', color='blue', density=True, edgecolor='black')
    ax1.hist(adv_values, bins=30, alpha=0.6, label='Adversarial', color='red', density=True, edgecolor='black')

    # Add KDE curves
    if len(safe_values) > 1:
        kde_safe = gaussian_kde(safe_values)
        x_safe = np.linspace(safe_values.min(), safe_values.max(), 100)
        ax1.plot(x_safe, kde_safe(x_safe), color='blue', linewidth=2, linestyle='--')

    if len(adv_values) > 1:
        kde_adv = gaussian_kde(adv_values)
        x_adv = np.linspace(adv_values.min(), adv_values.max(), 100)
        ax1.plot(x_adv, kde_adv(x_adv), color='red', linewidth=2, linestyle='--')

    ax1.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Distribution: {metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = axes[1]
    data_to_plot = [safe_values, adv_values]
    box = ax2.boxplot(data_to_plot, labels=['Safe', 'Adversarial'], patch_artist=True,
                      widths=0.6, showmeans=True, meanline=True)

    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax2.set_title(f'Box Plot: {metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'{metric_name}_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    → Saved plot: {output_path}")


def compute_statistics(safe_values, adv_values, metric_name):
    """Compute and print statistical analysis for a given metric"""
    safe_values = np.array(safe_values)
    adv_values = np.array(adv_values)

    safe_mean = np.mean(safe_values)
    safe_std = np.std(safe_values)
    adv_mean = np.mean(adv_values)
    adv_std = np.std(adv_values)

    print(f"\n{metric_name.upper().replace('_', ' ')}:")
    print(f"  Safe Prompts:")
    print(f"    Mean: {safe_mean:.4f} ± {safe_std:.4f}")
    print(f"    Min: {np.min(safe_values):.4f}, Max: {np.max(safe_values):.4f}")

    print(f"  Adversarial Prompts:")
    print(f"    Mean: {adv_mean:.4f} ± {adv_std:.4f}")
    print(f"    Min: {np.min(adv_values):.4f}, Max: {np.max(adv_values):.4f}")

    # T-test
    t_stat, p_value = stats.ttest_ind(adv_values, safe_values)
    print(f"  Statistical Test (Independent t-test):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4e}")

    if p_value < 0.05:
        print(f"    ✓ Significant difference (p < 0.05)")
        if adv_mean > safe_mean:
            print(f"    → Adversarial prompts have HIGHER {metric_name} (supports hypothesis)")
        else:
            print(f"    → Adversarial prompts have LOWER {metric_name}")
    else:
        print(f"    ✗ No significant difference (p ≥ 0.05)")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((safe_std**2 + adv_std**2) / 2)
    cohens_d = (adv_mean - safe_mean) / pooled_std if pooled_std > 0 else 0
    print(f"    Cohen's d: {cohens_d:.4f}")

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }


def main(args):
    print("="*80)
    print("PREDICTIVE ENTROPY COMPUTATION SCRIPT")
    print("="*80)
    print(f"Device: {constants.DEVICE}")
    print(f"Model path: {args.model_path}")
    print(f"N posterior samples: {constants.N_POSTERIOR_SAMPLES}")
    print(f"Temperature: {args.temperature}")
    print(f"Uncertainty metric: {args.metric}")
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
    # Just load the base model WITHOUT LoRA initially
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        constants.MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=constants.DEVICE
    )

    # Load the fine-tuned adapter weights (this adds LoRA on top of base model)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.to(constants.DEVICE)

    # CRITICAL: Enable gradients for LoRA parameters (they're frozen by default)
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    model.eval()

    # Convert to float32 for stability during entropy computation
    model = model.float()

    # Verify LoRA params are trainable
    lora_params = [n for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    print(f"✓ Model loaded with {len(lora_params)} trainable LoRA parameters")

    # 3. Load Fisher matrix
    print("\n[3/5] Loading Fisher information matrix...")
    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    fisher_diag = load_fisher_matrix(fisher_path)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Compute entropy for safe prompts
    print(f"\n[4/5] Computing predictive entropy for SAFE prompts (metric: {args.metric})...")
    safe_entropies = compute_predictive_entropy(
        model=model,
        prompts=safe_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=constants.N_POSTERIOR_SAMPLES,
        temperature=args.temperature,
        device=constants.DEVICE,
        metric=args.metric,
        debug=args.debug
    )

    # Clear cache between computations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Compute entropy for adversarial prompts
    print(f"\n[5/5] Computing predictive entropy for ADVERSARIAL prompts (metric: {args.metric})...")
    adv_entropies = compute_predictive_entropy(
        model=model,
        prompts=harmful_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=constants.N_POSTERIOR_SAMPLES,
        temperature=args.temperature,
        device=constants.DEVICE,
        metric=args.metric,
        debug=args.debug
    )

    # Statistical analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Use compute_statistics for cleaner analysis
    stats_results = compute_statistics(safe_entropies, adv_entropies, args.metric)

    # Generate visualizations if requested
    if args.visualize:
        plot_metric_distributions(safe_entropies, adv_entropies, args.metric, args.model_path)

    # Extract values for results dict
    safe_mean = np.mean(safe_entropies)
    safe_std = np.std(safe_entropies)
    adv_mean = np.mean(adv_entropies)
    adv_std = np.std(adv_entropies)




    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'n_posterior_samples': constants.N_POSTERIOR_SAMPLES,
        'temperature': args.temperature,
        'metric': args.metric,
        'safe_prompts': {
            'n_samples': len(safe_test),
            'mean_entropy': float(safe_mean),
            'std_entropy': float(safe_std),
            'min_entropy': float(np.min(safe_entropies)),
            'max_entropy': float(np.max(safe_entropies)),
            'entropies': [float(x) for x in safe_entropies],
        },
        'adversarial_prompts': {
            'n_samples': len(harmful_test),
            'mean_entropy': float(adv_mean),
            'std_entropy': float(adv_std),
            'min_entropy': float(np.min(adv_entropies)),
            'max_entropy': float(np.max(adv_entropies)),
            'entropies': [float(x) for x in adv_entropies],
        },
        'statistical_test': stats_results
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
    parser.add_argument(
        "--metric",
        type=str,
        default="intersection_probs_entropy",
        choices=["mutual_information", "variance", "mean_entropy", "intersection_probs_entropy", "predictive_entropy"],
        help="Uncertainty metric to compute (default: intersection_probs_entropy)"
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        choices=[True, False],
        help="Debug Noise During Inference"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate distribution plots for the uncertainty metric"
    )
    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    if not os.path.exists(fisher_path):
        raise ValueError(f"Fisher matrix not found at: {fisher_path}")

    main(args)
