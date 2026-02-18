#!/usr/bin/env python3
"""
Compute the 20 entropy-trace aggregation metrics for test prompts
using a trained LoRA ensemble.

This mirrors the prompt-entropy evaluation pipeline:
  1. Load trained LoRA ensemble
  2. For each prompt, compute per-position predictive entropy across adapters
  3. Aggregate each trace into 20 scalar features
  4. Evaluate detection (AUROC, AP, TPR@FPR, ECE, Brier) using each feature

Usage:
    python compute_ensemble_trace_metrics.py --ensemble_dir saved_models/ensemble_lora_123456
    python compute_ensemble_trace_metrics.py --ensemble_dir saved_models/my_ensemble --score_key slope
    python compute_ensemble_trace_metrics.py --ensemble_dir saved_models/my_ensemble --run_all
"""
import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.ensemble_lora.inference import load_ensemble_from_directory
from approaches.ensemble_lora.uncertainty import compute_entropy_trace_features


SCORE_KEYS = [
    "mean", "std", "min", "max", "median",
    "p10", "p90", "trimmed_mean",
    "first_mean", "last_mean",
    "auc", "slope",
    "frac_above_q", "range",
    "delta_end", "delta_seg", "spearman_rho",
    "total_variation", "monotonicity_up",
    "peak_pos",
]


# ── Detection helpers (same as prompt_entropy) ──────────────────────────

def compute_ece(y_true, probs, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            continue
        acc = float(y_true[m].mean())
        conf = float(probs[m].mean())
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def eval_detection(y_true, scores):
    scores = scores.astype(np.float64)

    thr = float(np.median(scores))
    preds = (scores >= thr).astype(int)
    acc = float((preds == y_true).mean())

    auroc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))
    tpr_1 = tpr_at_fpr(y_true, scores, 0.01)
    tpr_01 = tpr_at_fpr(y_true, scores, 0.001)

    s_min, s_max = float(scores.min()), float(scores.max())
    if abs(s_max - s_min) < 1e-12:
        prob = np.full_like(scores, 0.5, dtype=np.float64)
    else:
        prob = (scores - s_min) / (s_max - s_min)
        prob = np.clip(prob, 0.0, 1.0)

    ece = compute_ece(y_true, prob, n_bins=10)
    brier = float(brier_score_loss(y_true, prob))

    return {
        "accuracy": acc,
        "auroc": auroc,
        "average_precision": ap,
        "tpr@1%fpr": tpr_1,
        "tpr@0.1%fpr": tpr_01,
        "ece": ece,
        "brier": brier,
        "auroc_flipped": float(roc_auc_score(y_true, -scores)),
    }


# ── Main ────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 80)
    print("ENSEMBLE LORA — ENTROPY TRACE METRICS")
    print("=" * 80)
    print(f"Device          : {constants.DEVICE}")
    print(f"Base Model      : {constants.MODEL_NAME}")
    print(f"Ensemble Dir    : {args.ensemble_dir}")
    print(f"Max Length      : {constants.MAX_LENGTH}")
    print(f"Score key(s)    : {'ALL' if args.run_all else args.score_key}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load test data
    print("\n[1/4] Loading test data...")
    n_test = getattr(args, 'n_test', None) or constants.N_TEST_PER_CATEGORY
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=n_test,
        safe_dataset=getattr(args, 'safe_dataset', None),
        harmful_dataset=getattr(args, 'harmful_dataset', None),
        balance=getattr(args, 'balance', False),
        balance_seed=getattr(args, 'balance_seed', 42),
    )
    safe_test = data["safe_test"]
    harmful_test = data["harmful_test"]
    print(f"  Safe: {len(safe_test)}  |  Harmful: {len(harmful_test)}")

    # 2) Load ensemble
    print(f"\n[2/4] Loading ensemble from {args.ensemble_dir}...")
    ensemble = load_ensemble_from_directory(
        base_model_name=constants.MODEL_NAME,
        ensemble_dir=args.ensemble_dir,
        device=constants.DEVICE,
    )
    print(f"  Adapters: {len(ensemble.adapter_paths)}")

    # 3) Compute entropy trace features
    print("\n[3/4] Computing entropy trace features...")
    print("  Processing SAFE prompts...")
    safe_features = compute_entropy_trace_features(
        ensemble, safe_test, max_length=constants.MAX_LENGTH,
    )
    print("  Processing HARMFUL prompts...")
    harmful_features = compute_entropy_trace_features(
        ensemble, harmful_test, max_length=constants.MAX_LENGTH,
    )

    # 4) Evaluate detection
    print("\n[4/4] Evaluating detection metrics...")
    y_true = np.concatenate([
        np.zeros(len(safe_test), dtype=int),
        np.ones(len(harmful_test), dtype=int),
    ])

    keys_to_run = SCORE_KEYS if args.run_all else [args.score_key]
    all_results = {}

    for key in keys_to_run:
        safe_scores = np.array([f[key] for f in safe_features], dtype=np.float64)
        harmful_scores = np.array([f[key] for f in harmful_features], dtype=np.float64)
        scores = np.concatenate([safe_scores, harmful_scores])

        metrics = eval_detection(y_true, scores)
        all_results[key] = metrics

        print(f"\n  --- {key} ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.4f}")
            else:
                print(f"    {k:20s}: {v}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_keys" if args.run_all else args.score_key
    out_path = os.path.join(args.output_dir, f"ensemble_trace_{suffix}_{ts}.json")

    payload = {
        "timestamp": ts,
        "ensemble_dir": args.ensemble_dir,
        "base_model": constants.MODEL_NAME,
        "n_adapters": len(ensemble.adapter_paths),
        "n_safe_test": len(safe_test),
        "n_harmful_test": len(harmful_test),
        "detection_metrics": all_results,
        "safe_features": safe_features,
        "harmful_features": harmful_features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved: {out_path}")

    # Print summary table if --run_all
    if args.run_all:
        print("\n" + "=" * 80)
        print(f"{'key':20s} {'AUROC':>8s} {'AUROC_flip':>10s} {'AP':>8s} {'TPR@1%':>8s}")
        print("-" * 60)
        for key in SCORE_KEYS:
            m = all_results[key]
            print(f"{key:20s} {m['auroc']:8.4f} {m['auroc_flipped']:10.4f} "
                  f"{m['average_precision']:8.4f} {m['tpr@1%fpr']:8.4f}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble LoRA — entropy trace aggregation metrics for detection"
    )
    parser.add_argument(
        "--ensemble_dir", type=str, required=True,
        help="Directory containing trained ensemble adapters",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/ensemble_lora_trace",
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--score_key", type=str, default="slope", choices=SCORE_KEYS,
        help="Which aggregated feature to use as detection score (default: slope)",
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help="Evaluate ALL 20 score keys and print a summary table",
    )
    parser.add_argument(
        "--safe_dataset", type=str, default=None,
        help="Registry name for safe test prompts (e.g. 'xstest'). Default: ultrachat.",
    )
    parser.add_argument(
        "--harmful_dataset", type=str, default=None,
        help="Registry name for harmful test prompts (e.g. 'strongreject'). Default: advbench.",
    )
    parser.add_argument(
        "--n_test", type=int, default=None,
        help="Max prompts per test category (default: constants.N_TEST_PER_CATEGORY).",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Subsample to 1:1 class balance.",
    )
    parser.add_argument(
        "--balance_seed", type=int, default=42,
        help="Random seed for balanced subsampling (default: 42).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ensemble_dir):
        raise ValueError(f"Ensemble directory does not exist: {args.ensemble_dir}")

    main(args)
