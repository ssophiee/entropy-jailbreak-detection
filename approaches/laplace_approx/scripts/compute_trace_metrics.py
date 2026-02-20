#!/usr/bin/env python3
"""
Compute the 20 entropy-trace aggregation metrics for test prompts
using a Laplace-approximated LoRA model.

This mirrors the prompt-entropy evaluation pipeline:
  1. Load fine-tuned LoRA model + Fisher diagonal
  2. For each prompt, draw posterior samples and compute per-position
     predictive entropy across samples
  3. Aggregate each trace into 20 scalar features
  4. Evaluate detection (AUROC, AP, TPR@FPR, ECE, Brier) using each feature

Usage:
    python compute_trace_metrics.py --model_path saved_models/your_model
    python compute_trace_metrics.py --model_path saved_models/your_model --score_key slope
    python compute_trace_metrics.py --model_path saved_models/your_model --run_all
"""
import logging
import logging
import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.laplace_approx.uncertainty import compute_entropy_trace_features


log = logging.getLogger(__name__)


log = logging.getLogger(__name__)


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


def load_fisher_matrix(fisher_path):
    log.info("Loading Fisher matrix from: %s", fisher_path)
    log.info("Loading Fisher matrix from: %s", fisher_path)
    checkpoint = torch.load(fisher_path, map_location=constants.DEVICE)
    fisher_diag = checkpoint["fisher_diag"]
    log.info("  Loaded Fisher matrix with %d parameters", len(fisher_diag))
    log.info("  Loaded Fisher matrix with %d parameters", len(fisher_diag))
    return fisher_diag


# ── Main ────────────────────────────────────────────────────────────────

def main(args):
    # Resolve n_samples
    n_samples = args.n_samples if args.n_samples is not None else constants.N_POSTERIOR_SAMPLES

    log.info("=" * 80)
    log.info("LAPLACE APPROX — ENTROPY TRACE METRICS")
    log.info("=" * 80)
    log.info("Device          : %s", constants.DEVICE)
    log.info("Model path      : %s", args.model_path)
    log.info("N samples       : %d", n_samples)
    log.info("Temperature     : %s", args.temperature)
    log.info("Score key(s)    : %s", "ALL" if args.run_all else args.score_key)
    log.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load test data
    log.info("[1/5] Loading test data...")
    n_test_arg = getattr(args, 'n_test', None)
    if n_test_arg == "max":
        n_test = 999999  # Load all available prompts
        log.info("Loading all available prompts (n_test=max)")
    elif n_test_arg is not None:
        n_test = int(n_test_arg)
    else:
        n_test = constants.N_TEST_PER_CATEGORY
    log.info("[1/5] Loading test data...")
    n_test_arg = getattr(args, 'n_test', None)
    if n_test_arg == "max":
        n_test = 999999  # Load all available prompts
        log.info("Loading all available prompts (n_test=max)")
    elif n_test_arg is not None:
        n_test = int(n_test_arg)
    else:
        n_test = constants.N_TEST_PER_CATEGORY
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
    log.info("  Safe: %d  |  Harmful: %d", len(safe_test), len(harmful_test))
    log.info("  Safe: %d  |  Harmful: %d", len(safe_test), len(harmful_test))

    # 2) Load model
    log.info("[2/5] Loading fine-tuned LoRA model...")
    log.info("[2/5] Loading fine-tuned LoRA model...")
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        constants.MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=constants.DEVICE,
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.to(constants.DEVICE)

    # Enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.eval()
    model = model.float()

    lora_params = [n for n, p in model.named_parameters()
                   if "lora" in n.lower() and p.requires_grad]
    log.info("  Trainable LoRA parameters: %d", len(lora_params))
    log.info("  Trainable LoRA parameters: %d", len(lora_params))

    # 3) Load Fisher matrix
    log.info("[3/5] Loading Fisher information matrix...")
    log.info("[3/5] Loading Fisher information matrix...")
    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    fisher_diag = load_fisher_matrix(fisher_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4) Compute entropy trace features
    log.info("[4/5] Computing entropy trace features...")
    log.info("  Processing SAFE prompts...")
    log.info("[4/5] Computing entropy trace features...")
    log.info("  Processing SAFE prompts...")
    safe_features = compute_entropy_trace_features(
        model=model,
        prompts=safe_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=n_samples,
        temperature=args.temperature,
        device=constants.DEVICE,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("  Processing HARMFUL prompts...")
    log.info("  Processing HARMFUL prompts...")
    harmful_features = compute_entropy_trace_features(
        model=model,
        prompts=harmful_test,
        tokenizer=tokenizer,
        fisher_diag=fisher_diag,
        n_samples=n_samples,
        temperature=args.temperature,
        device=constants.DEVICE,
    )

    # 5) Evaluate detection
    log.info("[5/5] Evaluating detection metrics...")
    log.info("[5/5] Evaluating detection metrics...")
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

        log.info("  --- %s ---", key)
        log.info("  --- %s ---", key)
        for k, v in metrics.items():
            if isinstance(v, float):
                log.info("    %-20s: %.4f", k, v)
                log.info("    %-20s: %.4f", k, v)
            else:
                log.info("    %-20s: %s", k, v)
                log.info("    %-20s: %s", k, v)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_keys" if args.run_all else args.score_key
    out_path = os.path.join(args.output_dir, f"laplace_trace_{suffix}_{ts}.json")

    payload = {
        "timestamp": ts,
        "model_path": args.model_path,
        "base_model": constants.MODEL_NAME,
        "n_posterior_samples": n_samples,
        "temperature": args.temperature,
        "n_safe_test": len(safe_test),
        "n_harmful_test": len(harmful_test),
        "detection_metrics": all_results,
        "safe_features": safe_features,
        "harmful_features": harmful_features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Saved results to: %s", out_path)
    log.info("Saved results to: %s", out_path)

    # Print summary table if --run_all
    if args.run_all:
        log.info("=" * 80)
        log.info("%-20s %8s %10s %8s %8s", "key", "AUROC", "AUROC_flip", "AP", "TPR@1%")
        log.info("-" * 60)
        log.info("=" * 80)
        log.info("%-20s %8s %10s %8s %8s", "key", "AUROC", "AUROC_flip", "AP", "TPR@1%")
        log.info("-" * 60)
        for key in SCORE_KEYS:
            m = all_results[key]
            log.info("%-20s %8.4f %10.4f %8.4f %8.4f",
                     key, m["auroc"], m["auroc_flipped"], m["average_precision"], m["tpr@1%fpr"])
        log.info("=" * 80)
            log.info("%-20s %8.4f %10.4f %8.4f %8.4f",
                     key, m["auroc"], m["auroc_flipped"], m["average_precision"], m["tpr@1%fpr"])
        log.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laplace LoRA — entropy trace aggregation metrics for detection"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the fine-tuned model directory (must contain fisher_diag.pt)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/laplace_trace",
        help="Directory for output JSON",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of posterior samples (default: constants.N_POSTERIOR_SAMPLES=50)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05,
        help="Temperature for posterior sampling (default: 0.05)",
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
        "--n_test", type=str, default=None,
        help="Max prompts per test category. Use 'max' to load all available prompts (default: constants.N_TEST_PER_CATEGORY=50).",
        "--n_test", type=str, default=None,
        help="Max prompts per test category. Use 'max' to load all available prompts (default: constants.N_TEST_PER_CATEGORY=50).",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Subsample to 1:1 class balance.",
    )
    parser.add_argument(
        "--balance_seed", type=int, default=42,
        help="Random seed for balanced subsampling (default: 42).",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Optional path to write logs to a file in addition to stdout.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Optional path to write logs to a file in addition to stdout.",
    )

    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    fisher_path = os.path.join(args.model_path, "fisher_diag.pt")
    if not os.path.exists(fisher_path):
        raise ValueError(f"Fisher matrix not found at: {fisher_path}")

    main(args)