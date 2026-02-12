# approaches/prompt_entropy/compute_prompt_entropy.py
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

from src.data_utils import load_training_and_test_data
import src.constants as constants

from approaches.prompt_entropy.prompt_entropy import compute_prompt_entropy_features


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
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


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def eval_detection(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    - AUROC/AP/TPR@FPR computed on raw scores (any real).
    - For ECE/Brier we min-max scale scores to [0,1] (monotonic pseudo-prob).
    - Accuracy uses median threshold (no tuning).
    """
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
        probs = np.full_like(scores, 0.5, dtype=np.float64)
    else:
        probs = (scores - s_min) / (s_max - s_min)
        probs = np.clip(probs, 0.0, 1.0)

    ece = compute_ece(y_true, probs, n_bins=10)
    brier = float(brier_score_loss(y_true, probs))

    return {
        "accuracy": acc,
        "auroc": auroc,
        "average_precision": ap,
        "tpr@1%fpr": tpr_1,
        "tpr@0.1%fpr": tpr_01,
        "ece": ece,
        "brier": brier,
        "prob_transform": f"minmax[{s_min:.4f},{s_max:.4f}]",
    }


def main(args: argparse.Namespace) -> None:
    device = constants.DEVICE
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load data
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY,
    )
    safe_test = data["safe_test"]
    harmful_test = data["harmful_test"]

    # 2) Load base model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if (device.type == "cuda" and args.fp16) else None,
    ).to(device)
    model.eval()

    # 3) Compute features
    safe_res = compute_prompt_entropy_features(
        model, tok, safe_test,
        device=device,
        max_length=args.max_length,
        trim_ratio=args.trim_ratio,
        first_frac=args.first_frac,
        last_frac=args.last_frac,
        high_q=args.high_q,
        return_entropies=args.save_traces,
    )
    harmful_res = compute_prompt_entropy_features(
        model, tok, harmful_test,
        device=device,
        max_length=args.max_length,
        trim_ratio=args.trim_ratio,
        first_frac=args.first_frac,
        last_frac=args.last_frac,
        high_q=args.high_q,
        return_entropies=args.save_traces,
    )

    # 4) Choose score feature
    def extract_scores(res_list) -> np.ndarray:
        return np.array([r.features[args.score_key] for r in res_list], dtype=np.float64)

    safe_scores = extract_scores(safe_res)
    harmful_scores = extract_scores(harmful_res)

    scores = np.concatenate([safe_scores, harmful_scores])
    y_true = np.concatenate([np.zeros(len(safe_scores), dtype=int), np.ones(len(harmful_scores), dtype=int)])

    metrics = eval_detection(y_true, scores)

    # Convenience: show opposite direction too
    metrics["auroc_flipped"] = float(roc_auc_score(y_true, -scores))

    print("\n=== Prompt-Reading Entropy Detection Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating)):
            print(f"{k:20s}: {float(v):.4f}")
        else:
            print(f"{k:20s}: {v}")
    print(f"score_key            : {args.score_key}")

    # 5) Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"prompt_entropy_{args.score_key}_{ts}.json")

    def pack(res_list):
        packed = []
        for r in res_list:
            packed.append({
                "prompt": r.prompt,
                "n_tokens": r.n_tokens,
                "features": r.features,
                "entropies": r.entropies if args.save_traces else None,
            })
        return packed

    payload = {
        "timestamp": ts,
        "base_model": args.model_name,
        "device": str(device),
        "score_key": args.score_key,
        "params": {
            "max_length": args.max_length,
            "trim_ratio": args.trim_ratio,
            "first_frac": args.first_frac,
            "last_frac": args.last_frac,
            "high_q": args.high_q,
        },
        "metrics": metrics,
        "safe": pack(safe_res),
        "harmful": pack(harmful_res),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=getattr(constants, "MODEL_NAME", "gpt2"))
    parser.add_argument("--output_dir", type=str, default="results/prompt_entropy")

    # Compute settings
    parser.add_argument("--max_length", type=int, default=512, help="truncate prompt length in tokens")
    parser.add_argument("--fp16", action="store_true")

    # Aggregation knobs
    parser.add_argument("--trim_ratio", type=float, default=0.10)
    parser.add_argument("--first_frac", type=float, default=0.25)
    parser.add_argument("--last_frac", type=float, default=0.25)
    parser.add_argument("--high_q", type=float, default=0.90)

    # Which aggregated feature to use as detection score
    parser.add_argument(
        "--score_key",
        type=str,
        default="slope",
        choices=[
            # level
            "mean", "median", "trimmed_mean", "p10", "p90",
            # extremes/dispersion
            "min", "max", "std", "range",
            # position-aware aggregates
            "first_mean", "last_mean",
            # trend / structure
            "slope", "delta_end", "delta_seg", "spearman_rho",
            # volatility / dynamics
            "total_variation", "monotonicity_up",
            # structure / spikes
            "frac_above_q", "peak_pos",
            # legacy
            "auc",
        ]
    )

    # Output controls
    parser.add_argument("--save_traces", action="store_true", help="store per-token entropy traces in JSON (large)")

    args = parser.parse_args()
    main(args)
