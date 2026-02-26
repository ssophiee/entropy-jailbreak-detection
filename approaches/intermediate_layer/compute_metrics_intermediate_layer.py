#!/usr/bin/env python3
"""
Compute intermediate-layer entropy metrics for test prompts using a
fine-tuned LoRA model (no Laplace sampling — single forward pass per prompt).

For each prompt, hidden states are extracted at evenly-spaced transformer
layers, projected through the final norm + lm_head to obtain per-layer
entropy traces, then aggregated into scalar features using the same
aggregate_entropy_features() used by the trace metrics pipeline.

Features per prompt:
  - L{k}_{metric}          : all 30 metrics on layer k's position trace
  - depth_{metric}         : all 30 metrics on the layer-mean depth profile
  - depth_spread_{metric}  : all 30 metrics on the layer-std depth profile

Usage:
    python compute_intermediate_entropy_metrics.py --model_path saved_models/your_model
    python compute_intermediate_entropy_metrics.py --model_path saved_models/your_model --run_all
    python compute_intermediate_entropy_metrics.py --model_path saved_models/your_model --score_key depth_slope
"""
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

this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data
from approaches.prompt_entropy.prompt_entropy import (
    aggregate_entropy_features,
    get_model_components,
    get_layers_to_probe,
    compute_intermediate_layer_entropies,
    aggregate_layer_entropy_surface,
)

log = logging.getLogger(__name__)

N_PROBE_DEFAULT = 8  # number of layers to probe


def _build_score_keys(n_probe: int, model) -> list:
    """
    Derive the full list of score keys dynamically from the model,
    since layer indices depend on architecture.
    """
    layers_to_probe, _ = get_layers_to_probe(model, n_probe=n_probe)

    base_metrics = list(aggregate_entropy_features([0.0] * 10).keys())

    keys = []

    # Per-layer position traces
    for layer_idx in layers_to_probe:
        for m in base_metrics:
            keys.append(f"L{layer_idx}_{m}")

    # Depth profile
    for m in base_metrics:
        keys.append(f"depth_{m}")

    # Depth spread profile
    for m in base_metrics:
        keys.append(f"depth_spread_{m}")

    return keys


# ── Detection helpers (identical to compute_trace_metrics.py) ────────────

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


# ── Feature computation ──────────────────────────────────────────────────

def compute_features_for_prompts(
    model,
    tokenizer,
    prompts,
    final_norm,
    lm_head,
    layers_to_probe,
    device,
    max_length,
):
    from tqdm import tqdm
    all_features = []

    for p in tqdm(prompts, desc="Intermediate entropy"):
        H, T = compute_intermediate_layer_entropies(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            final_norm=final_norm,
            lm_head=lm_head,
            layers_to_probe=layers_to_probe,
            device=device,
            max_length=max_length,
        )

        if T < 2 or H.shape[1] == 0:
            # Empty feature dict — will be filled with zeros below
            feats = aggregate_layer_entropy_surface(
                np.zeros((len(layers_to_probe), 2)), layers_to_probe
            )
        else:
            feats = aggregate_layer_entropy_surface(H, layers_to_probe)

        all_features.append(feats)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_features


# ── Main ─────────────────────────────────────────────────────────────────

def main(args):
    log.info("=" * 80)
    log.info("INTERMEDIATE LAYER ENTROPY METRICS")
    log.info("=" * 80)
    log.info("Device          : %s", constants.DEVICE)
    log.info("Model path      : %s", args.model_path)
    log.info("N probe layers  : %d", args.n_probe)
    log.info("Score key(s)    : %s", "ALL" if args.run_all else args.score_key)
    log.info("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load test data
    log.info("[1/4] Loading test data...")
    n_test = 999999 if getattr(args, "n_test", None) == "max" \
        else int(args.n_test) if args.n_test is not None \
        else constants.N_TEST_PER_CATEGORY

    if n_test == 999999:
        log.info("Loading all available prompts (n_test=max)")

    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=n_test,
        safe_dataset=getattr(args, "safe_dataset", None),
        harmful_dataset=getattr(args, "harmful_dataset", None),
        balance=getattr(args, "balance", False),
        balance_seed=getattr(args, "balance_seed", 42),
    )
    safe_test = data["safe_test"]
    harmful_test = data["harmful_test"]
    log.info("  Safe: %d  |  Harmful: %d", len(safe_test), len(harmful_test))

    # 2) Load model (no Fisher needed — single forward pass)
    log.info("[2/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        constants.MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=constants.DEVICE,
    )

    # If model_path is a HuggingFace model id or equals the base model, skip LoRA
    is_base_model = (args.model_path == constants.MODEL_NAME or 
                    not os.path.exists(args.model_path))

    if is_base_model:
        log.info("  Running base model (no LoRA adapter)")
        model = base_model
    else:
        log.info("  Loading LoRA adapter from: %s", args.model_path)
        model = PeftModel.from_pretrained(base_model, args.model_path)

    model.to(constants.DEVICE)
    model.eval()
    model = model.float()

    # Derive model components universally
    final_norm, lm_head = get_model_components(model)
    layers_to_probe, n_layers = get_layers_to_probe(model, n_probe=args.n_probe)
    log.info("  n_layers        : %d", n_layers)
    log.info("  layers_to_probe : %s", layers_to_probe)

    # Build score keys now that we know the layer indices
    score_keys = _build_score_keys(args.n_probe, model)
    log.info("  Total features  : %d", len(score_keys))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3) Compute features
    log.info("[3/4] Computing intermediate entropy features...")
    log.info("  Processing SAFE prompts...")
    safe_features = compute_features_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=safe_test,
        final_norm=final_norm,
        lm_head=lm_head,
        layers_to_probe=layers_to_probe,
        device=constants.DEVICE,
        max_length=getattr(constants, "MAX_LENGTH", 512),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("  Processing HARMFUL prompts...")
    harmful_features = compute_features_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_test,
        final_norm=final_norm,
        lm_head=lm_head,
        layers_to_probe=layers_to_probe,
        device=constants.DEVICE,
        max_length=getattr(constants, "MAX_LENGTH", 512),
    )

    # 4) Evaluate
    log.info("[4/4] Evaluating detection metrics...")
    y_true = np.concatenate([
        np.zeros(len(safe_test), dtype=int),
        np.ones(len(harmful_test), dtype=int),
    ])

    keys_to_run = score_keys if args.run_all else [args.score_key]
    all_results = {}

    for key in keys_to_run:
        safe_scores = np.array(
            [f.get(key, 0.0) for f in safe_features], dtype=np.float64
        )
        harmful_scores = np.array(
            [f.get(key, 0.0) for f in harmful_features], dtype=np.float64
        )
        scores = np.concatenate([safe_scores, harmful_scores])

        try:
            metrics = eval_detection(y_true, scores)
        except Exception as e:
            log.warning("  Skipping key %s: %s", key, e)
            continue

        all_results[key] = metrics

        if not args.run_all:
            log.info("  --- %s ---", key)
            for k, v in metrics.items():
                log.info("    %-20s: %.4f", k, v)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_keys" if args.run_all else args.score_key
    out_path = os.path.join(
        args.output_dir, f"intermediate_entropy_{suffix}_{ts}.json"
    )

    payload = {
        "timestamp": ts,
        "model_path": args.model_path,
        "base_model": constants.MODEL_NAME,
        "n_probe": args.n_probe,
        "layers_probed": layers_to_probe,
        "n_layers_total": n_layers,
        "safe_dataset": getattr(args, "safe_dataset", None),
        "harmful_dataset": getattr(args, "harmful_dataset", None),
        "n_safe_test": len(safe_test),
        "n_harmful_test": len(harmful_test),
        "detection_metrics": all_results,
        "safe_features": safe_features,
        "harmful_features": harmful_features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("Saved results to: %s", out_path)

    # Summary table
    if args.run_all and all_results:
        log.info("=" * 80)
        log.info("%-40s %8s %10s %8s %8s",
                 "key", "AUROC", "AUROC_flip", "AP", "TPR@1%")
        log.info("-" * 80)
        # Sort by AUROC descending so best features float to top
        sorted_keys = sorted(
            all_results.keys(),
            key=lambda k: max(
                all_results[k]["auroc"],
                all_results[k]["auroc_flipped"]
            ),
            reverse=True,
        )
        for key in sorted_keys:
            m = all_results[key]
            log.info("%-40s %8.4f %10.4f %8.4f %8.4f",
                     key, m["auroc"], m["auroc_flipped"],
                     m["average_precision"], m["tpr@1%fpr"])
        log.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intermediate layer entropy metrics for harmful prompt detection"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="results/intermediate_entropy")
    parser.add_argument("--n_probe", type=int, default=N_PROBE_DEFAULT,
                        help="Number of layers to probe (default: 8)")
    parser.add_argument("--score_key", type=str, default="depth_slope",
                        help="Feature to use as detection score when not --run_all")
    parser.add_argument("--run_all", action="store_true",
                        help="Evaluate all features and print sorted summary table")
    parser.add_argument("--safe_dataset", type=str, default=None)
    parser.add_argument("--harmful_dataset", type=str, default=None)
    parser.add_argument("--n_test", type=str, default=None,
                        help="Max prompts per category. Use 'max' for all.")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--balance_seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_file", type=str, default=None)

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

    # if not os.path.exists(args.model_path):
    #     raise ValueError(f"Model path does not exist: {args.model_path}")

    main(args)