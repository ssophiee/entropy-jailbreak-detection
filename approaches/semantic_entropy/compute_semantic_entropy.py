#!/usr/bin/env python3
"""
Compute semantic entropy for safe vs harmful prompts using an *untrained/base* LLM.

This script:
1) Loads test data (safe + harmful prompts)
2) Loads a base LLM (no LoRA, no Fisher)
3) Samples multiple completions per prompt
4) Clusters completions by semantic similarity (embedding-based)
5) Computes semantic entropy and runs stats
6) Saves a JSON report

Usage:
  python scripts/compute_semantic_entropy.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --n_samples 25 --temperature 0.9 --top_p 0.95 \
    --cluster_threshold 0.82 \
    --output_dir results/semantic_entropy
"""

import os, sys

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../.."))
sys.path.insert(0, REPO_ROOT)

import json
import argparse
import numpy as np
from datetime import datetime
from scipy import constants, stats

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import *
from src.data_utils import load_training_and_test_data
from approaches.semantic_entropy.semantic_entropy import compute_semantic_entropy


def compute_statistics(safe_values, adv_values, metric_name: str):
    safe_values = np.array(safe_values, dtype=float)
    adv_values = np.array(adv_values, dtype=float)

    safe_mean, safe_std = float(np.mean(safe_values)), float(np.std(safe_values))
    adv_mean, adv_std = float(np.mean(adv_values)), float(np.std(adv_values))

    print(f"\n{metric_name.upper().replace('_', ' ')}:")
    print(f"  Safe Prompts:")
    print(f"    Mean: {safe_mean:.4f} ± {safe_std:.4f}")
    print(f"    Min: {float(np.min(safe_values)):.4f}, Max: {float(np.max(safe_values)):.4f}")

    print(f"  Harmful Prompts:")
    print(f"    Mean: {adv_mean:.4f} ± {adv_std:.4f}")
    print(f"    Min: {float(np.min(adv_values)):.4f}, Max: {float(np.max(adv_values)):.4f}")

    t_stat, p_value = stats.ttest_ind(adv_values, safe_values)
    print(f"  Statistical Test (Independent t-test):")
    print(f"    t-statistic: {float(t_stat):.4f}")
    print(f"    p-value: {float(p_value):.4e}")

    significant = bool(p_value < 0.05)
    if significant:
        print("    ✓ Significant difference (p < 0.05)")
        if adv_mean > safe_mean:
            print(f"    → Harmful prompts have HIGHER {metric_name} (supports hypothesis)")
        else:
            print(f"    → Harmful prompts have LOWER {metric_name}")
    else:
        print("    ✗ No significant difference (p ≥ 0.05)")

    pooled_std = np.sqrt((safe_std**2 + adv_std**2) / 2.0)
    cohens_d = (adv_mean - safe_mean) / pooled_std if pooled_std > 1e-12 else 0.0
    print(f"    Cohen's d: {float(cohens_d):.4f}")

    return {
        "safe_mean": safe_mean,
        "safe_std": safe_std,
        "harmful_mean": adv_mean,
        "harmful_std": adv_std,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": significant,
    }


def main(args):
    print("=" * 80)
    print("SEMANTIC ENTROPY COMPUTATION SCRIPT (UNTRAINED BASE LLM)")
    print("=" * 80)
    device = DEVICE
    print(f"Device: {device}")
    print(f"Base model: {args.model_name}")
    print(f"Embedding model: {args.embed_model_name}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Decoding: temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")
    print(f"Clustering: threshold={args.cluster_threshold}")
    print("=" * 80)

    # 1) Load data
    print("\n[1/4] Loading test data...")
    data = load_training_and_test_data(
        n_safe_train=N_SAFE_TRAIN,
        n_benign_train=N_BENIGN_TRAIN,
        n_test_per_category=N_TEST_PER_CATEGORY,
    )
    safe_test = data["safe_test"]
    harmful_test = data["harmful_test"]
    print(f"Loaded {len(safe_test)} safe test prompts")
    print(f"Loaded {len(harmful_test)} harmful test prompts")

    # 2) Load base model (no LoRA)
    print("\n[2/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Make sure we have a pad token to avoid generate() warnings/errors
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if (device.type == "cuda" and args.fp16) else None,
        device_map=None,
    ).to(device)
    model.eval()
    print("✓ Base model loaded")

    # 3) Compute semantic entropy
    print("\n[3/4] Computing semantic entropy on SAFE prompts...")
    safe_res = compute_semantic_entropy(
        model=model,
        tokenizer=tokenizer,
        prompts=safe_test,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        embed_model_name=args.embed_model_name,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        cluster_threshold=args.cluster_threshold,
        device=device,
        disable_tqdm=args.no_tqdm,
    )

    print("\n[3/4] Computing semantic entropy on HARMFUL prompts...")
    harmful_res = compute_semantic_entropy(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_test,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        embed_model_name=args.embed_model_name,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        cluster_threshold=args.cluster_threshold,
        device=device,
        disable_tqdm=args.no_tqdm,
    )

    # Extract metrics
    safe_H = [r.semantic_entropy for r in safe_res]
    harmful_H = [r.semantic_entropy for r in harmful_res]

    safe_Hk = [r.semantic_entropy_normalized_logK for r in safe_res]
    harmful_Hk = [r.semantic_entropy_normalized_logK for r in harmful_res]

    safe_Hc = [r.semantic_entropy_normalized_logC for r in safe_res]
    harmful_Hc = [r.semantic_entropy_normalized_logC for r in harmful_res]

    # 4) Stats + save
    print("\n[4/4] Statistical analysis...")
    stats_raw = compute_statistics(safe_H, harmful_H, "semantic_entropy")
    stats_logK = compute_statistics(safe_Hk, harmful_Hk, "semantic_entropy_normalized_logK")
    stats_logC = compute_statistics(safe_Hc, harmful_Hc, "semantic_entropy_normalized_logC")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"semantic_entropy_{ts}.json")

    def pack_results(res_list):
        packed = []
        for r in res_list:
            packed.append({
                "prompt": r.prompt,
                "completions": r.completions if args.save_completions else None,
                "cluster_ids": r.cluster_ids,
                "cluster_sizes": r.cluster_sizes,
                "n_clusters": r.n_clusters,
                "semantic_entropy": r.semantic_entropy,
                "semantic_entropy_normalized_logK": r.semantic_entropy_normalized_logK,
                "semantic_entropy_normalized_logC": r.semantic_entropy_normalized_logC,
            })
        return packed

    payload = {
        "timestamp": ts,
        "device": str(device),
        "base_model": args.model_name,
        "embed_model": args.embed_model_name,
        "params": {
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "cluster_threshold": args.cluster_threshold,
            "embed_batch_size": args.embed_batch_size,
            "embed_max_length": args.embed_max_length,
        },
        "statistics": {
            "semantic_entropy": stats_raw,
            "semantic_entropy_normalized_logK": stats_logK,
            "semantic_entropy_normalized_logC": stats_logC,
        },
        "safe": pack_results(safe_res),
        "harmful": pack_results(harmful_res),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="results/semantic_entropy")

    # Sampling / generation
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 weights when on CUDA")

    # Embedding + clustering
    parser.add_argument("--embed_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--embed_max_length", type=int, default=256)
    parser.add_argument("--cluster_threshold", type=float, default=0.82)

    # Output controls
    parser.add_argument("--save_completions", action="store_true", help="Store raw completions in JSON (can be large)")
    parser.add_argument("--no_tqdm", action="store_true")

    args = parser.parse_args()

    print(f"Found Device: {DEVICE}")

    main(args)
