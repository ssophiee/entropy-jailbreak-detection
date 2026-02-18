#!/usr/bin/env python3
"""
Multi-dataset evaluation runner.

Reads a YAML config and runs the existing compute_trace_metrics scripts
for every combination of harmful × safe datasets via subprocess.
After each pair, calls summary.sh to produce a per-pair CSV.

Usage:
    python scripts/run_multi_dataset_eval.py
    python scripts/run_multi_dataset_eval.py --config configs/my_config.yaml
    python scripts/run_multi_dataset_eval.py --dry_run
"""
import argparse
import glob
import os
import subprocess
import sys
from itertools import product

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

APPROACH_MODULES = {
    "laplace_approx": "approaches.laplace_approx.scripts.compute_trace_metrics",
    "ensemble_lora": "approaches.ensemble_lora.scripts.compute_ensemble_trace_metrics",
}

SUMMARY_SCRIPTS = {
    "laplace_approx": os.path.join(
        REPO_ROOT, "approaches", "laplace_approx", "scripts", "summary.sh"
    ),
    "ensemble_lora": os.path.join(
        REPO_ROOT, "approaches", "laplace_approx", "scripts", "summary.sh"
    ),
}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_n_samples(val):
    if isinstance(val, str) and val.strip().lower() == "max":
        return 999_999
    return int(val)


def build_command(approach, model_cfg, eval_cfg, safe_name, harmful_name, output_dir):
    """Build the subprocess command for one dataset pair."""
    module = APPROACH_MODULES[approach]
    n_samples = resolve_n_samples(eval_cfg.get("n_samples_per_dataset", 500))

    cmd = [
        sys.executable, "-m", module,
        "--run_all",
        "--safe_dataset", safe_name,
        "--harmful_dataset", harmful_name,
        "--n_test", str(n_samples),
        "--balance",
        "--balance_seed", str(eval_cfg.get("balance_seed", 42)),
        "--output_dir", output_dir,
    ]

    if approach == "laplace_approx":
        cmd.extend(["--model_path", model_cfg["path"]])
        cmd.extend(["--temperature", str(eval_cfg.get("temperature", 0.05))])
    elif approach == "ensemble_lora":
        cmd.extend(["--ensemble_dir", model_cfg["path"]])

    return cmd


def find_latest_json(directory):
    """Find the most recently created JSON file in a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.json")), reverse=True)
    return files[0] if files else None


def run_summary_sh(approach, json_path, csv_path):
    """Call summary.sh on a single --run_all JSON to produce a per-pair CSV."""
    script = SUMMARY_SCRIPTS[approach]
    if not os.path.exists(script):
        print(f"  WARNING: summary script not found at {script}, skipping CSV")
        return
    subprocess.run(["bash", script, json_path, csv_path], cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset jailbreak detection evaluation"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(REPO_ROOT, "configs", "eval_config.yaml"),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the evaluation plan and exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    output_cfg = cfg["output"]

    approach = eval_cfg.get("approach", "laplace_approx")
    base_out = output_cfg.get("base_dir", "results/multi_dataset")

    harmful_names = cfg["harmful_datasets"]
    safe_names = cfg["safe_datasets"]
    pairs = list(product(safe_names, harmful_names))

    # ── Print plan ───────────────────────────────────────────────────────
    print("=" * 70)
    print("MULTI-DATASET EVALUATION")
    print("=" * 70)
    print(f"Approach       : {approach}")
    print(f"Model          : {model_cfg['name']}")
    print(f"Adapter        : {model_cfg['path']} ({model_cfg.get('source', 'local')})")
    print(f"Samples/dataset: {eval_cfg.get('n_samples_per_dataset', 500)}")
    print(f"Balance        : 1:1 (seed={eval_cfg.get('balance_seed', 42)})")
    print(f"Output         : {base_out}")
    print(f"\nHarmful ({len(harmful_names)}): {harmful_names}")
    print(f"Safe    ({len(safe_names)}): {safe_names}")
    print(f"Total pairs: {len(pairs)}")
    print()
    for i, (safe, harmful) in enumerate(pairs, 1):
        print(f"  {i:2d}. {safe} vs {harmful}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Exiting.")
        return

    # ── Run each pair ────────────────────────────────────────────────────
    completed = 0

    for i, (safe_name, harmful_name) in enumerate(pairs, 1):
        label = f"{safe_name}_vs_{harmful_name}"
        pair_out = os.path.join(base_out, label)
        os.makedirs(pair_out, exist_ok=True)

        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(pairs)}] {label}")
        print(f"{'─' * 70}")

        cmd = build_command(approach, model_cfg, eval_cfg,
                            safe_name, harmful_name, pair_out)
        print(f"  CMD: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=REPO_ROOT)

        if result.returncode != 0:
            print(f"  WARNING: pair {label} exited with code {result.returncode}")
            continue

        # Generate per-pair CSV via summary.sh
        json_path = find_latest_json(pair_out)
        if json_path:
            csv_path = os.path.join(pair_out, f"{label}_summary.csv")
            run_summary_sh(approach, json_path, csv_path)

        completed += 1

    print(f"\nDone. {completed}/{len(pairs)} pairs completed.")
    print(f"Results in: {base_out}")


if __name__ == "__main__":
    main()
