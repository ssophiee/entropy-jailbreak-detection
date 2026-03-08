"""
Compute combined AUROC for trend features (monotonicity, spearman_rho, mean_acceleration)
from existing JSON result files — no GPU needed.

Usage:
    python -m approaches.intermediate_layer.compute_combined_score \
        --results_dirs results/intermediate_entropy/0703 results/intermediate_entropy \
        --output results/intermediate_entropy/combined_score_results.md

Combination: for each run, extract the three feature vectors, orient each so that
higher score = more harmful, min-max normalise, then average.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

# ── Helpers ──────────────────────────────────────────────────────────────────

def depth_layer(layers_probed: list, frac: float) -> int:
    """Return the probed layer index closest to `frac` fraction of network depth."""
    n = max(layers_probed) + 1
    target = frac * (n - 1)
    return min(layers_probed, key=lambda l: abs(l - target))


def extract(features: list, key: str) -> np.ndarray:
    return np.array([f.get(key, 0.0) for f in features], dtype=np.float64)


def orient(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Flip so that higher score = more likely harmful."""
    if roc_auc_score(y_true, scores) < 0.5:
        return -scores
    return scores


def normalise(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ── Main ─────────────────────────────────────────────────────────────────────

def process_file(path: str) -> dict | None:
    with open(path) as f:
        d = json.load(f)

    safe_feats = d.get("safe_features")
    harm_feats = d.get("harmful_features")
    if not safe_feats or not harm_feats:
        return None

    layers = d.get("layers_probed", [])
    if not layers:
        return None

    model = d.get("model_path", "?").split("/")[-1]
    safe_ds = d.get("safe_dataset", "?")
    harm_ds = d.get("harmful_dataset", "?")

    y_true = np.array(
        [0] * len(safe_feats) + [1] * len(harm_feats), dtype=int
    )

    # Pick layers at ~69% and ~25% depth
    l69 = depth_layer(layers, 0.69)
    l25 = depth_layer(layers, 0.25)

    features = {
        "monotonicity": f"L{l69}_monotonicity_up",
        "spearman_rho": f"L{l69}_spearman_rho",
        "mean_accel_69": f"L{l69}_mean_acceleration",
        "mean_accel_25": f"L{l25}_mean_acceleration",
    }

    all_feats = safe_feats + harm_feats
    scores = {}
    for name, key in features.items():
        s = extract(all_feats, key)
        scores[name] = orient(s, y_true)

    # Individual AUROCs
    individual = {name: auroc(y_true, s) for name, s in scores.items()}

    # Combinations
    def combo(*names):
        normed = [normalise(scores[n]) for n in names]
        return auroc(y_true, np.mean(normed, axis=0))

    combined = {
        "mono+spear":            combo("monotonicity", "spearman_rho"),
        "mono+accel25":          combo("monotonicity", "mean_accel_25"),
        "spear+accel25":         combo("spearman_rho", "mean_accel_25"),
        "mono+spear+accel25":    combo("monotonicity", "spearman_rho", "mean_accel_25"),
        "mono+spear+accel69":    combo("monotonicity", "spearman_rho", "mean_accel_69"),
    }

    return {
        "model": model,
        "safe_ds": safe_ds,
        "harm_ds": harm_ds,
        "l69": l69,
        "l25": l25,
        "individual": individual,
        "combined": combined,
    }


def fmt(v: float) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "  n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirs", nargs="+",
                        default=["results/intermediate_entropy/0703",
                                 "results/intermediate_entropy"])
    parser.add_argument("--output", default=None,
                        help="Optional path to save markdown output")
    args = parser.parse_args()

    # Collect all JSONs, deduplicate by (model, safe_ds, harm_ds) — prefer 0703
    seen: dict[tuple, dict] = {}
    for d in args.results_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = process_file(path)
            if r is None:
                continue
            key = (r["model"], r["safe_ds"], r["harm_ds"])
            if key not in seen:
                seen[key] = r

    if not seen:
        print("No valid JSON files found.")
        return

    # Group by model
    by_model: dict[str, list] = defaultdict(list)
    for r in seen.values():
        by_model[r["model"]].append(r)

    lines = []
    lines.append("# Combined Score Results\n")
    lines.append("Effective AUROC for individual trend features and their combinations.")
    lines.append("Combination = orient each feature (higher=harmful), min-max normalise, average.\n")

    combo_keys = ["mono+spear", "mono+accel25", "spear+accel25",
                  "mono+spear+accel25", "mono+spear+accel69"]
    indiv_keys = ["monotonicity", "spearman_rho", "mean_accel_25", "mean_accel_69"]

    for model, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda r: (r["safe_ds"], r["harm_ds"]))
        l69_ex = rows[0]["l69"]
        l25_ex = rows[0]["l25"]
        lines.append(f"\n## {model}  (depth layers: ~69%→L{l69_ex}, ~25%→L{l25_ex})\n")

        # Header
        header = (
            f"| {'safe × harmful':<35} "
            f"| {'mono':>6} | {'spear':>6} | {'accel25':>7} | {'accel69':>7} "
            f"‖ {'m+s':>6} | {'m+a25':>6} | {'s+a25':>6} "
            f"| {'m+s+a25':>8} | {'m+s+a69':>8} |"
        )
        sep = "|" + "-" * 37 + "|" + ("|" + "-" * 8) * 4 + "‖" + ("|" + "-" * 8) * 5 + "|"
        lines.append(header)
        lines.append(sep)

        for r in rows:
            label = f"{r['safe_ds']} × {r['harm_ds']}"
            iv = r["individual"]
            cv = r["combined"]
            lines.append(
                f"| {label:<35} "
                f"| {fmt(iv['monotonicity']):>6} | {fmt(iv['spearman_rho']):>6} "
                f"| {fmt(iv['mean_accel_25']):>7} | {fmt(iv['mean_accel_69']):>7} "
                f"‖ {fmt(cv['mono+spear']):>6} | {fmt(cv['mono+accel25']):>6} "
                f"| {fmt(cv['spear+accel25']):>6} "
                f"| {fmt(cv['mono+spear+accel25']):>8} | {fmt(cv['mono+spear+accel69']):>8} |"
            )

        # Means
        def mean_col(key, src):
            vals = [r[src][key] for r in rows if not np.isnan(r[src].get(key, float("nan")))]
            return np.mean(vals) if vals else float("nan")

        lines.append(sep)
        lines.append(
            f"| {'MEAN':<35} "
            f"| {fmt(mean_col('monotonicity','individual')):>6} "
            f"| {fmt(mean_col('spearman_rho','individual')):>6} "
            f"| {fmt(mean_col('mean_accel_25','individual')):>7} "
            f"| {fmt(mean_col('mean_accel_69','individual')):>7} "
            f"‖ {fmt(mean_col('mono+spear','combined')):>6} "
            f"| {fmt(mean_col('mono+accel25','combined')):>6} "
            f"| {fmt(mean_col('spear+accel25','combined')):>6} "
            f"| {fmt(mean_col('mono+spear+accel25','combined')):>8} "
            f"| {fmt(mean_col('mono+spear+accel69','combined')):>8} |"
        )

    output = "\n".join(lines)
    print(output)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
