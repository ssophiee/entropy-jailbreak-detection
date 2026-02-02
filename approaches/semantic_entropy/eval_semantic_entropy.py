import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss
)

# -------------------------
# Utils
# -------------------------

def scores_to_probs_ecdf(scores, eps=1e-6):
    """
    Monotonic mapping scores -> (0,1) using empirical CDF (rank transform).
    Stable vs min-max and preserves ordering.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    # ranks in [0, n-1]
    ranks = scores.argsort().argsort().astype(float)
    probs = (ranks + 1.0) / (n + 1.0)  # avoid 0/1 exactly
    probs = np.clip(probs, eps, 1.0 - eps)
    return probs


def compute_accuracy(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return (preds == y_true).mean()


def compute_ece(y_true, scores, n_bins=10):
    """
    Expected Calibration Error (ECE)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins) - 1

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = scores[mask].mean()
        ece += (mask.sum() / len(scores)) * abs(acc - conf)
    return ece


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


# -------------------------
# Main evaluation
# -------------------------

def evaluate_from_json(json_path, score_key="semantic_entropy"):
    with open(json_path, "r") as f:
        data = json.load(f)

    safe_scores = np.array([x[score_key] for x in data["safe"]], dtype=float)
    harmful_scores = np.array([x[score_key] for x in data["harmful"]], dtype=float)

    scores = np.concatenate([safe_scores, harmful_scores])
    y_true = np.concatenate([
        np.zeros(len(safe_scores), dtype=int),
        np.ones(len(harmful_scores), dtype=int)
    ])

    # -------------------------
    # Threshold for accuracy (median score)
    # -------------------------
    threshold = np.median(scores)

    metrics = {}
    metrics["accuracy"] = compute_accuracy(y_true, scores, threshold)

    # -------------------------
    # AUROC / AP are ranking-based: raw scores OK
    # -------------------------
    metrics["auroc"] = roc_auc_score(y_true, scores)
    metrics["average_precision"] = average_precision_score(y_true, scores)
    metrics["tpr@1%fpr"] = tpr_at_fpr(y_true, scores, 0.01)
    metrics["tpr@0.1%fpr"] = tpr_at_fpr(y_true, scores, 0.001)

    # -------------------------
    # ECE + Brier require probabilities in [0,1]
    # Convert scores -> pseudo-prob via min-max scaling
    # (monotonic, so AUROC/AP unaffected; calibration is "relative")
    # -------------------------
    probs = scores_to_probs_ecdf(scores)
    metrics["ece"] = compute_ece(y_true, probs, n_bins=10)
    metrics["brier"] = brier_score_loss(y_true, probs)
    metrics["prob_transform"] = "ecdf_rank"


    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str)
    parser.add_argument(
        "--score_key",
        type=str,
        default="semantic_entropy",
        help="Which field to use as detection score"
    )
    args = parser.parse_args()

    metrics = evaluate_from_json(args.json_path, args.score_key)

    print("\n=== Semantic Entropy Detection Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating)):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")

