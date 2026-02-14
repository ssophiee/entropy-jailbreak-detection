"""Uncertainty quantification for LoRA ensembles."""
from typing import List, Union, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.uncertainty import compute_uncertainty_metrics
from approaches.prompt_entropy.prompt_entropy import aggregate_entropy_features


def compute_predictive_entropy(
    ensemble_inference,
    prompts: List[str],
    max_length: int = 256,
    metric: Optional[str] = "intersection_probs_entropy",
) -> Union[List[float], List[Dict[str, float]]]:
    """
    Compute uncertainty metrics for a list of prompts using ensemble.

    Args:
        ensemble_inference: EnsembleLoRAInference instance
        prompts: List of prompts
        max_length: Max sequence length
        metric: Specific metric to return, or None for all metrics
            Options: "predictive_entropy", "mutual_information", "variance",
                     "mean_confidence", "intersection_probs_entropy", "mean_entropy"

    Returns:
        If metric is specified: List of values for that metric
        If metric is None: List of dicts with all metrics
    """
    # Get ensemble predictions for all prompts
    _, all_probs = ensemble_inference.ensemble_predict(
        prompts, max_length=max_length
    )
    # all_probs: [n_adapters, n_prompts, vocab_size]

    # Process each prompt individually
    all_metrics = []
    for i in range(len(prompts)):
        prompt_probs = all_probs[:, i:i+1, :]  # [n_adapters, 1, vocab_size]
        metrics = compute_uncertainty_metrics(prompt_probs)
        all_metrics.append(metrics)

    if metric is None:
        return all_metrics
    else:
        return [m[metric] for m in all_metrics]


def compute_entropy_trace_features(
    ensemble_inference,
    prompts: List[str],
    max_length: int = 256,
) -> List[Dict[str, float]]:
    """
    Compute per-position predictive-entropy trace across the ensemble,
    then aggregate each trace into the same 20 scalar features used
    by the prompt-entropy approach.

    For each token position t, predictive entropy is:
        H[E_adapters[p(x_{t+1} | x_{<=t})]]

    This gives a per-position uncertainty trace that is then summarised
    with: mean, std, min, max, median, p10, p90, trimmed_mean,
    first_mean, last_mean, auc, slope, frac_above_q, range,
    delta_end, delta_seg, spearman_rho, total_variation,
    monotonicity_up, peak_pos.

    Args:
        ensemble_inference: EnsembleLoRAInference instance
        prompts: List of prompts
        max_length: Max sequence length

    Returns:
        List of dicts (one per prompt), each with the 20 aggregated features.
    """
    all_features = []

    for prompt in tqdm(prompts, desc="Entropy trace features"):
        # all_logits: [n_adapters, T, vocab_size]
        all_logits = ensemble_inference.ensemble_predict_all_positions(
            prompt, max_length=max_length
        )

        T = all_logits.shape[1]
        if T < 2:
            all_features.append(aggregate_entropy_features([]))
            continue

        # Compute predictive entropy at each position t (predicting t+1)
        # Use logits[:, :-1, :] — position t predicts token t+1
        logits_pred = all_logits[:, :-1, :]           # [n_adapters, T-1, V]
        probs = F.softmax(logits_pred, dim=-1)        # [n_adapters, T-1, V]

        # Mean probability across adapters at each position
        mean_probs = probs.mean(dim=0)                # [T-1, V]

        # Predictive entropy per position: H = -sum(p * log p)
        H = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)  # [T-1]
        entropies = H.tolist()

        features = aggregate_entropy_features(entropies)
        all_features.append(features)

    return all_features
