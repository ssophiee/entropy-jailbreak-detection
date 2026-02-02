"""Uncertainty quantification for LoRA ensembles."""
from typing import List, Union, Dict, Optional

from src.uncertainty import compute_uncertainty_metrics


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
