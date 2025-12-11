"""Uncertainty quantification for LoRA ensembles."""
import torch
import torch.nn.functional as F
from typing import List, Union, Dict, Optional
import numpy as np
from tqdm import tqdm


def predictive_entropy(probs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute predictive entropy from ensemble predictions.

    Predictive Entropy = H[E[p(y|x)]] where expectation is over ensemble

    Args:
        probs: Probability tensor [n_adapters, n_prompts, vocab_size]
               or [n_prompts, vocab_size] if already averaged
        dim: Dimension to average over (0 for adapters)

    Returns:
        entropy: [n_prompts] tensor of entropy values
    """
    # Average predictions across ensemble
    if probs.dim() == 3:
        mean_probs = probs.mean(dim=dim)  # [n_prompts, vocab_size]
    else:
        mean_probs = probs

    # Compute entropy: -sum(p * log(p))
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    return entropy


def mutual_information(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute mutual information (epistemic uncertainty).

    MI = H[E[p(y|x)]] - E[H[p(y|x)]]
       = Predictive Entropy - Expected Data Entropy

    High MI indicates model uncertainty (disagreement among adapters).

    Args:
        probs: [n_adapters, n_prompts, vocab_size]

    Returns:
        mi: [n_prompts] mutual information values
    """
    # Predictive entropy
    pred_entropy = predictive_entropy(probs, dim=0)

    # Expected data entropy (average entropy of each adapter)
    data_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [n_adapters, n_prompts]
    expected_data_entropy = data_entropy.mean(dim=0)  # [n_prompts]

    mi = pred_entropy - expected_data_entropy
    return mi


def variance_of_predictions(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute variance across ensemble predictions.

    Args:
        probs: [n_adapters, n_prompts, vocab_size]

    Returns:
        variance: [n_prompts] variance of predicted class probabilities
    """
    # Get predicted class for each adapter
    predicted_probs, _ = probs.max(dim=-1)  # [n_adapters, n_prompts]

    # Variance across adapters
    variance = predicted_probs.var(dim=0)  # [n_prompts]
    return variance


def ensemble_diversity(probs: torch.Tensor) -> torch.Tensor:
    """
    Measure ensemble diversity (disagreement rate).

    Computes fraction of adapters that disagree on predicted class.

    Args:
        probs: [n_adapters, n_prompts, vocab_size]

    Returns:
        diversity: [n_prompts] fraction in [0, 1]
    """
    # Get predicted class for each adapter
    predicted_classes = probs.argmax(dim=-1)  # [n_adapters, n_prompts]

    # Count unique predictions per prompt
    diversity_scores = []
    for prompt_idx in range(predicted_classes.shape[1]):
        predictions = predicted_classes[:, prompt_idx]
        unique_preds = torch.unique(predictions)
        diversity = 1.0 - (1.0 / len(unique_preds))  # 0 if all agree, high if diverse
        diversity_scores.append(diversity)

    return torch.tensor(diversity_scores)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted confidences match actual accuracy.

    Args:
        probs: [n_prompts, vocab_size] predicted probabilities
        labels: [n_prompts] true class labels
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected Calibration Error
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()

    # Bin by confidence
    ece = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if bin_mask.sum() > 0:
            bin_accuracy = accuracies[bin_mask].mean()
            bin_confidence = confidences[bin_mask].mean()
            bin_size = bin_mask.sum().float()
            ece += (bin_size / len(probs)) * torch.abs(bin_accuracy - bin_confidence)

    return ece.item()


def compute_uncertainty_metrics(
    ensemble_probs: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive uncertainty metrics for ensemble.

    Args:
        ensemble_probs: [n_adapters, n_prompts, vocab_size]
        labels: Optional true labels for calibration metrics

    Returns:
        metrics: Dictionary with various uncertainty measures
    """
    metrics = {
        "predictive_entropy": predictive_entropy(ensemble_probs),
        "mutual_information": mutual_information(ensemble_probs),
        "variance": variance_of_predictions(ensemble_probs),
        "diversity": ensemble_diversity(ensemble_probs),
    }

    # Mean ensemble prediction
    mean_probs = ensemble_probs.mean(dim=0)
    metrics["mean_confidence"] = mean_probs.max(dim=-1)[0]

    # Calibration if labels provided
    if labels is not None:
        metrics["ece"] = expected_calibration_error(mean_probs, labels)

    return metrics


def compute_entropy_for_prompts(
    ensemble_inference,
    prompts: List[str],
    max_length: int = 256,
    return_all_metrics: bool = False
) -> Union[List[float], Dict[str, List[float]]]:
    """
    Compute uncertainty metrics for a list of prompts using ensemble.

    Args:
        ensemble_inference: EnsembleLoRAInference instance
        prompts: List of prompts
        max_length: Max sequence length
        return_all_metrics: If True, return all uncertainty metrics

    Returns:
        If return_all_metrics=False: List of predictive entropy values
        If return_all_metrics=True: Dict with all uncertainty metrics
    """
    # Get ensemble predictions
    _, all_probs = ensemble_inference.ensemble_predict(
        prompts, max_length=max_length
    )

    # Compute metrics
    metrics = compute_uncertainty_metrics(all_probs)

    if return_all_metrics:
        return {k: v.tolist() for k, v in metrics.items() if isinstance(v, torch.Tensor)}
    else:
        return metrics["predictive_entropy"].tolist()


def rank_by_uncertainty(
    prompts: List[str],
    uncertainties: List[float],
    descending: bool = True
) -> List[tuple]:
    """
    Rank prompts by uncertainty scores.

    Args:
        prompts: List of prompts
        uncertainties: Corresponding uncertainty values
        descending: If True, rank highest uncertainty first

    Returns:
        ranked: List of (prompt, uncertainty, rank) tuples
    """
    sorted_indices = np.argsort(uncertainties)
    if descending:
        sorted_indices = sorted_indices[::-1]

    ranked = [
        (prompts[i], uncertainties[i], rank)
        for rank, i in enumerate(sorted_indices)
    ]
    return ranked


def detect_ood_by_entropy(
    in_distribution_prompts: List[str],
    test_prompts: List[str],
    ensemble_inference,
    threshold_percentile: float = 90.0,
    max_length: int = 256
) -> Dict:
    """
    Detect out-of-distribution samples using predictive entropy.

    Args:
        in_distribution_prompts: Known in-distribution prompts
        test_prompts: Test prompts to evaluate
        ensemble_inference: EnsembleLoRAInference instance
        threshold_percentile: Percentile of ID entropy to use as threshold
        max_length: Max sequence length

    Returns:
        results: Dict with OOD detection results
    """
    # Compute entropy for ID data
    print("Computing entropy for in-distribution data...")
    id_entropies = compute_entropy_for_prompts(
        ensemble_inference, in_distribution_prompts, max_length
    )

    # Set threshold
    threshold = np.percentile(id_entropies, threshold_percentile)

    # Compute entropy for test data
    print("Computing entropy for test data...")
    test_entropies = compute_entropy_for_prompts(
        ensemble_inference, test_prompts, max_length
    )

    # Flag OOD
    ood_flags = [e > threshold for e in test_entropies]

    results = {
        "id_entropies": id_entropies,
        "test_entropies": test_entropies,
        "threshold": threshold,
        "ood_flags": ood_flags,
        "n_ood": sum(ood_flags),
        "ood_rate": sum(ood_flags) / len(ood_flags) if ood_flags else 0.0
    }

    print(f"\nOOD Detection Results:")
    print(f"  Threshold (p{threshold_percentile}): {threshold:.4f}")
    print(f"  Detected OOD: {results['n_ood']}/{len(test_prompts)} ({results['ood_rate']:.1%})")

    return results


# Alias for compatibility
compute_predictive_entropy = compute_entropy_for_prompts
