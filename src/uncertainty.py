"""Shared uncertainty quantification utilities."""
import torch
import numpy as np


# from https://github.com/WangKaizheng/CreINNs/blob/main/CreINNs_main_implementation/CreINNTestMulti.py
def compute_intersection_probability(upper_probs, lower_probs):
    """
    Compute intersection probability for credal sets.

    For a single sample (1D arrays):
    - alpha determines where intersection sits between lower and upper bounds
    - alpha ≈ 0 means intersection is near lower bound (high certainty)
    - alpha ≈ 1 means intersection is near upper bound (high uncertainty)
    """
    if upper_probs.ndim == 1:
        # Single sample case
        alpha_num = 1.0 - np.sum(lower_probs)
        alpha_denom = np.sum(upper_probs - lower_probs)

        alpha = alpha_num / alpha_denom
        intersection_probs = (upper_probs - lower_probs) * alpha + lower_probs
    else:
        # Batched case (original CreINNs code)
        alpha_num = 1.0 - np.sum(lower_probs, axis=-1, keepdims=True)
        alpha_denom = np.sum(upper_probs - lower_probs, axis=-1, keepdims=True)
        alpha = alpha_num / alpha_denom
        intersection_probs = (upper_probs - lower_probs) * alpha + lower_probs

    return intersection_probs


def predictive_entropy(probs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute predictive entropy from ensemble/sample predictions.

    Predictive Entropy = H[E[p(y|x)]] where expectation is over samples

    Args:
        probs: Probability tensor [n_samples, n_prompts, vocab_size]
               or [n_prompts, vocab_size] if already averaged
        dim: Dimension to average over (0 for samples)

    Returns:
        entropy: [n_prompts] tensor of entropy values
    """
    # Average predictions across samples
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

    High MI indicates model uncertainty (disagreement among samples).

    Args:
        probs: [n_samples, n_prompts, vocab_size]

    Returns:
        mi: [n_prompts] mutual information values
    """
    # Predictive entropy
    pred_entropy = predictive_entropy(probs, dim=0)

    # Expected data entropy (average entropy of each sample)
    data_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [n_samples, n_prompts]
    expected_data_entropy = data_entropy.mean(dim=0)  # [n_prompts]

    mi = pred_entropy - expected_data_entropy
    return mi


def variance_of_predictions(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute variance across sample predictions.

    Args:
        probs: [n_samples, n_prompts, vocab_size]

    Returns:
        variance: [n_prompts] variance of predicted class probabilities
    """
    # Get predicted class for each sample
    predicted_probs, _ = probs.max(dim=-1)  # [n_samples, n_prompts]

    # Variance across samples
    variance = predicted_probs.var(dim=0)  # [n_prompts]
    return variance


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


def compute_uncertainty_metrics(all_probs: torch.Tensor) -> dict:
    """
    Compute comprehensive uncertainty metrics for a single prompt.

    Args:
        all_probs: [n_samples, 1, vocab_size] probability tensor for one prompt

    Returns:
        metrics: Dictionary with scalar uncertainty measures
    """
    # Get lower and upper bounds from samples
    lower_probs = all_probs.min(dim=0).values.squeeze()  # [vocab_size]
    upper_probs = all_probs.max(dim=0).values.squeeze()  # [vocab_size]

    # Mean prediction
    mean_probs = all_probs.mean(dim=0).squeeze()  # [vocab_size]

    # Compute metrics using shared functions
    metrics = {
        "predictive_entropy": predictive_entropy(all_probs).item(),
        "mutual_information": mutual_information(all_probs).item(),
        "variance": variance_of_predictions(all_probs).item(),
        "mean_confidence": mean_probs.max().item(),
    }

    # Intersection probability entropy (credal set metric)
    intersection_prob = compute_intersection_probability(
        upper_probs.cpu().numpy(),
        lower_probs.cpu().numpy()
    )
    metrics["intersection_probs_entropy"] = -(
        intersection_prob * np.log(intersection_prob + 1e-10)
    ).sum().item()

    # Mean entropy
    metrics["mean_entropy"] = -(
        mean_probs * torch.log(mean_probs + 1e-10)
    ).sum().item()

    return metrics
