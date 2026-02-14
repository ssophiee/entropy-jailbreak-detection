"""Uncertainty quantification for Laplace approximation."""
from typing import List, Union, Dict, Optional
from torch.nn import functional as F
import torch
from tqdm import tqdm

from src.constants import MAX_LENGTH, PRIOR_PRECISION
from src.uncertainty import compute_uncertainty_metrics
from approaches.prompt_entropy.prompt_entropy import aggregate_entropy_features


def compute_predictive_entropy(
    model,
    prompts: List[str],
    tokenizer,
    fisher_diag,
    n_samples: int = 20,
    temperature: float = 0.05,
    device: str = "cuda",
    metric: Optional[str] = "intersection_probs_entropy",
    debug: bool = False,
) -> Union[List[float], List[Dict[str, float]]]:
    """
    Compute uncertainty metrics using Laplace approximation.

    Args:
        model: The language model with LoRA parameters
        prompts: List of input prompts
        tokenizer: Tokenizer for the model
        fisher_diag: Diagonal Fisher information for LoRA parameters
        n_samples: Number of posterior samples
        temperature: Temperature scaling for posterior sampling
        device: Device to run computations on
        metric: Specific metric to return, or None for all metrics
            Options: "predictive_entropy", "mutual_information", "variance",
                     "mean_confidence", "intersection_probs_entropy", "mean_entropy"
        debug: Print debug information

    Returns:
        If metric is specified: List of values for that metric
        If metric is None: List of dicts with all metrics
    """
    print(f"Computing uncertainty for {len(prompts)} prompts...")

    model.eval()
    all_metrics = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Computing uncertainty"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        logit_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior: θ ~ N(θ_MAP, (temperature / Fisher))
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std

                        if debug:
                            param_norm = original_state[name].norm().item()
                            noise_norm = noise.norm().item()
                            noise_to_param_ratio = noise_norm / (param_norm + 1e-10)
                            print(f"\n{name}:")
                            print(f"  Param norm: {param_norm:.6f}")
                            print(f"  Noise norm: {noise_norm:.6f}")
                            print(f"  Noise/Param ratio: {noise_to_param_ratio:.4f}")

                        param.data = original_state[name] + noise

                # Forward pass with sampled parameters
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_samples.append(logits.cpu())

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Compute all metrics using shared function
        all_logits = torch.stack(logit_samples, dim=0)  # [n_samples, 1, vocab_size]
        all_probs = F.softmax(all_logits, dim=-1)  # [n_samples, 1, vocab_size]
        metrics = compute_uncertainty_metrics(all_probs)
        all_metrics.append(metrics)

        # Periodic cleanup
        if len(all_metrics) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if metric is None:
        return all_metrics
    else:
        return [m[metric] for m in all_metrics]


def compute_entropy_trace_features(
    model,
    prompts: List[str],
    tokenizer,
    fisher_diag,
    n_samples: int = 20,
    temperature: float = 0.05,
    device: str = "cuda",
) -> List[Dict[str, float]]:
    """
    Compute per-position predictive-entropy trace using Laplace posterior
    sampling, then aggregate each trace into the same 20 scalar features
    used by the prompt-entropy approach.

    For each token position t, predictive entropy is:
        H[E_samples[p(x_{t+1} | x_{<=t})]]
    where samples are drawn from the Laplace posterior over LoRA weights.

    Args:
        model: The language model with LoRA parameters
        prompts: List of input prompts
        tokenizer: Tokenizer for the model
        fisher_diag: Diagonal Fisher information for LoRA parameters
        n_samples: Number of posterior samples
        temperature: Temperature scaling for posterior sampling
        device: Device to run computations on

    Returns:
        List of dicts (one per prompt), each with the 20 aggregated features.
    """
    print(f"Computing entropy trace features for {len(prompts)} prompts...")

    model.eval()
    all_features = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Entropy trace features"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        T = inputs["input_ids"].shape[1]
        if T < 2:
            all_features.append(aggregate_entropy_features([]))
            continue

        # Collect full-sequence logits from each posterior sample
        logit_samples = []  # will be [n_samples, T, V]

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std
                        param.data = original_state[name] + noise

                # Forward pass — keep ALL positions
                outputs = model(**inputs)
                logit_samples.append(outputs.logits.squeeze(0).cpu())  # [T, V]

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Stack: [n_samples, T, V]
        all_logits = torch.stack(logit_samples, dim=0)

        # Positions predicting next token: [:, :-1, :]
        logits_pred = all_logits[:, :-1, :]            # [n_samples, T-1, V]
        probs = F.softmax(logits_pred, dim=-1)         # [n_samples, T-1, V]

        # Mean probability across posterior samples at each position
        mean_probs = probs.mean(dim=0)                 # [T-1, V]

        # Predictive entropy per position
        H = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)  # [T-1]
        entropies = H.tolist()

        features = aggregate_entropy_features(entropies)
        all_features.append(features)

        # Periodic cleanup
        if len(all_features) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_features
