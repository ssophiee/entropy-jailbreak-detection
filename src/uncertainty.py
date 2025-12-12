from torch.nn import functional as F
import torch
from tqdm import tqdm
import numpy as np

import os, sys
import torch
import argparse
from datetime import datetime

this_dir = os.path.dirname(__file__)        
repo_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.insert(0, repo_root)


from src.constants import MAX_LENGTH, PRIOR_PRECISION

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
        print(f"Alpha: {alpha[0]:.6f}")
        intersection_probs = (upper_probs - lower_probs) * alpha + lower_probs

    return intersection_probs


def compute_uncertainty_metrics(all_logits, top_k=100):
    """Compute multiple credal set metrics"""
    # TODO: on the call was mentioned that softmax might not be ideal here
    all_probs = F.softmax(all_logits, dim=-1)  # [n_samples, 1, vocab_size]

    lower_probs = all_probs.min(dim=0).values.squeeze()
    upper_probs = all_probs.max(dim=0).values.squeeze()

    # Entropy-based metrics (mean entropy, intersection entropy)
    mean_probs = all_probs.mean(dim=0).squeeze()
    mean_probs_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()

    intersection_prob = compute_intersection_probability(upper_probs.cpu().numpy(), lower_probs.cpu().numpy())
    intersection_prob_entropy = -(intersection_prob * np.log(intersection_prob + 1e-10)).sum().item()

    # TODO: credal set might not be needed (or might be incorrectly defined)
    _, top_indices = torch.topk(mean_probs, k=top_k)
    top_probs = all_probs[:, 0, top_indices]
    lower_topk = top_probs.min(dim=0).values
    upper_topk = top_probs.max(dim=0).values
    widths_topk = upper_topk - lower_topk
    widths_full = upper_probs - lower_probs

    # TODO: might be the case that those metrics are not needed: credal_width_full, credal_width_topk, 
    # credal_width_topk_mean, credal_width_topk_max
    return {
        # 'credal_width_full': widths_full.sum().item(),
        # 'credal_width_topk': widths_topk.sum().item(),
        # 'credal_width_topk_mean': widths_topk.mean().item(),
        # 'credal_width_topk_max': widths_topk.max().item(),
        # 'top_k': top_k,
        'vocab_size': all_probs.shape[-1],
        'mean_entropy': mean_probs_entropy,
        "intersection_probs_entropy": intersection_prob_entropy
    }
def compute_predictive_credal_sets(model, prompts, tokenizer, fisher_diag,
                                   n_samples=20, temperature=0.05, 
                                   top_k=100, device="cuda"):
    """
    Compute credal sets using Laplace approximation.
    
    Args:
        top_k: If None, use full vocabulary. If int, use top-K budgeting.
    """
    print(f"Computing credal sets for {len(prompts)} prompts...")
    
    model.eval()
    results = []
    
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Computing credal sets"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)
        
        logit_samples = []
        
        with torch.no_grad():
            for sample_idx in range(n_samples):
                # Sample from posterior
                total_noise_norm = 0.0
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std

                        # Print meaningful statistics
                        param_norm = original_state[name].norm().item()
                        noise_norm = noise.norm().item()
                        noise_to_param_ratio = noise_norm / (param_norm + 1e-10)
                        
                        print(f"\n{name}:")
                        print(f"  Param norm: {param_norm:.6f}")
                        print(f"  Noise norm: {noise_norm:.6f}")
                        print(f"  Noise/Param ratio: {noise_to_param_ratio:.4f} ({noise_to_param_ratio*100:.2f}%)")
                        print(f"  Param mean: {original_state[name].mean().item():.6f}, std: {original_state[name].std().item():.6f}")
                        print(f"  Noise mean: {noise.mean().item():.6f}, std: {noise.std().item():.6f}")

                        param.data = original_state[name] + noise
                        total_noise_norm += noise.norm().item()

                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_samples.append(logits.cpu())

                # Restore
                for name, param in lora_params.items():
                    param.data = original_state[name]
        
        # Compute metrics 
        all_logits = torch.stack(logit_samples, dim=0)
        metrics = compute_uncertainty_metrics(all_logits, top_k=top_k)
        results.append(metrics)
        
        # Cleanup
        if len(results) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def compute_predictive_entropy(model, prompts, tokenizer, fisher_diag,
                               n_samples=20, temperature=0.05, device="cuda",
                               metric="mutual_information", debug=False):
    """
    Compute epistemic uncertainty metrics using Laplace approximation.

    Args:
        metric: One of "mutual_information", "predictive_variance", "mean_entropy"
            - mutual_information: I[y;θ|x] = H[E[p]] - E[H[p]] (RECOMMENDED for adversarial detection)
            - predictive_variance: Variance of max probability across samples
            - mean_entropy: H[E[p]] (original, less sensitive to epistemic uncertainty)

    Returns:
        List of uncertainty values (higher = more uncertain)
    """
    print(f"Computing {metric} for {len(prompts)} prompts...")

    model.eval()
    uncertainties = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc=f"Computing {metric}"):
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        logit_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior: θ ~ N(θ_MAP, temperature / Fisher)
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION  # Add small constant for stability
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std

                        # Print meaningful statistics
                        param_norm = original_state[name].norm().item()
                        noise_norm = noise.norm().item()
                        noise_to_param_ratio = noise_norm / (param_norm + 1e-10)

                        if debug:
                          print(f"\n{name}:")
                          print(f"  Param norm: {param_norm:.6f}")
                          print(f"  Noise norm: {noise_norm:.6f}")
                          print(f"  Noise/Param ratio: {noise_to_param_ratio:.4f} ({noise_to_param_ratio*100:.2f}%)")
                          print(f"  Param mean: {original_state[name].mean().item():.6f}, std: {original_state[name].std().item():.6f}")
                          print(f"  Noise mean: {noise.mean().item():.6f}, std: {noise.std().item():.6f}")

                        param.data = original_state[name] + noise

                # Forward pass with sampled parameters
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Last token logits
                logit_samples.append(logits.cpu())

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Compute uncertainty metric
        all_logits = torch.stack(logit_samples, dim=0)  # [n_samples, 1, vocab_size]
        all_probs = F.softmax(all_logits, dim=-1).squeeze(1)  # [n_samples, vocab_size]

        if metric == "mutual_information":
            # I[y;θ|x] = H[E[p(y|θ)]] - E[H[p(y|θ)]]
            # This captures epistemic uncertainty - how much the model disagrees with itself

            # H[E[p]]: Entropy of the mean distribution
            mean_probs = all_probs.mean(dim=0)  # [vocab_size]
            h_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()

            # E[H[p]]: Expected entropy across samples
            sample_entropies = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=1)  # [n_samples]
            mean_h = sample_entropies.mean()

            uncertainty = (h_mean - mean_h).item()

        elif metric == "predictive_variance":
            # Variance in the predicted probability of the most likely class
            max_probs = all_probs.max(dim=1).values  # [n_samples]
            uncertainty = max_probs.var().item()

        elif metric == "mean_entropy":
            # Original metric: entropy of the mean distribution
            mean_probs = all_probs.mean(dim=0)
            uncertainty = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

        uncertainties.append(uncertainty)

        # Periodic cleanup
        if len(uncertainties) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return uncertainties