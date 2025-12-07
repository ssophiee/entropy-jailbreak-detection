from torch.nn import functional as F
import torch
from tqdm import tqdm
import numpy as np
from src.constants import MAX_LENGTH

# # from https://github.com/WangKaizheng/CreINNs/blob/main/CreINNs_main_implementation/CreINNTestMulti.py
# def compute_intersection_probability(upper_probs, lower_probs):
#     alpha_num = 1.0 - np.sum(lower_probs, axis=-1, keepdims=True)
#     alpha_denom = np.sum(upper_probs-lower_probs, axis=-1, keepdims=True)

#     samples = alpha_num.shape[0]
#     alpha = alpha_num/alpha_denom

#     intersection_probs = (upper_probs - lower_probs) * alpha + 1.0 * lower_probs

#     return intersection_probs

def compute_intersection_probability(upper_probs, lower_probs):
    alpha_num = 1.0 - np.sum(lower_probs)
    alpha_denom = np.sum(upper_probs - lower_probs)
    
    alpha = alpha_num / alpha_denom
    
    intersection_probs = (upper_probs - lower_probs) * alpha + lower_probs
    
    return intersection_probs

# def compute_intersection_probability(upper_probs, lower_probs):
#     """Compute intersection probability with numerical stability fixes"""
    
#     # Ensure inputs are numpy arrays
#     if not isinstance(upper_probs, np.ndarray):
#         upper_probs = upper_probs.cpu().numpy() if hasattr(upper_probs, 'cpu') else np.array(upper_probs)
#     if not isinstance(lower_probs, np.ndarray):
#         lower_probs = lower_probs.cpu().numpy() if hasattr(lower_probs, 'cpu') else np.array(lower_probs)
    
#     # Normalize to ensure they sum to 1
#     upper_probs = upper_probs / (np.sum(upper_probs) + 1e-10)
#     lower_probs = lower_probs / (np.sum(lower_probs) + 1e-10)
    
#     alpha_num = 1.0 - np.sum(lower_probs)
#     alpha_denom = np.sum(upper_probs - lower_probs)
    
#     # Handle edge cases
#     if alpha_denom < 1e-10:  # Credal set collapsed (no uncertainty)
#         return lower_probs
    
#     if alpha_num < 0:  # Numerical error
#         return lower_probs
    
#     # Compute alpha with clipping to valid range [0, large value]
#     alpha = np.clip(alpha_num / alpha_denom, 0, 1e6)
    
#     # Compute intersection
#     intersection_probs = (upper_probs - lower_probs) * alpha + lower_probs
    
#     # Final normalization and clipping
#     intersection_probs = np.clip(intersection_probs, 0, 1)
#     intersection_probs = intersection_probs / (np.sum(intersection_probs) + 1e-10)
    
#     return intersection_probs


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
                                   top_k=None, device="cuda"):
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
            for _ in range(n_samples):
                # Sample from posterior
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + 1e-6
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std
                        param.data = original_state[name] + noise
                
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_samples.append(logits.cpu())
                
                # Restore
                for name, param in lora_params.items():
                    param.data = original_state[name]
        
        # Compute metrics 
        all_logits = torch.stack(logit_samples, dim=0)
        metrics = compute_credal_metrics(all_logits, top_k=top_k)
        results.append(metrics)
        
        # Cleanup
        if len(results) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def compute_predictive_entropy(model, prompts, tokenizer, fisher_diag,
                               n_samples=20, temperature=0.05, device="cuda"):
    """
    Compute predictive entropy using Laplace approximation.
    H[p(y|x,D)] = -∑ p(y|x,D) log p(y|x,D)
    where p(y|x,D) ≈ ∫ p(y|x,θ) q(θ|D) dθ
    """
    print(f"Computing entropy for {len(prompts)} prompts...")

    model.eval()
    entropies = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Computing entropy"):
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
                        precision = fisher_diag[name] + 1e-6  # Add small constant for stability
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std
                        param.data = original_state[name] + noise


                # Forward pass with sampled parameters
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Last token logits
                logit_samples.append(logits.cpu())

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Compute predictive distribution: p(y|x,D) ≈ mean over samples
        all_logits = torch.stack(logit_samples, dim=0)  # [n_samples, 1, vocab_size]
        mean_probs = F.softmax(all_logits, dim=-1).mean(dim=0).squeeze()  # [vocab_size]
        # Compute entropy: H = -∑ p log p
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()
        entropies.append(entropy)

        # Periodic cleanup
        if len(entropies) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return entropies