"""
Bayesian LoRA inference using Kronecker-factored Laplace approximation.

This properly implements the bayesian_lora library's approach as shown in:
- Reference: https://github.com/MaximeRobeyns/bayesian_lora
- Notebook: bayesian_lora_entropy_pipeline.ipynb (cells 26-30)

Uses jacobian_mean() and variance() to compute Gaussian predictive distribution,
then samples to compute uncertainty metrics.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def compute_predictive_entropy_bayesian_lora(
    model, prompts, tokenizer, kronecker_factors,
    lora_rank, n_kfac=8, prior_var=1.0, n_samples=50,
    device="cuda", target_ids=None, metric="mutual_information",
    max_length=512, debug=False
):
    """
    Compute predictive entropy using Kronecker-factored Bayesian LoRA.

    This follows the reference notebook's implementation exactly.

    Args:
        model: Fine-tuned PEFT model with LoRA adapters
        prompts: List of text prompts to evaluate
        tokenizer: HuggingFace tokenizer
        kronecker_factors: Dict of Kronecker factors from calculate_kronecker_factors()
        lora_rank: Rank of LoRA adapters (e.g., 4, 8, 16)
        n_kfac: Rank used in Kronecker factorization (default: 8)
        prior_var: Prior variance hyperparameter (default: 1.0)
        n_samples: Number of samples from predictive distribution (default: 50)
        device: 'cuda' or 'cpu'
        target_ids: Optional list of token IDs to restrict vocabulary.
                    If None, uses full vocabulary (memory intensive!)
        metric: Uncertainty metric to compute:
                - "mutual_information": I[y;θ|x] = H[E[p]] - E[H[p]] (RECOMMENDED)
                - "mean_entropy": H[E[p]] (less sensitive to epistemic uncertainty)
                - "predictive_variance": Var[max_prob] across samples
        max_length: Maximum sequence length for tokenization
        debug: Print debug information

    Returns:
        List of uncertainty values (one per prompt)
    """
    from bayesian_lora.bayesian_lora.main import jacobian_mean, variance

    uncertainties = []
    model.eval()

    # Convert prior_var to tensor if needed
    if not isinstance(prior_var, torch.Tensor):
        prior_var = torch.tensor(prior_var, device=device, dtype=torch.float32)

    # Convert target_ids to tensor if needed (for efficient GPU indexing)
    if target_ids is not None:
        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, device=device, dtype=torch.long)
        elif target_ids.device != torch.device(device):
            target_ids = target_ids.to(device)
        n_logits = len(target_ids)
        if debug:
            print(f"Using restricted vocabulary: {n_logits} tokens")
    else:
        n_logits = model.config.vocab_size
        if debug:
            print(f"⚠️  Using full vocabulary: {n_logits} tokens (may run out of memory!)")

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Computing Bayesian {metric}")):
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Define output callback to extract logits
        def output_callback(outputs):
            """
            Extract last token logits, optionally restrict to target_ids.
            This is called by jacobian_mean() to get the model outputs.
            """
            logits = outputs.logits[:, -1, :]  # [batch, full_vocab]

            if target_ids is not None:
                # Restrict to target vocabulary
                logits = logits[:, target_ids]  # [batch, n_target_tokens]

            return logits

        with torch.no_grad():
            # Step 1: Compute Jacobian and mean prediction (MAP estimate)
            # jacobian_mean returns: (jacobian_dict, f_mu)
            # - jacobian_dict: gradients of logits w.r.t. LoRA parameters
            # - f_mu: mean logits at MAP parameters
            jacobian, f_mu = jacobian_mean(
                model,
                inputs,
                target_ids=None,  # Always None - we handle selection in output_callback
                is_sc=False,
                output_callback=output_callback
            )

            if debug and prompt_idx == 0:
                print(f"\nDebug info (first prompt):")
                print(f"  f_mu shape: {f_mu.shape}")
                print(f"  Jacobian keys: {list(jacobian.keys())[:3]}...")

            # Step 2: Compute posterior predictive variance using Kronecker factors
            # variance() returns covariance matrix over logits
            # Shape: [batch_size, n_logits, n_logits]
            f_var = variance(
                inputs,
                jacobian,
                kronecker_factors,
                prior_var,
                n_logits,      # Number of output logits
                lora_rank,     # LoRA rank
                n_kfac,        # KFAC rank
                device
            )

            if debug and prompt_idx == 0:
                print(f"  f_var shape: {f_var.shape}")
                print(f"  Variance trace: {torch.trace(f_var[0]).item():.6f}")
                print(f"  Variance max: {f_var.max().item():.6f}")

            # Step 3: Sample logits from posterior predictive distribution
            # p(logits | x, D) ~ N(f_mu, f_var)

            # Add small diagonal for numerical stability
            stabilized_var = f_var + 1e-6 * torch.eye(n_logits, device=device).unsqueeze(0)

            # Cholesky decomposition: Var = L @ L^T
            try:
                L = torch.linalg.cholesky(stabilized_var)
            except RuntimeError as e:
                if debug:
                    print(f"  ⚠️  Cholesky failed for prompt {prompt_idx}, using diagonal approximation")
                # Fallback: use diagonal approximation
                diag_var = torch.diagonal(stabilized_var, dim1=-2, dim2=-1)
                L = torch.diag_embed(torch.sqrt(torch.clamp(diag_var, min=1e-8)))

            # Expand for batch sampling
            f_mu_expanded = f_mu.expand(n_samples, *f_mu.shape)  # [n_samples, batch, n_logits]
            L_expanded = L.expand(n_samples, *L.shape)           # [n_samples, batch, n_logits, n_logits]

            # Sample: logits = f_mu + L @ eps, where eps ~ N(0, I)
            eps = torch.randn_like(f_mu_expanded).unsqueeze(-1)  # [n_samples, batch, n_logits, 1]
            logit_samples = f_mu_expanded + (L_expanded @ eps).squeeze(-1)  # [n_samples, batch, n_logits]

            # Step 4: Compute uncertainty metric
            # Convert logits to probabilities
            prob_samples = torch.softmax(logit_samples, dim=-1)  # [n_samples, batch, n_logits]
            prob_samples = prob_samples.squeeze(1)  # Remove batch dim: [n_samples, n_logits]

            if metric == "mutual_information":
                # I[y;θ|x] = H[E[p(y|θ)]] - E[H[p(y|θ)]]
                # Measures epistemic uncertainty (model disagreement)

                # H[E[p]]: Entropy of the mean distribution
                mean_probs = prob_samples.mean(dim=0)  # [n_logits]
                h_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()

                # E[H[p]]: Expected entropy across samples
                sample_entropies = -(prob_samples * torch.log(prob_samples + 1e-10)).sum(dim=1)  # [n_samples]
                mean_h = sample_entropies.mean()

                uncertainty = (h_mean - mean_h).item()

            elif metric == "mean_entropy":
                # Entropy of the mean distribution
                mean_probs = prob_samples.mean(dim=0)
                uncertainty = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()

            elif metric == "predictive_variance":
                # Variance in predicted max probability
                max_probs = prob_samples.max(dim=1).values  # [n_samples]
                uncertainty = max_probs.var().item()

            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from: mutual_information, mean_entropy, predictive_variance")

            uncertainties.append(uncertainty)

            if debug and (prompt_idx + 1) % 10 == 0:
                print(f"\nProcessed {prompt_idx + 1}/{len(prompts)} prompts")
                print(f"  Last uncertainty ({metric}): {uncertainty:.6f}")

        # Periodic cleanup
        if len(uncertainties) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return uncertainties