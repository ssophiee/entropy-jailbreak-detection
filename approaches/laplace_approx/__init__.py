"""laplace_approx approach package.

Laplace Approximation for Uncertainty Quantification
=====================================================

This package implements Laplace approximation using LoRA adapters:

Key Features:
- Diagonal Fisher Information: Efficient approximation of posterior
- Posterior Sampling: Sample from approximate posterior for uncertainty
- Credal Sets: Compute credal set metrics for robust uncertainty

Usage:
------

# 1. Setup model with LoRA
from approaches.laplace_approx import setup_model_and_lora

model, tokenizer = setup_model_and_lora(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device=device,
    lora_rank=8
)

# 2. Train and compute Fisher information
from approaches.laplace_approx import train_lora, collect_laplace_data, compute_diagonal_fisher

model = train_lora(model, train_loader, epochs=1)
laplace_data = collect_laplace_data(model, train_loader, max_batches=50)
fisher_diag, lora_params = compute_diagonal_fisher(model, laplace_data)

# 3. Compute uncertainty metrics
from approaches.laplace_approx import compute_predictive_entropy

uncertainties = compute_predictive_entropy(
    model, prompts, tokenizer, fisher_diag,
    n_samples=20, metric="mutual_information"
)
"""

# Laplace approximation (Fisher computation)
from .laplace import (
    collect_laplace_data,
    compute_diagonal_fisher,
)

# Training
from .training import (
    setup_model_and_lora,
    train_lora,
    load_finetuned,
)

# Uncertainty
from .uncertainty import (
    compute_predictive_entropy,
    compute_entropy_trace_features,
)

__all__ = [
    # Laplace
    "collect_laplace_data",
    "compute_diagonal_fisher",
    # Training
    "setup_model_and_lora",
    "train_lora",
    "load_finetuned",
    # Uncertainty
    "compute_predictive_entropy",
    "compute_entropy_trace_features",
]
