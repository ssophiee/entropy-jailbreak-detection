"""ensemble_lora approach package.

LoRA Ensemble Approach for Uncertainty Quantification
======================================================

This package implements ensemble method using LoRA (Low-Rank Adapters):

Key Features:
- Memory Efficient: Only one base model + multiple tiny adapters
- Better Uncertainty: Ensemble provides calibrated uncertainty estimates
- Practical: Feasible for large models (LLaMA, Qwen, GPT, etc.)

Usage:
------

# 1. Train ensemble of LoRA adapters
from approaches.ensemble_lora import train_lora_ensemble

adapter_paths = train_lora_ensemble(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    train_loader=my_dataloader,
    n_adapters=5,
    epochs=1,
    save_dir="./saved_models/ensemble_lora"
)

# 2. Run inference with adapter swapping
from approaches.ensemble_lora import EnsembleLoRAInference

ensemble = EnsembleLoRAInference(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    adapter_paths=adapter_paths
)

# Get ensemble predictions
agg_probs, all_probs = ensemble.ensemble_predict(["What is AI?"])

# 3. Compute uncertainty metrics
from approaches.ensemble_lora import compute_uncertainty_metrics
metrics = compute_uncertainty_metrics(all_probs)
"""

# Training
from .train_ensemble import (
    train_lora_ensemble,
    train_single_adapter
)

# Inference
from .inference import (
    EnsembleLoRAInference,
    load_ensemble_from_directory,
    swap_and_predict
)

# Uncertainty
from .uncertainty import (
    predictive_entropy,
    mutual_information,
    variance_of_predictions,
    expected_calibration_error,
    compute_uncertainty_metrics,
    compute_predictive_entropy,
)

__all__ = [
    # Training
    "train_lora_ensemble",
    "train_single_adapter",
    # Inference
    "EnsembleLoRAInference",
    "load_ensemble_from_directory",
    "swap_and_predict",
    # Uncertainty
    "predictive_entropy",
    "mutual_information",
    "variance_of_predictions",
    "expected_calibration_error",
    "compute_uncertainty_metrics",
    "compute_predictive_entropy",
]
