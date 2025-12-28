from bayesian_lora.bayesian_lora import calculate_kronecker_factors
from bayesian_lora.bayesian_lora.main import jacobian_mean, variance
import torch

def fwd_call_wrapper(model, batch, tokenizer, device):
    """
    Wrapper for model forward call that returns logits
    Required by bayesian_lora library
    """
    # If batch is already tokenized
    if isinstance(batch, dict) and 'input_ids' in batch:
      batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
      outputs = model(**batch)
      return outputs.logits[:, -1, :]  # Last token logits

    # If batch is raw text
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    return outputs.logits[:, -1, :]


def compute_kronecker_factors(model, train_loader, tokenizer, device="cuda",
                               n_kfac=8, lr_threshold=1e-2, max_batches=None,
                               target_modules=["lora"]):
    """
    Compute Kronecker factors using bayesian-lora library.

    This follows the reference notebook implementation (cells 23-24).

    Args:
        model: PEFT model with LoRA adapters
        train_loader: Can be either:
                      - DataLoader yielding dicts with 'input_ids', 'attention_mask'
                      - List of text prompts (will be converted to batches)
        tokenizer: HuggingFace tokenizer
        device: 'cuda' or 'cpu'
        n_kfac: Rank for Kronecker factorization (default: 8)
        lr_threshold: Low-rank threshold (default: 1e-2)
        max_batches: Maximum number of batches to use (for memory efficiency)
        target_modules: List of module keywords to target (default: ["lora"])

    Returns:
        factors: Dict of Kronecker factors
        model: The PEFT model (returned for convenience)
    """
    print("="*60)
    print("COMPUTING KRONECKER FACTORS")
    print("="*60)

    # Prepare data loader
    # If train_loader is a list of prompts, convert to batches
    if isinstance(train_loader, list):
        print(f"Converting {len(train_loader)} prompts to batches...")
        batch_size = 2
        if max_batches is not None:
            train_loader = train_loader[:max_batches * batch_size]
        kfac_loader = [train_loader[i:i+batch_size]
                      for i in range(0, len(train_loader), batch_size)]
        print(f"Using {len(kfac_loader)} batches for KFAC computation")
    else:
        # It's already a DataLoader
        if max_batches is not None:
            print(f"Using first {max_batches} batches for KFAC computation")
            # Convert to list and slice
            kfac_loader = []
            for idx, batch in enumerate(train_loader):
                if idx >= max_batches:
                    break
                kfac_loader.append(batch)
        else:
            kfac_loader = train_loader

    # Define forward call wrapper
    def fwd_call(model_instance, batch):
        """
        Wrapper function for model forward pass.
        Required by bayesian_lora library.

        Returns the last token logits for each prompt in the batch.
        """
        # Handle both text prompts and pre-tokenized batches
        if isinstance(batch, dict) and 'input_ids' in batch:
            # Pre-tokenized batch
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        else:
            # Text prompts
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

        outputs = model_instance(**inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits

        return logits

    # Clear cache before computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute Kronecker factors
    try:
        factors = calculate_kronecker_factors(
            model,                          # PEFT model with LoRA
            fwd_call,                       # Forward call wrapper
            kfac_loader,                    # Training data
            n_kfac=n_kfac,                 # Rank for Kronecker factorization
            lr_threshold=lr_threshold,      # Low-rank threshold
            target_module_keywords=target_modules,  # Target LoRA modules
            use_tqdm=True
        )

        print(f"\nKronecker factors computed for {len(factors)} modules")
        print(f"Module names (first 3): {list(factors.keys())[:3]}...")
        if len(factors) > 3:
            print(f"  ... and {len(factors) - 3} more modules")

    except Exception as e:
        print(f"\nError computing Kronecker factors: {e}")
        raise

    print("="*60)
    return factors, model


def get_lora_rank(model):
    """Extract LoRA rank from the model"""
    for name, module in model.named_modules():
        if 'lora' in name.lower() and hasattr(module, 'r'):
            return module.r
    return 8  # Default fallback