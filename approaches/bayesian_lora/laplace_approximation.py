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
                               n_kfac=8, lr_threshold=1e-4):
    """
    Compute Kronecker factors using PATCHED bayesian-lora

    With the patches applied, this works directly with PEFT LoRA models!
    """
    print("Computing Kronecker factors (this may take a while)...")

    # Get base model (no merging needed with patches!)
    print("  Step 1: Preparing base model...")
    base_model = model.get_base_model()
    base_model.requires_grad_(True)

    # Ensure LoRA parameters have gradients
    for name, param in base_model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    # Verify gradients are enabled
    grad_params = sum(p.requires_grad for p in base_model.parameters())
    total_params = sum(1 for _ in base_model.parameters())
    print(f"    ✓ Gradients enabled for {grad_params}/{total_params} parameters")

    # Forward call for KFAC
    def fwd_call(model_instance, batch):
        """
        Forward call that handles pre-tokenized data

        batch is a dict with keys: 'input_ids', 'attention_mask', 'labels'
        """
        if isinstance(batch, dict):
            # Pre-tokenized: Use input_ids and attention_mask directly
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = base_model(  # Use base_model here!
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            # Raw text: Tokenize first (fallback for compatibility)
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                             truncation=True, max_length=512).to(device)
            outputs = base_model(**inputs)  # Use base_model here!

        # Return logits for last token position
        return outputs.logits[:, -1, :]

    # Compute KFAC with patched bayesian-lora
    print("  Step 2: Computing Kronecker factors...")
    factors = calculate_kronecker_factors(
        base_model,  # 🔧 FIX: Use base_model, not "modelBaseExceptionGroup"
        fwd_call,
        train_loader,
        n_kfac=n_kfac,
        lr_threshold=lr_threshold,
        target_module_keywords=["q_proj", "v_proj"],
        use_tqdm=True
    )

    print(f"\n✓ Computed Kronecker factors for {len(factors)} modules")
    for name in list(factors.keys())[:3]:
        print(f"  - {name}")
    if len(factors) > 3:
        print(f"  ... and {len(factors) - 3} more modules")

    return factors, base_model  # Return base_model, not merged model


def get_lora_rank(model):
    """Extract LoRA rank from the model"""
    for name, module in model.named_modules():
        if 'lora' in name.lower() and hasattr(module, 'r'):
            return module.r
    return 8  # Default fallback