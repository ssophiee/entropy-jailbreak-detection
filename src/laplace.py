import torch
from tqdm import tqdm
from src.constants import DEVICE

def collect_laplace_data(model, train_loader, max_batches=None):
    """Collect limited data for Laplace approximation to save memory"""
    laplace_data = []

    for idx, batch in enumerate(tqdm(train_loader, desc="Collecting Laplace data")):
        if max_batches and idx >= max_batches:
            break

        laplace_data.append({
            'input_ids': batch['input_ids'].to(DEVICE),
            'attention_mask': batch['attention_mask'].to(DEVICE),
            'labels': batch['labels'].to(DEVICE)
        })

    print(f"Collected {len(laplace_data)} batches for Laplace approximation")
    return laplace_data


def compute_diagonal_fisher(model, laplace_data, device="cuda"):
    """
    Compute diagonal Fisher information matrix (Laplace approximation).
    This is a fallback if bayesian_lora fails.
    """
    print("Computing diagonal Fisher approximation...")

    model.train()

    # CRITICAL: Ensure LoRA parameters have requires_grad=True
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # Get LoRA parameters only
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    print(f"Number of LoRA parameters: {len(lora_params)}")

    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found with requires_grad=True!")

    # Initialize diagonal Fisher
    fisher_diag = {n: torch.zeros_like(p) for n, p in lora_params.items()}

    # Accumulate squared gradients
    grad_norms = []
    for batch_idx, batch in enumerate(tqdm(laplace_data, desc="Computing Fisher")):
        model.zero_grad()

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()

        # Debug: Check gradient magnitudes
        if batch_idx == 0:
            print(f"\nBatch 0 diagnostics:")
            print(f"  Loss: {loss.item():.6f}")
            total_grad_norm = 0.0
            for name, param in lora_params.items():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
            print(f"  Total gradient norm: {total_grad_norm:.6e}")

        # Accumulate squared gradients (diagonal of Fisher)
        with torch.no_grad():
            batch_grad_norm = 0.0
            for name, param in lora_params.items():
                if param.grad is not None:
                    fisher_diag[name] += param.grad.pow(2)
                    batch_grad_norm += param.grad.norm().item()
            grad_norms.append(batch_grad_norm)

        # Free memory
        del outputs, loss

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # DO NOT average - keep as sum for proper Bayesian scaling
    # The Fisher matrix represents curvature of the sum of log-likelihoods,
    # so it should scale with the number of data points
    n_batches = len(laplace_data)
    

    # Report Fisher statistics
    print("\n" + "="*60)
    print("FISHER MATRIX STATISTICS")
    print("="*60)
    print(f"Average gradient norm per batch: {sum(grad_norms)/len(grad_norms):.6e}")
    print(f"\nFisher diagonal statistics (top 5 layers):")
    fisher_stats = []
    for name, values in fisher_diag.items():
        fisher_stats.append((name, values.max().item(), values.mean().item()))
    fisher_stats.sort(key=lambda x: x[1], reverse=True)
    for name, max_val, mean_val in fisher_stats[:5]:
        print(f"  {name}")
        print(f"    Max: {max_val:.6e}, Mean: {mean_val:.6e}")

    # Warning if Fisher values are too small
    max_fisher = max(v.max().item() for v in fisher_diag.values())
    if max_fisher < 1e-2:
        print("\n⚠️  WARNING: Fisher values are very small (< 1e-2)!")
        print("   This may cause instability in posterior sampling.")
        print("   Possible causes:")
        print("   1. Gradients are too small (model overfitted or loss too small)")
        print("   2. Too few batches for Fisher computation")
        print("   3. Learning rate was too small during training")
    elif max_fisher > 1e3:
        print("\n⚠️  WARNING: Fisher values are very large (> 1e3)!")
        print("   Consider using more batches or reducing temperature.")
    print("="*60)

    print("✓ Diagonal Fisher computed")
    return fisher_diag, lora_params