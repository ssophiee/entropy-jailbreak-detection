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

    # Get LoRA parameters only
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    print(f"Number of LoRA parameters: {len(lora_params)}")

    # Initialize diagonal Fisher
    fisher_diag = {n: torch.zeros_like(p) for n, p in lora_params.items()}

    # Accumulate squared gradients
    for batch in tqdm(laplace_data, desc="Computing Fisher"):
        model.zero_grad()

        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()

        # Accumulate squared gradients (diagonal of Fisher)
        with torch.no_grad():
            for name, param in lora_params.items():
                if param.grad is not None:
                    fisher_diag[name] += param.grad.pow(2)

        # Free memory
        del outputs, loss

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average over batches
    n_batches = len(laplace_data)
    for name in fisher_diag:
        fisher_diag[name] /= n_batches

    print("✓ Diagonal Fisher computed")
    return fisher_diag, lora_params