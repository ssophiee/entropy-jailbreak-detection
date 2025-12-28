"""Train multiple LoRA adapters with different random seeds for ensemble."""
import os
import gc
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from tqdm import tqdm


def train_lora_ensemble(
    base_model_name,
    train_loader,
    n_adapters=5,
    epochs=1,
    lr=3e-4,
    lora_rank=4,
    lora_alpha=8,
    lora_dropout=0.1,
    device="cuda",
    save_dir=None,
    base_seed=42
):
    """
    Train multiple LoRA adapters with different random seeds.

    This creates an ensemble of LoRA adapters on the same base model,
    each trained with a different initialization seed. This provides
    diversity for better uncertainty estimation.

    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-3B-Instruct")
        train_loader: PyTorch DataLoader with 'input_ids', 'attention_mask', 'labels'
        n_adapters: Number of LoRA adapters to train
        epochs: Training epochs per adapter
        lr: Learning rate
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save adapters (default: ./saved_models/ensemble_lora)
        base_seed: Base random seed (each adapter uses base_seed + i)

    Returns:
        adapter_paths: List of paths to saved adapters
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "saved_models", "ensemble_lora")

    os.makedirs(save_dir, exist_ok=True)
    adapter_paths = []

    print(f"\n{'='*60}")
    print(f"Training Ensemble of {n_adapters} LoRA Adapters")
    print(f"{'='*60}")
    print(f"Base Model: {base_model_name}")
    print(f"LoRA Config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Training: {epochs} epochs, lr={lr}")
    print(f"Save Directory: {save_dir}\n")

    for adapter_idx in range(n_adapters):
        seed = base_seed + adapter_idx
        print(f"\n{'─'*60}")
        print(f"Training Adapter {adapter_idx + 1}/{n_adapters} (seed={seed})")
        print(f"{'─'*60}")

        # Set random seed for reproducibility and diversity
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load fresh base model
        print(f"  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,  # Reduce RAM usage during loading
            use_safetensors=True,     # Use safer tensor format
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            init_lora_weights=True  # Random initialization
        )

        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        model.to(device)
        model.train()

        if adapter_idx == 0:  # Print only once
            print(f"  Trainable parameters:")
            model.print_trainable_parameters()

        # Train
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                batch_count += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            print(f"  Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        # Save adapter
        adapter_name = f"adapter_{adapter_idx}_seed{seed}"
        adapter_path = os.path.join(save_dir, adapter_name)
        os.makedirs(adapter_path, exist_ok=True)

        model.save_pretrained(adapter_path)
        adapter_paths.append(adapter_path)
        print(f"  ✓ Saved adapter to: {adapter_path}")

        # Cleanup
        del model, base_model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"✓ Ensemble Training Complete!")
    print(f"{'='*60}")
    print(f"Trained {n_adapters} adapters")
    print(f"Adapters saved to: {save_dir}\n")

    return adapter_paths

def train_single_adapter(
    base_model_name,
    train_loader,
    epochs=1,
    lr=3e-4,
    lora_rank=4,
    lora_alpha=8,
    lora_dropout=0.1,
    device="cuda",
    save_path=None,
    seed=42
):
    """
    Train a single LoRA adapter (helper function).

    Args:
        base_model_name: HuggingFace model name
        train_loader: PyTorch DataLoader
        epochs: Training epochs
        lr: Learning rate
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: Device to train on
        save_path: Path to save adapter (optional)
        seed: Random seed

    Returns:
        model: Trained PEFT model
        save_path: Path where adapter was saved (if save_path provided)
    """
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure and apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    # Train
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"Saved adapter to: {save_path}")
        return model, save_path

    return model
