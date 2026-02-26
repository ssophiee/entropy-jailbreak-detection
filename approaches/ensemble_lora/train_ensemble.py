"""Train multiple LoRA adapters with different random seeds for ensemble."""
import logging
import os
import gc
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from tqdm import tqdm

log = logging.getLogger(__name__)


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
    base_seed=42,
    gradient_accumulation_steps=4,
):
    """
    Train multiple LoRA adapters with different random seeds.

    The base model is loaded once and reused for all adapters.
    Only LoRA weights and optimizer are reset between adapters.

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

    log.info("=" * 60)
    log.info("Training Ensemble of %d LoRA Adapters", n_adapters)
    log.info("=" * 60)
    log.info("Base Model: %s", base_model_name)
    log.info("LoRA Config: rank=%d, alpha=%d, dropout=%.2f", lora_rank, lora_alpha, lora_dropout)
    log.info("Training: %d epochs, lr=%s", epochs, lr)
    log.info("Save Directory: %s", save_dir)

    # Load base model ONCE and reuse for all adapters
    log.info("Loading base model (cached for all adapters)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    base_model.gradient_checkpointing_enable()
    log.info("Base model loaded (gradient checkpointing enabled)")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )

    for adapter_idx in range(n_adapters):
        seed = base_seed + adapter_idx
        log.info("-" * 60)
        log.info("Training Adapter %d/%d (seed=%d)", adapter_idx + 1, n_adapters, seed)
        log.info("-" * 60)

        # Set random seed for reproducibility and diversity
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Apply fresh LoRA on the cached base model
        model = get_peft_model(base_model, lora_config)
        model.to(device)
        model.train()

        if adapter_idx == 0:
            trainable, total = 0, 0
            for p in model.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            log.info("  Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1

                pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            log.info("  Epoch %d/%d - Average Loss: %.4f", epoch + 1, epochs, avg_loss)

        # Save adapter (only LoRA weights, not base model)
        adapter_name = f"adapter_{adapter_idx}_seed{seed}"
        adapter_path = os.path.join(save_dir, adapter_name)
        os.makedirs(adapter_path, exist_ok=True)

        model.save_pretrained(adapter_path)
        adapter_paths.append(adapter_path)
        log.info("  Saved adapter to: %s", adapter_path)

        # Cleanup LoRA wrapper + optimizer, keep base_model
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup base model after all adapters are done
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("=" * 60)
    log.info("Ensemble Training Complete! Trained %d adapters", n_adapters)
    log.info("Adapters saved to: %s", save_dir)
    log.info("=" * 60)

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
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    base_model.gradient_checkpointing_enable()

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
    gradient_accumulation_steps = 4
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps

        log.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, epoch_loss / len(train_loader))

    # Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        log.info("Saved adapter to: %s", save_path)
        return model, save_path

    return model