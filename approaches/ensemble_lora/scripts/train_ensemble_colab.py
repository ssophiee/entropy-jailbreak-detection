#!/usr/bin/env python3
"""
Training script for LoRA ensemble - OPTIMIZED FOR GOOGLE COLAB.

This version includes:
- Better memory management
- Model caching to avoid reloading
- Progress monitoring
- Timeout handling

Usage in Colab:
    !python approaches/ensemble_lora/scripts/train_ensemble_colab.py
    !python approaches/ensemble_lora/scripts/train_ensemble_colab.py --n_adapters 3
"""
import os
import sys
import gc
import argparse
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data, create_dataloader
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


def train_single_lora_adapter(
    base_model,
    tokenizer,
    train_loader,
    adapter_idx,
    seed,
    save_path,
    epochs=5,
    lr=1e-2,
    lora_rank=16,
    lora_alpha=32,
    device="cuda"
):
    """Train a single LoRA adapter on pre-loaded base model."""
    print(f"\n{'─'*60}")
    print(f"Training Adapter {adapter_idx + 1} (seed={seed})")
    print(f"{'─'*60}")

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True
    )

    # Apply LoRA to base model (creates a copy with adapters)
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    if adapter_idx == 0:
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

    # Save adapter (only LoRA weights, not base model)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"  ✓ Saved adapter to: {save_path}")

    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return save_path


def main(args):
    print("=" * 80)
    print("ENSEMBLE LORA TRAINING SCRIPT (COLAB OPTIMIZED)")
    print("=" * 80)
    print(f"Device: {constants.DEVICE}")
    print(f"Base Model: {constants.MODEL_NAME}")
    print(f"Number of Adapters: {args.n_adapters}")
    print(f"LoRA Rank: {constants.LORA_RANK}")
    print(f"LoRA Alpha: {constants.LORA_ALPHA}")
    print(f"Epochs: {constants.EPOCHS}")
    print(f"Learning Rate: {constants.LEARNING_RATE}")
    print(f"Batch Size: {constants.BATCH_SIZE}")
    print(f"Max Length: {constants.MAX_LENGTH}")
    print(f"Base Seed: {args.base_seed}")
    print("=" * 80)

    # 1. Load training data
    print("\n[1/4] Loading training data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )
    train_prompts = data['train_prompts']
    print(f"Loaded {len(train_prompts)} training prompts")

    # 2. Load tokenizer
    print("\n[2/4] Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Create DataLoader
    print("\n[3/4] Creating DataLoader...")
    train_loader = create_dataloader(
        train_prompts,
        tokenizer,
        max_length=constants.MAX_LENGTH,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    print(f"DataLoader created with {len(train_loader)} batches")

    # 4. Load base model ONCE (this is the key optimization)
    print("\n[4/4] Loading base model (this may take a few minutes)...")
    print("  NOTE: Model will be cached and reused for all adapters")

    # Free up memory before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            constants.MODEL_NAME,
            torch_dtype=torch.float16 if str(constants.DEVICE) == "cuda" else torch.float32,
            device_map="auto" if str(constants.DEVICE) == "cuda" else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        print(f"  ✓ Base model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        print("\n  Troubleshooting tips:")
        print("    1. Make sure you have GPU enabled (Runtime > Change runtime type)")
        print("    2. Try restarting runtime and clearing cache")
        print("    3. Run: !rm -rf /root/.cache/huggingface/hub/*")
        raise

    # 5. Train ensemble using the cached base model
    print("\n" + "=" * 60)
    print(f"Training Ensemble of {args.n_adapters} LoRA Adapters")
    print("=" * 60)

    adapter_paths = []

    for adapter_idx in range(args.n_adapters):
        seed = args.base_seed + adapter_idx
        adapter_name = f"adapter_{adapter_idx}_seed{seed}"
        adapter_path = os.path.join(args.save_dir, adapter_name)

        try:
            saved_path = train_single_lora_adapter(
                base_model=base_model,
                tokenizer=tokenizer,
                train_loader=train_loader,
                adapter_idx=adapter_idx,
                seed=seed,
                save_path=adapter_path,
                epochs=constants.EPOCHS,
                lr=constants.LEARNING_RATE,
                lora_rank=constants.LORA_RANK,
                lora_alpha=constants.LORA_ALPHA,
                device=str(constants.DEVICE)
            )
            adapter_paths.append(saved_path)
        except Exception as e:
            print(f"\n  ✗ Error training adapter {adapter_idx}: {e}")
            print(f"  Continuing with next adapter...")
            continue

    # Cleanup base model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Ensemble directory: {args.save_dir}")
    print(f"Successfully trained: {len(adapter_paths)}/{args.n_adapters} adapters")
    print("\nAdapter paths:")
    for i, path in enumerate(adapter_paths, 1):
        print(f"  {i}. {path}")
    print("\nNext step: Run compute_ensemble_entropy.py to evaluate on test prompts")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA ensemble (Colab optimized)")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save ensemble (default: saved_models/ensemble_lora_<timestamp>)"
    )
    parser.add_argument(
        "--n_adapters",
        type=int,
        default=5,
        help="Number of LoRA adapters to train in ensemble (default: 5)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed (each adapter uses base_seed + i)"
    )

    args = parser.parse_args()

    # Set default save_dir with timestamp if not provided
    if args.save_dir is None:
        timestamp = int(datetime.now().timestamp())
        args.save_dir = os.path.join("saved_models", f"ensemble_lora_{timestamp}")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
