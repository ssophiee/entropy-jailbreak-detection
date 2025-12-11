#!/usr/bin/env python3
"""
Training script for LoRA ensemble.

This script:
1. Loads training data (safe + benign prompts)
2. Trains multiple LoRA adapters with different random seeds
3. Saves all adapters for later ensemble inference

Usage:
    python train_ensemble.py
    python train_ensemble.py --n_adapters 10 --save_dir saved_models/my_ensemble
"""
import os
import sys
import argparse
from datetime import datetime
from transformers import AutoTokenizer

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data, create_dataloader
from approaches.ensemble_lora.train_ensemble import train_lora_ensemble


def main(args):
    print("=" * 80)
    print("ENSEMBLE LORA TRAINING SCRIPT")
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
    print("\n[1/3] Loading training data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )
    train_prompts = data['train_prompts']
    print(f"Loaded {len(train_prompts)} training prompts")

    # 2. Get tokenizer (need to load model to get it)
    print("\n[2/3] Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Create DataLoader
    print("\n[3/3] Creating DataLoader...")
    train_loader = create_dataloader(
        train_prompts,
        tokenizer,
        max_length=constants.MAX_LENGTH,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    print(f"DataLoader created with {len(train_loader)} batches")

    # 4. Train ensemble
    print("\n[4/4] Training ensemble of LoRA adapters...")
    adapter_paths = train_lora_ensemble(
        base_model_name=constants.MODEL_NAME,
        train_loader=train_loader,
        n_adapters=args.n_adapters,
        epochs=constants.EPOCHS,
        lr=constants.LEARNING_RATE,
        lora_rank=constants.LORA_RANK,
        lora_alpha=constants.LORA_ALPHA,
        lora_dropout=0.1,
        device=constants.DEVICE,
        save_dir=args.save_dir,
        base_seed=args.base_seed
    )

    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Ensemble directory: {args.save_dir}")
    print(f"Number of adapters: {len(adapter_paths)}")
    print("\nAdapter paths:")
    for i, path in enumerate(adapter_paths, 1):
        print(f"  {i}. {path}")
    print("\nNext step: Run compute_ensemble_entropy.py to evaluate on test prompts")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA ensemble")
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
