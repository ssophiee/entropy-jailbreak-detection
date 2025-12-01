#!/usr/bin/env python3
"""
Training script for fine-tuning LoRA model and computing Fisher information matrix.

This script:
1. Loads training data (safe + benign prompts)
2. Sets up model with LoRA adapters
3. Fine-tunes the model
4. Saves the fine-tuned model
5. Computes Laplace approximation (diagonal Fisher information matrix)
6. Saves Fisher matrix and LoRA parameters for later use

Usage:
    python train_and_compute_fisher.py
"""

import os
import torch
import argparse
from datetime import datetime

import src.constants as constants
from src.data_utils import load_training_and_test_data, create_dataloader
from src.training import setup_model_and_lora, train_lora
from src.laplace import collect_laplace_data, compute_diagonal_fisher


def main(args):
    print("="*80)
    print("TRAINING AND FISHER COMPUTATION SCRIPT")
    print("="*80)
    print(f"Device: {constants.DEVICE}")
    print(f"Model: {constants.MODEL_NAME}")
    print(f"LoRA Rank: {constants.LORA_RANK}")
    print(f"Epochs: {constants.EPOCHS}")
    print(f"Learning Rate: {constants.LEARNING_RATE}")
    print(f"Batch Size: {constants.BATCH_SIZE}")
    print(f"Max Length: {constants.MAX_LENGTH}")
    print(f"Max Laplace Batches: {constants.MAX_LAPLACE_BATCHES}")
    print("="*80)

    # 1. Load training data
    print("\n[1/6] Loading training data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )
    train_prompts = data['train_prompts']
    print(f"Loaded {len(train_prompts)} training prompts")

    # 2. Setup model and LoRA
    print("\n[2/6] Setting up model with LoRA...")
    model, tokenizer = setup_model_and_lora(
        constants.MODEL_NAME,
        constants.DEVICE,
        lora_rank=constants.LORA_RANK
    )

    # 3. Create DataLoader
    print("\n[3/6] Creating DataLoader...")
    train_loader = create_dataloader(
        train_prompts,
        tokenizer,
        max_length=constants.MAX_LENGTH,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    print(f"DataLoader created with {len(train_loader)} batches")

    # 4. Train the model
    print("\n[4/6] Training LoRA model...")
    model = train_lora(
        model,
        train_loader,
        epochs=constants.EPOCHS,
        lr=constants.LEARNING_RATE,
        device=constants.DEVICE,
        save_dir=args.save_dir,
        save_name=args.model_name
    )
    print("Training complete!")

    # 5. Collect data for Laplace approximation
    print("\n[5/6] Collecting data for Laplace approximation...")
    laplace_data = collect_laplace_data(
        model,
        train_loader,
        max_batches=constants.MAX_LAPLACE_BATCHES
    )

    # Clear CUDA cache before Fisher computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Compute diagonal Fisher matrix (Laplace approximation)
    print("\n[6/6] Computing diagonal Fisher information matrix...")
    fisher_diag, lora_params = compute_diagonal_fisher(
        model,
        laplace_data,
        device=constants.DEVICE
    )

    # Save Fisher matrix and model state
    save_path = os.path.join(args.save_dir, args.model_name or f"lora_finetuned_{int(datetime.now().timestamp())}")
    os.makedirs(save_path, exist_ok=True)

    fisher_save_path = os.path.join(save_path, "fisher_diag.pt")
    print(f"\nSaving Fisher diagonal to: {fisher_save_path}")
    torch.save({
        'fisher_diag': fisher_diag,
        'lora_param_names': list(lora_params.keys())
    }, fisher_save_path)

    print("\n" + "="*80)
    print("TRAINING AND FISHER COMPUTATION COMPLETE!")
    print("="*80)
    print(f"Model saved to: {save_path}")
    print(f"Fisher matrix saved to: {fisher_save_path}")
    print("\nNext step: Run compute_entropy.py to evaluate on test prompts")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA model and compute Fisher information")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory to save the model and Fisher matrix"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name for the saved model (default: timestamp-based name)"
    )

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
