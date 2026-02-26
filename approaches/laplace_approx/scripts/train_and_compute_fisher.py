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
import os, sys
import logging
import torch
import argparse
from datetime import datetime

from huggingface_hub import login

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)

this_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(this_dir, ".."))
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, parent_dir)


import src.constants as constants
from src.data_utils import load_training_and_test_data, create_dataloader
from training import setup_model_and_lora, train_lora
from laplace import collect_laplace_data, compute_diagonal_fisher

log = logging.getLogger(__name__)

def main(args):
    log.info("=" * 80)
    log.info("TRAINING AND FISHER COMPUTATION SCRIPT")
    log.info("=" * 80)
    log.info("Device:             %s", constants.DEVICE)
    log.info("Model:              %s", constants.MODEL_NAME)
    log.info("LoRA Rank:          %s", constants.LORA_RANK)
    log.info("Epochs:             %s", constants.EPOCHS)
    log.info("Learning Rate:      %s", constants.LEARNING_RATE)
    log.info("Batch Size:         %s", constants.BATCH_SIZE)
    log.info("Max Length:          %s", constants.MAX_LENGTH)
    log.info("Max Laplace Batches:%s", constants.MAX_LAPLACE_BATCHES)
    log.info("=" * 80)

    # 1. Load training data
    log.info("[1/6] Loading training data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY
    )
    train_prompts = data['train_prompts']
    log.info("Loaded %d training prompts", len(train_prompts))


    log.info("[2/6] Setting up model with LoRA...")
    model, tokenizer = setup_model_and_lora(
        constants.MODEL_NAME,
        constants.DEVICE,
        lora_rank=constants.LORA_RANK
    )

    log.info("[3/6] Creating DataLoader...")
    train_loader = create_dataloader(
        train_prompts,
        tokenizer,
        max_length=constants.MAX_LENGTH,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    log.info("DataLoader created with %d batches", len(train_loader))

    log.info("[4/6] Training LoRA model...")
    model = train_lora(
        model,
        train_loader,
        epochs=constants.EPOCHS,
        lr=constants.LEARNING_RATE,
        device=constants.DEVICE,
        save_dir=args.save_dir,
        save_name=args.model_name
    )
    log.info("Training complete!")


    log.info("[5/6] Collecting data for Laplace approximation...")
    laplace_data = collect_laplace_data(
        model,
        train_loader,
        max_batches=constants.MAX_LAPLACE_BATCHES
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("[6/6] Computing diagonal Fisher information matrix...")
    fisher_diag, lora_params = compute_diagonal_fisher(
        model,
        laplace_data,
        device=constants.DEVICE
    )

    # Save Fisher matrix and model state
    save_path = os.path.join(args.save_dir, args.model_name or f"lora_finetuned_{int(datetime.now().timestamp())}")
    os.makedirs(save_path, exist_ok=True)

    fisher_save_path = os.path.join(save_path, "fisher_diag.pt")
    log.info("Saving Fisher diagonal to: %s", fisher_save_path)
    torch.save({
        'fisher_diag': fisher_diag,
        'lora_param_names': list(lora_params.keys())
    }, fisher_save_path)

    log.info("=" * 80)
    log.info("TRAINING AND FISHER COMPUTATION COMPLETE!")
    log.info("=" * 80)
    log.info("Model saved to: %s", save_path)
    log.info("Fisher matrix saved to: %s", fisher_save_path)
    log.info("Next step: Run compute_entropy.py to evaluate on test prompts")
    log.info("=" * 80)


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
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Path to log file (default: <save_dir>/train.log).",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = args.log_file or "logs/train_laplace.log"
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    main(args)
