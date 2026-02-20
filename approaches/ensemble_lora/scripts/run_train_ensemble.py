#!/usr/bin/env python3
"""
Train LoRA ensemble on any supported model.

Usage:
    python approaches/ensemble_lora/scripts/run_train_ensemble.py
    python approaches/ensemble_lora/scripts/run_train_ensemble.py --model meta-llama/Llama-3.1-8B
    python approaches/ensemble_lora/scripts/run_train_ensemble.py --model Qwen/Qwen2.5-3B-Instruct --n_adapters 3
"""
import os
import sys
import logging
import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer

# Add repo root to path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
sys.path.insert(0, repo_root)

import src.constants as constants
from src.data_utils import load_training_and_test_data, create_dataloader
from approaches.ensemble_lora.train_ensemble import train_lora_ensemble

log = logging.getLogger(__name__)


def main(args):
    log.info("=" * 70)
    log.info("ENSEMBLE LORA TRAINING")
    log.info("=" * 70)
    log.info("Device:       %s", constants.DEVICE)
    log.info("Model:        %s", args.model)
    log.info("Adapters:     %s", args.n_adapters)
    log.info("Epochs:       %s", args.epochs)
    log.info("LR:           %s", args.lr)
    log.info("LoRA rank:    %s", args.lora_rank)
    log.info("LoRA alpha:   %s", args.lora_alpha)
    log.info("Batch size:   %s", args.batch_size)
    log.info("Base seed:    %s", args.base_seed)
    log.info("Save dir:     %s", args.save_dir)
    log.info("=" * 70)

    # 1. Load data
    log.info("Loading training data...")
    data = load_training_and_test_data(
        n_safe_train=constants.N_SAFE_TRAIN,
        n_benign_train=constants.N_BENIGN_TRAIN,
        n_test_per_category=constants.N_TEST_PER_CATEGORY,
    )
    train_prompts = data["train_prompts"]
    log.info("Loaded %d training prompts", len(train_prompts))

    # 2. Tokenizer + DataLoader
    log.info("Setting up tokenizer and dataloader...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = create_dataloader(
        train_prompts, tokenizer,
        max_length=constants.MAX_LENGTH,
        batch_size=args.batch_size,
        shuffle=True,
    )
    log.info("DataLoader: %d batches", len(train_loader))

    # 3. Train ensemble
    log.info("Training ensemble...")
    adapter_paths = train_lora_ensemble(
        base_model_name=args.model,
        train_loader=train_loader,
        n_adapters=args.n_adapters,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=constants.LORA_DROPOUT,
        device=str(constants.DEVICE),
        save_dir=args.save_dir,
        base_seed=args.base_seed,
    )

    log.info("Done. %d adapters saved to: %s", len(adapter_paths), args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA ensemble")
    parser.add_argument(
        "--model", type=str, default=constants.MODEL_NAME,
        help=f"HuggingFace model name (default: {constants.MODEL_NAME})"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save ensemble (default: saved_models/ensemble/ensemble_lora_<model>_<timestamp>)"
    )
    parser.add_argument("--n_adapters", type=int, default=5, help="Number of LoRA adapters (default: 5)")
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS, help=f"Training epochs (default: {constants.EPOCHS})")
    parser.add_argument("--lr", type=float, default=constants.LEARNING_RATE, help=f"Learning rate (default: {constants.LEARNING_RATE})")
    parser.add_argument("--lora_rank", type=int, default=constants.LORA_RANK, help=f"LoRA rank (default: {constants.LORA_RANK})")
    parser.add_argument("--lora_alpha", type=int, default=constants.LORA_ALPHA, help=f"LoRA alpha (default: {constants.LORA_ALPHA})")
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE, help=f"Batch size (default: {constants.BATCH_SIZE})")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Path to log file (default: <save_dir>/train_ensemble.log).",
    )

    args = parser.parse_args()

    if args.save_dir is None:
        model_short = args.model.split("/")[-1]
        timestamp = int(datetime.now().timestamp())
        args.save_dir = os.path.join("saved_models/ensemble", f"ensemble_lora_{model_short}_{timestamp}")

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = args.log_file or "logs/train_ensemble.log"
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
