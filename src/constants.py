import os
import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)

# MAX_LENGTH for tokenization
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

MAX_LAPLACE_BATCHES = int(os.getenv("MAX_LAPLACE_BATCHES", "40"))

# Model parameters
MODEL_NAME = "meta-llama/Llama-3.1-8B" # "Qwen/Qwen2.5-3B-Instruct"

# Data parameters
N_SAFE_TRAIN = 200  # Doubled from 100 for better Fisher information
N_BENIGN_TRAIN = 200  # Doubled from 100 for better Fisher information
N_TEST_PER_CATEGORY = 9999

# Training parameters
BATCH_SIZE = 4  # Micro-batch size (effective batch = BATCH_SIZE * gradient_accumulation_steps)
EPOCHS = 5  # Increased from 3 for better convergence
MAX_LENGTH = 512
LEARNING_RATE = 2e-4 # 1e-2 # 1e-2  # Increased from 3e-4 for better Fisher information

# LoRA parameters - CRITICAL FOR MEMORY
LORA_RANK = 16  # Increased from 4 to 16 for better capacity (was too small!)
LORA_ALPHA = 32 #32  # Scale with rank (typically 2x rank)
LORA_DROPOUT = 0.1

# Bayesian parameters
LR_THRESHOLD = 1e-2  # Threshold for low-rank approximation
N_POSTERIOR_SAMPLES = 50  # Samples from predictive distribution
MAX_LAPLACE_BATCHES = 100  # Increased from 50 to use all training data (200 prompts with batch_size=2)
TEMPERATURE = 0.01  # Temperature for posterior sampling (use with sum-based Fisher)
PRIOR_PRECISION = 1 # 1e-4  # Prior precision for Laplace approximation (regularization for low-Fisher parameters)

# Directory to adapter checkpoints
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
