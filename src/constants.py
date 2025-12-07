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
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Data parameters
N_SAFE_TRAIN = 100
N_BENIGN_TRAIN = 100
N_TEST_PER_CATEGORY = 50

# Training parameters
BATCH_SIZE = 2
EPOCHS = 1
MAX_LENGTH = 128
LEARNING_RATE = 3e-4

# LoRA parameters - CRITICAL FOR MEMORY
LORA_RANK = 4  # Keep small for memory
LORA_ALPHA = 8
LORA_DROPOUT = 0.1

# Bayesian parameters
LR_THRESHOLD = 1e-2  # Threshold for low-rank approximation
N_POSTERIOR_SAMPLES = 50  # Samples from predictive distribution
MAX_LAPLACE_BATCHES=20

# Directory to adapter checkpoints
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINE_TUNED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
