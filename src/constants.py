import os
import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)

MAX_LENGTH = 512
MODEL_NAME = "meta-llama/Llama-3.1-8B"

N_SAFE_TRAIN = 200
N_BENIGN_TRAIN = 200
N_TEST_PER_CATEGORY = 9999
