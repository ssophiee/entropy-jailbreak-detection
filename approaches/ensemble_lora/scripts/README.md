# Ensemble LoRA Scripts

This directory contains scripts for training and evaluating LoRA ensembles for uncertainty quantification.

## Scripts

### 1. `train_ensemble.py`

Trains multiple LoRA adapters with different random seeds to create an ensemble.

**Usage:**
```bash
# Train ensemble with default settings (5 adapters)
python approaches/ensemble_lora/scripts/train_ensemble.py

# Train ensemble with custom number of adapters
python approaches/ensemble_lora/scripts/train_ensemble.py --n_adapters 10

# Specify custom save directory
python approaches/ensemble_lora/scripts/train_ensemble.py --save_dir saved_models/my_ensemble

# Specify custom base seed
python approaches/ensemble_lora/scripts/train_ensemble.py --base_seed 123
```

**Arguments:**
- `--save_dir`: Directory to save ensemble adapters (default: `saved_models/ensemble_lora_<timestamp>`)
- `--n_adapters`: Number of LoRA adapters to train (default: 5)
- `--base_seed`: Base random seed, each adapter uses `base_seed + i` (default: 42)

**Output:**
Creates a directory containing multiple adapter subdirectories:
```
saved_models/ensemble_lora_<timestamp>/
├── adapter_0_seed42/
├── adapter_1_seed43/
├── adapter_2_seed44/
└── ...
```

### 2. `compute_ensemble_entropy.py`

Computes uncertainty metrics for test prompts using a trained LoRA ensemble.

**Usage:**
```bash
# Compute uncertainty metrics
python approaches/ensemble_lora/scripts/compute_ensemble_entropy.py \
    --ensemble_dir saved_models/ensemble_lora_123456

# Specify custom output file
python approaches/ensemble_lora/scripts/compute_ensemble_entropy.py \
    --ensemble_dir saved_models/my_ensemble \
    --output results/my_results.json
```

**Arguments:**
- `--ensemble_dir`: Directory containing trained ensemble adapters (required)
- `--output`: Output JSON file path (default: `<ensemble_dir>/uncertainty_metrics.json`)

**Output:**
Creates a JSON file with uncertainty metrics:
```json
{
  "metadata": {
    "ensemble_dir": "...",
    "base_model": "...",
    "n_adapters": 5,
    "timestamp": "..."
  },
  "safe": {
    "predictive_entropy": [...],
    "mutual_information": [...],
    "variance": [...],
    "diversity": [...],
    "mean_confidence": [...],
    "prompts": [...]
  },
  "benign": {
    "predictive_entropy": [...],
    ...
  }
}
```

## Workflow

1. **Train ensemble:**
   ```bash
   python approaches/ensemble_lora/scripts/train_ensemble.py \
       --n_adapters 5 \
       --save_dir saved_models/my_ensemble
   ```

2. **Compute uncertainty:**
   ```bash
   python approaches/ensemble_lora/scripts/compute_ensemble_entropy.py \
       --ensemble_dir saved_models/my_ensemble
   ```

3. **Analyze results:**
   ```python
   import json
   with open("saved_models/my_ensemble/uncertainty_metrics.json") as f:
       results = json.load(f)

   # Compare safe vs benign uncertainty
   import numpy as np
   safe_entropy = np.array(results["safe"]["predictive_entropy"])
   benign_entropy = np.array(results["benign"]["predictive_entropy"])

   print(f"Safe entropy: {safe_entropy.mean():.4f} ± {safe_entropy.std():.4f}")
   print(f"Benign entropy: {benign_entropy.mean():.4f} ± {benign_entropy.std():.4f}")
   ```

## Uncertainty Metrics

The ensemble provides multiple uncertainty metrics:

- **Predictive Entropy**: H[E[p(y|x)]] - Total uncertainty
- **Mutual Information**: H[E[p(y|x)]] - E[H[p(y|x)]] - Epistemic uncertainty (model disagreement)
- **Variance**: Variance of predicted class probabilities across ensemble
- **Diversity**: Fraction of ensemble members that disagree on predicted class
- **Mean Confidence**: Average confidence of ensemble prediction

## Configuration

Model and training parameters are controlled via [src/constants.py](../../../src/constants.py):

- `MODEL_NAME`: Base model to use
- `LORA_RANK`, `LORA_ALPHA`: LoRA hyperparameters
- `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`: Training settings
- `N_SAFE_TRAIN`, `N_BENIGN_TRAIN`: Training data sizes
- `N_TEST_PER_CATEGORY`: Test data size per category
