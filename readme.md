# Adversarial Prompt Detection using Bayesian Uncertainty

Detecting adversarial/harmful prompts in LLMs using Bayesian uncertainty quantification. The hypothesis: adversarial prompts that the model hasn't seen during safety training will produce higher uncertainty.

## Project Structure

```
.
├── src/                          # Shared utilities
│   ├── constants.py              # Configuration (model, training params)
│   ├── data_utils.py             # Data loading (AdvBench, safe prompts)
│   └── uncertainty.py            # Shared uncertainty metrics
│
├── approaches/
│   ├── laplace_approx/           # Laplace Approximation approach
│   │   ├── laplace.py            # Fisher information computation
│   │   ├── training.py           # LoRA training utilities
│   │   ├── uncertainty.py        # Laplace-specific inference (posterior sampling)
│   │   └── scripts/
│   │       ├── train_and_compute_fisher.py
│   │       └── compute_entropy.py
│   │
│   └── ensemble_lora/            # Ensemble LoRA approach
│       ├── train_ensemble.py     # Multi-adapter training
│       ├── inference.py          # Ensemble inference
│       ├── uncertainty.py        # Ensemble-specific inference
│       └── scripts/
│           ├── compute_ensemble_entropy.py
│           └── README.md
│
└── saved_models/                 # Trained model checkpoints
```

## Approaches

### 1. Laplace Approximation

Uses diagonal Fisher information matrix to approximate the posterior over LoRA weights, then samples from this posterior to compute predictive uncertainty.

**Train model and compute Fisher matrix:**
```bash
python approaches/laplace_approx/scripts/train_and_compute_fisher.py \
    --save_dir saved_models \
    --model_name my_laplace_model
```

**Compute uncertainty on test prompts:**
```bash
python approaches/laplace_approx/scripts/compute_entropy.py \
    --model_path saved_models/my_laplace_model \
    --metric intersection_probs_entropy \
    --temperature 0.05 \
    --visualize
```

**Metrics available:** `mutual_information`, `mean_entropy`, `variance`, `intersection_probs_entropy`, `predictive_entropy`

### 2. Ensemble LoRA

Trains multiple LoRA adapters with different random seeds and uses ensemble disagreement as uncertainty measure.

**Train ensemble:**
```bash
python approaches/ensemble_lora/scripts/train_ensemble.py \
    --n_adapters 5 \
    --save_dir saved_models/my_ensemble
```

**Compute uncertainty:**
```bash
python approaches/ensemble_lora/scripts/compute_ensemble_entropy.py \
    --ensemble_dir saved_models/my_ensemble \
    --visualize
```

**Metrics computed:** Same as Laplace approach - `mutual_information`, `mean_entropy`, `variance`, `intersection_probs_entropy`, `predictive_entropy`

See [approaches/ensemble_lora/scripts/README.md](approaches/ensemble_lora/scripts/README.md) for detailed documentation.

## Configuration

Edit `src/constants.py` to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `LORA_RANK` | 16 | LoRA rank |
| `EPOCHS` | 5 | Training epochs |
| `LEARNING_RATE` | 1e-2 | Learning rate |
| `N_POSTERIOR_SAMPLES` | 50 | Samples for uncertainty estimation |
| `TEMPERATURE` | 0.01 | Posterior sampling temperature |

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers peft datasets scipy matplotlib seaborn

# Configure device (optional)
echo "DEVICE=cuda" > .env
```

## Data

The project uses:
- **Safe prompts**: Benign, harmless prompts for training
- **Adversarial prompts**: From [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) for testing

Data loading is handled by `src/data_utils.py`.
