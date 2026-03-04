# What Intermediate Layers Know: Jailbreak Patterns in Intermediate Entropy Traces

Detecting adversarial/harmful prompts in LLMs by probing entropy dynamics at intermediate transformer layers.

## TL;DR

> The key discriminative signal between safe and adversarial prompts is not the level of uncertainty, but **how uncertainty changes across token positions within intermediate layers**.

Static statistics (mean, median, quantiles) fail. The signal lives in *dynamics*: how entropy drifts or oscillates as tokens unfold. At **layer 22 (~70% depth) of Llama-3.1-8B**, `monotonicity_up` achieves AUROC **0.9995** (WildJailbreak vs AdvBench) — with no training, no threshold tuning, and no additional parameters.

See full results and methodology: [results/intermediate_entropy/results.md](results/intermediate_entropy/results.md)

---

## Project Structure

```
.
├── src/                              # Shared utilities
│   ├── constants.py                  # Configuration (model, training params)
│   ├── data_utils.py                 # Data loading (AdvBench, safe prompts)
│   └── uncertainty.py                # Shared uncertainty metrics
│
├── approaches/
│   ├── intermediate_layer/           # [MAIN] Intermediate layer entropy approach
│   │   └── compute_metrics_intermediate_layer.py
│   │
│   ├── prompt_entropy/               # Earlier approach: output-layer entropy traces
│   │   └── prompt_entropy.py
│   │
│   ├── laplace_approx/               # Earlier approach: Laplace approximation
│   │   ├── laplace.py
│   │   ├── training.py
│   │   ├── uncertainty.py
│   │   └── scripts/
│   │       ├── train_and_compute_fisher.py
│   │       ├── compute_entropy.py
│   │       └── run_interm_layer_entropy.py
│   │
│   └── ensemble_lora/                # Earlier approach: Ensemble LoRA
│       ├── train_ensemble.py
│       ├── inference.py
│       ├── uncertainty.py
│       └── scripts/
│           ├── compute_ensemble_entropy.py
│           └── README.md
│
├── results/
│   └── intermediate_entropy/         # [MAIN] Results for intermediate layer approach
│       ├── results.md                # Full methodology + AUROC tables
│       └── intermediate_entropy_all_keys_*.json
│
├── logs/
│   └── intermediate_layer/           # Per-model run logs (llama8b, gemma-7b, phi-4, qwen3-8b)
│
└── saved_models/                     # Trained model checkpoints (Laplace/Ensemble approaches)
```

---

## Main Approach: Intermediate Layer Entropy

### Key idea

Each prompt is passed through the base model under **teacher forcing**. At each token position, the model produces a probability distribution over the vocabulary — the **token-level predictive entropy** `H(t)` is computed from that distribution. Rather than reading from the final output layer only, hidden states are extracted at **8 probe layers** distributed across depth and projected through the model's own `final_norm + lm_head`.

This yields an **entropy trace** per layer per prompt. The trace is then aggregated into ~40 scalar features × 8 layers = **320 features total**, each evaluated as an unsupervised detector (AUROC, Average Precision, TPR@1%FPR) — no classifier trained, no labels used for threshold fitting.

**Probe layers (Llama-3.1-8B, 32 total):** `[0, 4, 8, 13, 17, 22, 26, 31]`

### Best metric: `L22_monotonicity_up`

The fraction of consecutive token pairs where entropy *increases* from one position to the next at layer 22:

```
monotonicity_up = (1 / (T-1)) · Σ_{t=1}^{T-1}  𝟙[H(t+1) > H(t)]
```

Layer 22 (~69% depth) consistently produces the strongest separation. This aligns with Kadali & Papalexakis (NeurIPS 2025 Workshop) who found the clearest jailbreak signal at ~70% depth in GPT-J using a completely different method — suggesting this may be architecture-general.

### Results summary

| Safe Dataset | Harmful Dataset | AUROC Flipped (L22_monotonicity_up) |
|---|---|---|
| WildJailbreak | AdvBench | **0.9995** |
| WildJailbreak | HarmBench | **0.9958** |
| WildJailbreak | StrongReject | **0.9371** |
| UltraChat | AdvBench | **0.9724** |
| UltraChat | HarmBench | **0.9562** |
| UltraChat | StrongReject | **0.7904** |

> AUROC Flipped > 0.5 means safe prompts score *higher* on this metric (threshold must be inverted). Full tables and discussion of the direction flip with JailbreakBench: [results/intermediate_entropy/results.md](results/intermediate_entropy/results.md)

### Run logs

Logs for all model/dataset combinations are in [logs/intermediate_layer/](logs/intermediate_layer/), organised by model:
- `llama8b/` — Llama-3.1-8B 
- `gemma-7b/`
- `phi-4/`
- `qwen3-8b/`

### Run the approach

```bash
python approaches/intermediate_layer/compute_metrics_intermediate_layer.py \
    --model meta-llama/Llama-3.1-8B \
    --safe_dataset wildjailbreak \
    --harmful_dataset advbench
```

---

## Earlier Approaches

These were developed before the intermediate layer direction and showed the limitations that motivated it.

### Prompt Entropy (output layer)

Entropy traces from the final output layer. Dynamic metrics (`slope`, `delta_seg`) reached AUROC ~0.80–0.83 on in-distribution data but **collapsed to 0.64 on XSTest+AdvBench** — out-of-distribution failure that motivated probing intermediate layers instead.

### Laplace Approximation

Fine-tunes LoRA adapters, computes diagonal Fisher information to approximate the posterior, then samples from the posterior to estimate predictive uncertainty. Strong in-distribution, but the fine-tuning calibration hurt generalisation.

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

**Metrics:** `mutual_information`, `mean_entropy`, `variance`, `intersection_probs_entropy`, `predictive_entropy`

### Ensemble LoRA

Trains multiple LoRA adapters with different seeds; uses ensemble disagreement as uncertainty. Same in-distribution/OOD pattern as Laplace.

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

See [approaches/ensemble_lora/scripts/README.md](approaches/ensemble_lora/scripts/README.md) for details.

---

## Configuration

Edit `src/constants.py` to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B` | Base model |
| `LORA_RANK` | 16 | LoRA rank (Laplace/Ensemble only) |
| `EPOCHS` | 5 | Training epochs (Laplace/Ensemble only) |
| `LEARNING_RATE` | 1e-2 | Learning rate (Laplace/Ensemble only) |
| `N_POSTERIOR_SAMPLES` | 50 | Samples for uncertainty (Laplace only) |
| `TEMPERATURE` | 0.01 | Posterior sampling temperature (Laplace only) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft datasets scipy matplotlib seaborn
```

## Data

- **Safe prompts:** WildJailbreak (benign), UltraChat, JailbreakBench (benign)
- **Harmful prompts:** [AdvBench](https://huggingface.co/datasets/walledai/AdvBench), HarmBench, StrongReject, ToxicChat

Data loading is handled by `src/data_utils.py`.
