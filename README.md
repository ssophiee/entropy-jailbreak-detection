# What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics

This repository contains code for the **ECML PKDD 2026** paper: *What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics*.


## 📄 Abstract

Jailbreak attacks reveal a persistent weakness in aligned Large Language Models: carefully crafted prompts can elicit policy-violating responses despite safety training. While most defenses operate at the prompt or output level, it remains unclear how harmful intent is encoded within the model's internal representations. We investigate this question by analyzing token-level predictive entropy trajectories across layers of a frozen LLM using the logit lens. We find that static aggregate statistics of prompt-level entropy (e.g., mean, variance) carry little discriminative signal, whereas features capturing how entropy evolves across token positions, such as monotonic rank-based trend scores, are substantially more informative. Importantly, this signal is not uniform across model depth: it consistently peaks at intermediate layers and degrades at the final layer, indicating that jailbreak-relevant structure emerges in mid-network semantic representations rather than at the output head. Across multiple models (Llama, Qwen, Gemma) and adversarial benchmarks, these entropy dynamics provide architecture-consistent separation without additional training. Together, our findings show that jailbreak behavior is reflected in structured intermediate uncertainty dynamics, clarifying both which entropy-derived features encode harmful intent and where in the network that signal is most pronounced.

**Keywords:** Jailbreak Detection, Large Language Models, Token-level Uncertainty Approximation


## 🚀 Getting Started

### 1️⃣ Environment Setup

Create a virtual environment and install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU support requires a CUDA-compatible PyTorch installation. See the [PyTorch install guide](https://pytorch.org/get-started/locally/) for your platform.

### 2️⃣ Configuration

Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```

Edit `.env` to set your device and Hugging Face token:
```
DEVICE="cuda"
HUGGINGFACE_TOKEN="your_huggingface_token_here"
```
A Hugging Face token is required for gated datasets (e.g., AdvBench, HarmBench). You can create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3️⃣ Running Experiments

Run the intermediate-layer entropy analysis for a single dataset pair:
```bash
python -m approaches.intermediate_layer.compute_metrics_intermediate_layer \
    --model_path meta-llama/Llama-3.1-8B \
    --safe_dataset ultrachat \
    --harmful_dataset advbench \
    --n_test max --balance --run_all
```

To reproduce results across all models evaluated in the paper, substitute `--model_path` with:
- `meta-llama/Llama-3.1-8B` (Llama 3, 32 layers)
- `Qwen/Qwen3-8B` (Qwen 3, 36 layers)
- `google/gemma-7b` (Gemma, 28 layers)

#### Dataset Combinations

The paper evaluates 9 dataset pairs (3 harmful × 3 safe):

| | `advbench` | `harmbench` | `strongreject` |
|---|---|---|---|
| **`ultrachat`** | Primary | Primary | Primary |
| **`wildjailbreak_benign`** | Primary | Primary | Primary |
| **`jailbreakbench_benign`** | Adversarial control | Adversarial control | Adversarial control |

All datasets are loaded automatically from Hugging Face via `src/data_utils.py`.

### 4️⃣ Results

Results are saved as JSON files to `results/intermediate_entropy/`. Each file contains:
- Per-prompt feature vectors (static and dynamic features at each probe layer)
- Detection metrics (AUROC, AP, TPR@FPR, ECE, Brier) for every feature

When `--run_all` is used, a sorted summary table is printed to stdout showing the best-performing features.


## 📂 Repository Structure

```
.
├── approaches/
│   ├── intermediate_layer/
│   │   └── compute_metrics_intermediate_layer.py   # Main experiment script
│   └── prompt_entropy/
│       └── prompt_entropy.py                       # Entropy computation, logit lens, feature extraction
├── src/
│   ├── constants.py                                # Device, model, and data configuration
│   └── data_utils.py                               # Dataset registry and data loading (HuggingFace)
├── supplementary_material.pdf                      # Supplementary material for the paper
├── requirements.txt                                # Python dependencies
├── .env.example                                    # Environment variable template
└── readme.md                                       # This file
```


## 📎 Supplementary Material

Extended results, layer-wise AUROC tables, additional ablations, and an analysis of JailbreakBench-benign distributional overlap are available in [`supplementary_material.pdf`](supplementary_material.pdf), included in this repository.

<!--
## 📢 Citation

If you use this code, or found it inspiring, please cite our paper:

```bibtex
@inproceedings{nikolenko2026intermediate,
  title={What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics},
  author={Nikolenko, Sofiia and Papucci, Michele and Rezaei, Mina and Manchingal, Shireen Kudukkil},
  booktitle={Joint European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year={2026}
}
```
-->

## 📬 Contact

For questions or issues, feel free to open an issue or contact [Shireen Kudukkil Manchingal](mailto:smanchingal@brookes.ac.uk).


## ⭐ Acknowledgments

MR was supported by the Amazon Research Award 2024. SN was supported by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Research, Technology and Space. SKM has received funding from the European Union's Horizon 2020 Research and Innovation program under Grant Agreement No. 964505 (E-pi).
