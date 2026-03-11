# What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics

## Abstract

Jailbreak attacks reveal a persistent weakness in aligned large language models (LLMs): carefully crafted prompts can elicit policy-violating responses despite safety training. We investigate how jailbreak-relevant signal is encoded in the internal dynamics of a base LLM by analyzing token-level predictive entropy trajectories across model layers. We find that static aggregate statistics of prompt-level entropy (e.g., mean, variance) carry little discriminative signal, whereas features capturing how entropy evolves across token positions, such as monotonic trend scores, are substantially more informative. Furthermore, this signal is not uniform across model depth: intermediate layers exhibit stronger and more consistent distributional differences between jailbreak and benign inputs than the final layer, suggesting that jailbreak-relevant structure is encoded in mid-network representations. Together, these findings characterize which entropy-derived features reflect harmful intent and where in the network that signal is most pronounced.

---

## Structure

```
.
├── approaches/
│   └── intermediate_layer/
│       └── compute_metrics_intermediate_layer.py
│
├── src/
│   ├── constants.py              # Configuration
│   └── data_utils.py             # Data loading
│
├── scripts/
│   └── run_all_datasets.sh       # Full evaluation sweep
│
├── logs/intermediate_layer/      # Per-model run logs
└── results/intermediate_entropy/ # AUROC tables and raw results
```

---

## Usage

**Setup:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Single run:**
```bash
python -m approaches.intermediate_layer.compute_metrics_intermediate_layer \
    --model_path meta-llama/Llama-3.1-8B \
    --safe_dataset wildjailbreak_benign \
    --harmful_dataset advbench \
    --n_test max --balance --run_all
```

Possible dataset combinations (9 total):
- **Harmful:** `advbench`, `strongreject`, `harmbench`
- **Safe:** `ultrachat`, `jailbreakbench_benign`, `wildjailbreak_benign`

Datasets are loaded automatically via `src/data_utils.py` (HuggingFace).
