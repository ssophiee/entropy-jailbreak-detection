# What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics

## Abstract

Jailbreak attacks reveal a persistent weakness in aligned Large Language Models: carefully crafted prompts can elicit policy-violating responses despite safety training. While most defenses operate at the prompt or output level, it remains unclear how harmful intent is encoded within the model’s internal representations. We investigate this question by analyzing token-level predictive entropy trajectories across layers of a frozen LLM using the logit lens. We find that static aggregate statistics of prompt-level entropy (e.g., mean, variance) carry little discriminative signal, whereas features capturing how entropy evolves across token positions, such as monotonic rank-based trend scores, are substantially more informative. Importantly, this signal is not uniform across model depth: it consistently peaks at intermediate layers and degrades at the final layer, indicating that jailbreak-relevant structure emerges in mid-network semantic representations rather than at the output head. Across multiple models (Llama, Qwen, Gemma) and adversarial benchmarks, these entropy dynamics provide architecture-consistent separation without additional training. Together, our findings show that jailbreak behavior is reflected in structured intermediate uncertainty dynamics, clarifying both which entropy-derived features encode harmful intent and where in the network that signal is most pronounced.

---

## Structure

```
.
├── approaches/
│   ├── intermediate_layer/
│   │   └── compute_metrics_intermediate_layer.py
│   └── prompt_entropy/
│       └── prompt_entropy.py
│
├── src/
│   ├── constants.py              # Configuration
│   └── data_utils.py             # Data loading
│
├── requirements.txt
└── .env.example
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
