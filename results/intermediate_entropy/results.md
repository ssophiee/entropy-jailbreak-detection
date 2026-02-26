# L22 Metrics — AUROC vs AUROC Flipped

**Model:** `meta-llama/Llama-3.1-8B` - No fine-tuning

---

## TL;DR

> **The key discriminative signal between safe and adversarial prompts is not the level of uncertainty, but how uncertainty changes across token positions within intermediate layers.**

Static statistics — `mean`, `median`, quantiles — consistently fail to separate harmful from safe prompts. The signal lives in *dynamics*: metrics that capture how entropy drifts, oscillates, or accelerates across the prompt. This mirrors the finding at the final output layer where `slope` (AUROC ≈ 0.80) and `delta_seg` (AUROC ≈ 0.83) substantially outperform central-tendency measures, with the interpretation that adversarial prompts induce a progressive, structured change in model confidence as tokens unfold — not uniformly high or low uncertainty.

**However, this dynamic signal did not hold reliably at the final layer.** The Laplace approximation approach (entropy traces from the fine-tuned output layer) collapsed out-of-distribution — XSTest+AdvBench dropped to AUROC 0.64 despite strong in-distribution performance. The dynamic signature generalises only when read from **intermediate layers of the base model**, where no fine-tuning calibration is required. At layer 22 (~70% depth), `monotonicity_up` achieves AUROC 0.91–1.00 across WildJailbreak and UltraChat safe baselines — without any trained classifier or threshold tuning.

---

## Algorithm

### Overview

Each prompt is passed through the model under **teacher forcing** — tokens are fed one at a time and at each position the model produces a probability distribution over the vocabulary. The **token-level predictive entropy** is computed at that position:

```
H(t) = -Σ_v  p(v|x_1,...,x_t) · log p(v|x_1,...,x_t)
```

This produces an **entropy trace** — a sequence of scalar uncertainty values `[H(1), H(2), ..., H(T)]` indexed by token position.

Rather than reading entropy only from the final output layer, hidden states are extracted at **8 probe layers** distributed across the model depth:

```
layers_probed = [0, 4, 8, 13, 17, 22, 26, 31]   (out of 32 total)
```

Each probe layer yields its own independent entropy trace. The traces are then aggregated into scalar detection features, producing ~40 features × 8 layers = **320 features total**. Each feature is evaluated independently as an unsupervised detection score — no classifier is trained, no threshold is tuned on labelled data.

### Step-by-step

1. **Tokenise** the prompt.
2. **Forward pass** with `output_hidden_states=True`; collect hidden states at each probe layer at every token position.
3. **Project** each intermediate hidden state through the model's own `final_norm + lm_head` to obtain a vocabulary distribution — this reuses the model's own output projection without any additional parameters.
4. **Compute entropy** `H(t)` from that projected distribution at each token position.
5. **Aggregate** the entropy trace into scalar features (see below).
6. **Evaluate** each scalar as a ranking score: compute AUROC, Average Precision, and TPR at 1% FPR against a labelled safe/harmful split.

### Implementation note

The entropy at layer `l` and token position `t` is computed as:

```python
h = hidden_states[layer_idx + 1].squeeze(0)[:-1, :]  # [T-1, d_model]
h_normed = final_norm(h)                              # [T-1, d_model]
logits   = lm_head(h_normed).float()                 # [T-1, vocab_size]
H[i]     = entropy_from_logits(logits)               # [T-1]
```

Key details:
- `hidden_states[layer_idx + 1]` — index offset by 1 because index 0 is the embedding layer; index `k` is the output of transformer layer `k-1`.
- `[:-1, :]` — only positions `0` to `T-2` are used. This is **teacher forcing**: position `t` predicts token `t+1`, so there are `T-1` valid prediction steps for a prompt of `T` tokens.
- The same `final_norm` and `lm_head` from the model head are reused — the intermediate hidden state is projected into vocabulary space using the model's own machinery, so no additional parameters are introduced.

---

## Metrics

### `L22_monotonicity_up`

**Layer:** 22 (~69% of model depth)

**What it measures:** The fraction of consecutive token pairs where entropy *increases* from one position to the next.

**Formula:**

```
monotonicity_up = (1 / (T-1)) · Σ_{t=1}^{T-1}  𝟙[H(t+1) > H(t)]
```

where `T` is the number of tokens and `𝟙[·]` is the indicator function.

**Range:** 0 to 1.
- `1.0` → entropy strictly increased at every consecutive step
- `0.5` → roughly random up/down fluctuation
- `0.0` → entropy was almost always decreasing

**Concrete example.** Suppose a prompt tokenises to 6 tokens, giving a 5-step entropy trace at layer 22:

```
t:    1     2     3     4     5
H:  2.31  2.45  2.38  2.61  2.70
         ↑     ↓     ↑     ↑
```

Steps where `H(t+1) > H(t)`: steps 1→2 ✓, 2→3 ✗, 3→4 ✓, 4→5 ✓ → 3 out of 4.

```
monotonicity_up = 3/4 = 0.75
```

A safe prompt might score 0.75; a harmful prompt tends to score lower (e.g. 0.40) when WildJailbreak/UltraChat is the safe baseline — meaning the model's uncertainty at this layer is less consistently rising for harmful inputs.

**Interpretation in results:** For WildJailbreak and UltraChat as safe baselines, harmful prompts score *lower* — AUROC Flipped is high because safe prompts have more monotonically rising entropy at L22. For JailbreakBench as safe baseline, the direction reverses entirely.

---

## Why Layer 22?

Layer 22 sits at approximately **69% of model depth** in Llama-3.1-8B (32 layers total). Empirically this depth consistently produces the strongest separation across dataset combinations. This is consistent with findings in Kadali & Papalexakis (NeurIPS 2025 Workshop — https://arxiv.org/abs/2510.06594) who also found the clearest jailbreak separation at layer 22 of GPT-J using tensor decomposition — a different method on a different model, suggesting the ~70% depth signal may be architecture-general rather than Llama-specific.

---

## Results

**Interpretation:**
- **AUROC** > 0.5 → harmful prompts score *higher* on the metric (forward direction, no threshold inversion needed).
- **AUROC Flipped** > 0.5 → safe prompts score *higher* (threshold must be inverted to use as a detector).
- The two tables are mutually exclusive views of the same ranking — `auroc_flipped = 1 − auroc`. They are never combined or averaged.
- Detectability threshold used: ≥ 0.75.

### AUROC Flipped — safe prompts score higher (invert threshold)

| Safe Dataset | Harmful Dataset | L22_monotonicity_up |
|---|---|---|
| WildJailbreak | AdvBench | **0.9995** |
| WildJailbreak | HarmBench | **0.9958** |
| WildJailbreak | StrongReject | **0.9371** |
| UltraChat | AdvBench | **0.9724** |
| UltraChat | HarmBench | **0.9562** |
| UltraChat | StrongReject | **0.7904** |
| JailbreakBench | AdvBench | 0.5648 |
| JailbreakBench | HarmBench | 0.4198 |
| JailbreakBench | StrongReject | 0.0908 |

### AUROC (raw) — harmful scores higher

| Safe Dataset | Harmful Dataset | L22_monotonicity_up |
|---|---|---|
| WildJailbreak | AdvBench | 0.0005 |
| WildJailbreak | HarmBench | 0.0042 |
| WildJailbreak | StrongReject | 0.0629 |
| UltraChat | AdvBench | 0.0276 |
| UltraChat | HarmBench | 0.0438 |
| UltraChat | StrongReject | 0.2096 |
| JailbreakBench | AdvBench | 0.4352 |
| JailbreakBench | HarmBench | 0.5802 |
| JailbreakBench | StrongReject | **0.9092** |

---

## Key Observations

- **WildJailbreak + UltraChat** as safe baseline: signal lives entirely in AUROC Flipped — safe prompts have more monotonically rising entropy at L22 than harmful ones. Detectable against AdvBench, HarmBench, and StrongReject. Fails against ToxicChat.
- **JailbreakBench** as safe baseline: signal flips to raw AUROC — harmful prompts score *higher*. Detectable against StrongReject and ToxicChat. Fails against AdvBench and HarmBench.
- The direction flip between safe datasets means a **fixed threshold is not portable** across safe baselines without knowing which direction applies — a key limitation for deployment.