# approaches/prompt_entropy/prompt_entropy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PromptEntropyResult:
    prompt: str
    n_tokens: int
    entropies: List[float]               # per-position next-token entropy over prompt
    features: Dict[str, float]           # aggregated features


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy H(p) where p = softmax(logits).
    logits: [..., V]
    returns: [...]
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)
    return H

def _kendall_tau_with_position(x: np.ndarray) -> float:
    """
    Kendall's tau-b between token position t=[0..n-1] and entropy values x.
    Uses all C(n,2) pairs (i,j) with j>i: concordant if x[j]>x[i], discordant if x[j]<x[i].
    More robust than Spearman: O(n log n) scipy implementation, tie-corrected.
    References: Kendall (1938) Biometrika 30(1-2):81-93; Mann (1945) Econometrica 13(3):245-259.
    """
    from scipy.stats import kendalltau
    n = x.size
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)  # token position indices
    return float(kendalltau(t, x).statistic)


def _spearman_rho_with_position(x: np.ndarray) -> float:
    """
    Spearman rank correlation between position t and values x.
    We implement ranks using argsort twice (ties unlikely for float entropies).
    """
    n = x.size
    if n < 2:
        return 0.0

    # ranks of x (0..n-1)
    rx = np.empty(n, dtype=np.float64)
    rx[np.argsort(x)] = np.arange(n, dtype=np.float64)

    # ranks of t are just t itself (already increasing), but keep symmetric
    rt = np.arange(n, dtype=np.float64)

    # Pearson correlation of ranks
    rx = rx - rx.mean()
    rt = rt - rt.mean()
    denom = (np.sqrt((rx * rx).mean()) * np.sqrt((rt * rt).mean())) + 1e-12
    return float((rx * rt).mean() / denom)

def aggregate_entropy_features(
    entropies: List[float],
    *,
    trim_ratio: float = 0.10,
    first_frac: float = 0.25,
    last_frac: float = 0.25,
) -> Dict[str, float]:
    """
    Turn a per-token entropy trace into scalar features for classification.
    Includes:
      - level (mean/median/quantiles)
      - volatility (std/ac1)
      - trend (slope, delta_seg, spearman_rho, kendall_tau, monotonicity_up, acceleration)
      - structure (peak_density, thirds)
    """
    empty = {
        "mean": 0.0, "std": 0.0, "max": 0.0, "median": 0.0,
        "trimmed_mean": 0.0, "first_mean": 0.0, "last_mean": 0.0,
        "slope": 0.0, "delta_end": 0.0, "delta_seg": 0.0,
        "spearman_rho": 0.0, "kendall_tau": 0.0, "monotonicity_up": 0.0,
        "ac1": 0.0, "mean_acceleration": 0.0, "std_acceleration": 0.0,
        "early_third": 0.0, "mid_third": 0.0, "late_third": 0.0,
        "early_vs_late": 0.0, "mid_vs_ends": 0.0,
        "peak_density": 0.0, "mean_peak_height": 0.0,
        "frac_above_1std": 0.0,
    }

    if len(entropies) == 0:
        return empty

    x = np.asarray(entropies, dtype=np.float64)
    n = x.size
    if n == 0:
        return empty

    # ── Level ────────────────────────────────────────────────────────────────
    mean = float(x.mean())
    std = float(x.std())
    mx = float(x.max())
    med = float(np.median(x))

    k = int(math.floor(trim_ratio * n))
    trimmed = float(np.sort(x)[k:n - k].mean()) if 2 * k < n else mean

    first_k = max(1, int(math.ceil(first_frac * n)))
    last_k  = max(1, int(math.ceil(last_frac  * n)))
    first_mean = float(x[:first_k].mean())
    last_mean  = float(x[-last_k:].mean())

    # ── Trend ─────────────────────────────────────────────────────────────────
    if n >= 2:
        t = np.arange(n, dtype=np.float64)
        slope = float(np.cov(t, x, bias=True)[0, 1] / (t.var() + 1e-12))
    else:
        slope = 0.0

    delta_end = float(x[-1] - x[0]) if n >= 2 else 0.0
    delta_seg = float(last_mean - first_mean)
    spearman_rho = _spearman_rho_with_position(x)
    kendall_tau = _kendall_tau_with_position(x)

    # ── Volatility ────────────────────────────────────────────────────────────
    if n >= 2:
        dx = np.diff(x)
        up = float(np.maximum(dx, 0.0).sum())
        denom = float(np.abs(dx).sum()) + 1e-12
        monotonicity_up = float(up / denom)

        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n >= 3 else 0.0
    else:
        monotonicity_up = 0.0
        ac1 = 0.0

    # Second derivative: is the drift accelerating?
    if n >= 3:
        ddx = np.diff(dx)
        mean_acceleration = float(ddx.mean())
        std_acceleration  = float(ddx.std())
    else:
        mean_acceleration = 0.0
        std_acceleration  = 0.0

    # ── Structure ─────────────────────────────────────────────────────────────
    # Thirds
    third = max(1, n // 3)
    early_third = float(x[:third].mean())
    mid_third   = float(x[third:2*third].mean()) if n >= 2*third else mean
    late_third  = float(x[2*third:].mean())      if n >= 2*third else mean
    early_vs_late = late_third - early_third
    mid_vs_ends   = mid_third - (early_third + late_third) / 2.0

    # Peak density (requires scipy; falls back gracefully)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(x, prominence=0.3)
        n_peaks = len(peaks)
        peak_density     = float(n_peaks / n)
        mean_peak_height = float(x[peaks].mean()) if n_peaks > 0 else 0.0
    except ImportError:
        peak_density     = 0.0
        mean_peak_height = 0.0

    # Self-relative spikiness
    frac_above_1std = float((x > mean + std).mean())

    return {
        "mean": mean,
        "std": std,
        "max": mx,
        "median": med,
        "trimmed_mean": trimmed,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "slope": slope,
        "delta_end": delta_end,
        "delta_seg": delta_seg,
        "spearman_rho": spearman_rho,
        "kendall_tau": kendall_tau,
        "monotonicity_up": monotonicity_up,
        "mean_acceleration": mean_acceleration,
        "std_acceleration": std_acceleration,
        "ac1": ac1,
        "early_third": early_third,
        "mid_third": mid_third,
        "late_third": late_third,
        "early_vs_late": early_vs_late,
        "mid_vs_ends": mid_vs_ends,
        "peak_density": peak_density,
        "mean_peak_height": mean_peak_height,
        "frac_above_1std": frac_above_1std,
    }


# ============== interm. layer entropy =========

def get_model_components(model):
    """
    Returns (final_norm, lm_head) for any HuggingFace decoder-only model,
    including PeftModel wrappers.
    """
    # Unwrap until we find the norm — handles arbitrary nesting depth
    # (plain model, PeftModel, double-wrapped, etc.)
    inner = model
    final_norm = None

    for _ in range(5):  # max 5 levels of unwrapping
        for attr in ["norm", "ln_f", "final_layernorm", "layer_norm"]:
            if hasattr(inner, attr):
                final_norm = getattr(inner, attr)
                break
        if final_norm is not None:
            break
        # Go one level deeper if possible
        if hasattr(inner, "model"):
            inner = inner.model
        else:
            break

    if final_norm is None:
        attrs = [a for a in dir(inner) if not a.startswith("_")]
        raise ValueError(
            f"Can't find final norm in {type(inner).__name__}. "
            f"Available attributes: {attrs}"
        )

    # lm_head: unwrap from top-level model (For PeftModel, lm_head lives on the base model)
    lm_head = None
    for candidate in [model, getattr(model, "model", None)]:
        if candidate is not None and hasattr(candidate, "lm_head"):
            lm_head = candidate.lm_head
            break

    if lm_head is None:
        raise ValueError("Can't find lm_head.")

    return final_norm, lm_head

def get_layers_to_probe(model, n_probe: int = 8) -> Tuple[List[int], int]:
    """
    Evenly spaced layer indices derived from model config.
    Always includes first and last layer.
    Works for any HuggingFace decoder-only model.
    """
    n_layers = None
    for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        if hasattr(model.config, attr):
            n_layers = getattr(model.config, attr)
            break
    if n_layers is None:
        raise ValueError(f"Can't determine n_layers from config: {model.config}")

    indices = np.linspace(0, n_layers - 1, n_probe, dtype=int).tolist()
    return sorted(set(indices)), n_layers


@torch.no_grad()
def compute_intermediate_layer_entropies(
    model,
    tokenizer,
    prompt: str,
    final_norm,
    lm_head,
    layers_to_probe: List[int],
    device: Optional[torch.device] = None,
    max_length: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Single forward pass with output_hidden_states=True.
    Projects each intermediate hidden state through final_norm + lm_head,
    computes entropy at each (layer, position) pair.

    Returns:
        H : np.ndarray [n_layers_probed, T-1]
        T : number of prompt tokens
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    tok = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    T = int(input_ids.shape[1])
    if T < 2:
        return np.zeros((len(layers_to_probe), 0), dtype=np.float32), T

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states: tuple of (n_layers+1) tensors each [1, T, d_model]
    # index 0 = embedding, index k = after transformer layer k-1
    hidden_states = outputs.hidden_states

    H = np.zeros((len(layers_to_probe), T - 1), dtype=np.float32)

    for i, layer_idx in enumerate(layers_to_probe):
        h = hidden_states[layer_idx + 1].squeeze(0)[:-1, :]  # [T-1, d_model]
        h_normed = final_norm(h)                              # [T-1, d_model]
        logits = lm_head(h_normed).float()                   # [T-1, vocab_size]
        H[i] = _entropy_from_logits(logits).cpu().numpy()
        del h, h_normed, logits

    del outputs, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return H, T


def aggregate_layer_entropy_surface(
    H: np.ndarray,           
    layers_to_probe: List[int],
) -> Dict[str, float]:
    """
    Aggregate [n_layers, T-1] entropy surface into scalar features.

    Three views:
    - L{k}_{metric}          : all metrics on each layer's position trace [T-1]
    - depth_{metric}         : all metrics on layer-mean profile [n_layers]
    - depth_spread_{metric}  : all metrics on layer-std profile [n_layers]
    """
    features = {}
    n_layers, T = H.shape

    if T == 0:
        return {}

    # ── View 1: per-layer position traces ────────────────────────────────
    for i, layer_idx in enumerate(layers_to_probe):
        for k, v in aggregate_entropy_features(H[i].tolist()).items():
            features[f"L{layer_idx}_{k}"] = v

    # ── View 2: layer-mean depth profile ─────────────────────────────────
    layer_mean = H.mean(axis=1)   # [n_layers]
    for k, v in aggregate_entropy_features(layer_mean.tolist()).items():
        features[f"depth_{k}"] = v

    # ── View 3: layer-std depth profile ──────────────────────────────────
    layer_std = H.std(axis=1)     # [n_layers]
    for k, v in aggregate_entropy_features(layer_std.tolist()).items():
        features[f"depth_spread_{k}"] = v

    return features