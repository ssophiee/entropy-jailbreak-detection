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


@torch.no_grad()
def compute_prompt_entropies(
    model,
    tokenizer,
    prompt: str,
    device: Optional[torch.device] = None,
    max_length: Optional[int] = None,
) -> Tuple[List[float], int]:
    """
    Entropy trace while reading the prompt (teacher forcing):
    For each position t (except the last), compute entropy of p(x_{t+1} | x_{<=t}).

    Single forward pass:
      logits[:, t, :] predicts token at t+1.

    Returns:
      entropies: list length (n_tokens - 1)
      n_tokens: number of prompt tokens
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
    input_ids = tok["input_ids"].to(device)        # [1, T]
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    T = int(input_ids.shape[1])
    if T < 2:
        return [], T

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits                              # [1, T, V]

    # Entropy at position t corresponds to predicting token t+1
    H = _entropy_from_logits(logits[:, :-1, :])      # [1, T-1]
    entropies = H.squeeze(0).detach().float().cpu().tolist()

    return entropies, T


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
    high_q: float = 0.90,
) -> Dict[str, float]:
    """
    Turn a per-token entropy trace into scalar features for classification.
    Includes:
      - level (mean/median/quantiles)
      - volatility (std/range/total_variation)
      - trend (slope, delta_end, delta_seg, spearman_rho, monotonicity_up)
      - structure (peak_pos)
    """
    # Stable keys even for empty traces
    empty = {
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
        "p10": 0.0, "p90": 0.0, "trimmed_mean": 0.0,
        "first_mean": 0.0, "last_mean": 0.0,
        "auc": 0.0, "slope": 0.0,
        "frac_above_q": 0.0, "range": 0.0,

        # new trend/structure metrics
        "delta_end": 0.0,
        "delta_seg": 0.0,
        "spearman_rho": 0.0,
        "total_variation": 0.0,
        "monotonicity_up": 0.0,
        "peak_pos": 0.0,
    }

    if len(entropies) == 0:
        return empty

    x = np.asarray(entropies, dtype=np.float64)
    n = x.size
    if n == 0:
        return empty

    # Basic stats
    mean = float(x.mean())
    std = float(x.std())
    mn = float(x.min())
    mx = float(x.max())
    med = float(np.median(x))
    p10 = float(np.quantile(x, 0.10))
    p90 = float(np.quantile(x, 0.90))
    rng = float(mx - mn)

    # Robust trimmed mean
    k = int(math.floor(trim_ratio * n))
    if 2 * k < n:
        xs = np.sort(x)
        trimmed = float(xs[k:n - k].mean())
    else:
        trimmed = mean

    # Early vs late entropies
    first_k = max(1, int(math.ceil(first_frac * n)))
    last_k = max(1, int(math.ceil(last_frac * n)))
    first_mean = float(x[:first_k].mean())
    last_mean = float(x[-last_k:].mean())

    # AUC-like (sum); you can normalize by n if you prefer, but mean already captures that.
    auc = float(x.sum())

    # Linear trend (slope) over position index
    if n >= 2:
        t = np.arange(n, dtype=np.float64)
        slope = float(np.cov(t, x, bias=True)[0, 1] / (t.var() + 1e-12))
    else:
        slope = 0.0

    # Spikiness: fraction above a high quantile threshold
    thr = float(np.quantile(x, high_q))
    frac_above = float((x >= thr).mean())

    # -----------------------
    # NEW: trend/structure
    # -----------------------

    # Endpoint drift
    delta_end = float(x[-1] - x[0]) if n >= 2 else 0.0

    # Segment drift (more robust than endpoints)
    delta_seg = float(last_mean - first_mean)

    # Monotonic trend robustness (rank correlation)
    spearman_rho = _spearman_rho_with_position(x)

    # Volatility / jaggedness
    if n >= 2:
        dx = np.diff(x)
        total_variation = float(np.abs(dx).sum())
        up = float(np.maximum(dx, 0.0).sum())
        denom = float(np.abs(dx).sum()) + 1e-12
        monotonicity_up = float(up / denom)  # 1 => mostly increasing, 0 => mostly decreasing
    else:
        total_variation = 0.0
        monotonicity_up = 0.0

    # Where the max happens (normalized position in [0,1])
    peak_idx = int(np.argmax(x))
    peak_pos = float(peak_idx / (n - 1)) if n >= 2 else 0.0

    return {
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "median": med,
        "p10": p10,
        "p90": p90,
        "trimmed_mean": trimmed,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "auc": auc,
        "slope": slope,
        "frac_above_q": frac_above,
        "range": rng,

        # new
        "delta_end": delta_end,
        "delta_seg": delta_seg,
        "spearman_rho": spearman_rho,
        "total_variation": total_variation,
        "monotonicity_up": monotonicity_up,
        "peak_pos": peak_pos,
    }


@torch.no_grad()
def compute_prompt_entropy_features(
    model,
    tokenizer,
    prompts: List[str],
    *,
    device: Optional[torch.device] = None,
    max_length: Optional[int] = None,
    trim_ratio: float = 0.10,
    first_frac: float = 0.25,
    last_frac: float = 0.25,
    high_q: float = 0.90,
    return_entropies: bool = False,
) -> List[PromptEntropyResult]:
    """
    Compute prompt entropy trace + aggregated features for each prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    results: List[PromptEntropyResult] = []
    for p in prompts:
        ent, T = compute_prompt_entropies(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            device=device,
            max_length=max_length,
        )
        feats = aggregate_entropy_features(
            ent,
            trim_ratio=trim_ratio,
            first_frac=first_frac,
            last_frac=last_frac,
            high_q=high_q,
        )
        results.append(
            PromptEntropyResult(
                prompt=p,
                n_tokens=T,
                entropies=ent if return_entropies else [],
                features=feats,
            )
        )
    return results
