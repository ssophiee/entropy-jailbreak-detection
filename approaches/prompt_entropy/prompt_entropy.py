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
    Collect entropy values while "reading" the prompt:
    For each position t (except the last), compute entropy of p(x_{t+1} | x_{<=t}).

    Efficient implementation: single forward pass on full prompt:
    logits[:, t, :] predicts token at t+1.

    Returns:
      entropies: list length (n_tokens - 1) typically
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
    # so use logits[:, :-1, :]
    H = _entropy_from_logits(logits[:, :-1, :])      # [1, T-1]
    entropies = H.squeeze(0).detach().float().cpu().tolist()

    return entropies, T


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
    """
    if len(entropies) == 0:
        # Keep feature keys stable
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
            "p10": 0.0, "p90": 0.0, "trimmed_mean": 0.0,
            "first_mean": 0.0, "last_mean": 0.0,
            "auc": 0.0, "slope": 0.0,
            "frac_above_q": 0.0, "range": 0.0,
        }

    x = np.asarray(entropies, dtype=np.float64)
    n = x.size

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

    # AUC-like (sum normalized by length); equivalent to mean but keeps explicit “area”
    auc = float(x.sum())

    # Linear trend (slope) over position index
    if n >= 2:
        t = np.arange(n, dtype=np.float64)
        # slope = cov(t,x)/var(t)
        slope = float(np.cov(t, x, bias=True)[0, 1] / (t.var() + 1e-12))
    else:
        slope = 0.0

    # Spikiness: fraction above a high quantile threshold
    thr = float(np.quantile(x, high_q))
    frac_above = float((x >= thr).mean())

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
