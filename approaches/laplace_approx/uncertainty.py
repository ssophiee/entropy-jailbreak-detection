"""Uncertainty quantification for Laplace approximation."""
from typing import List, Union, Dict, Optional
from torch.nn import functional as F
import torch
from tqdm import tqdm
from dataclasses import dataclass


from src.constants import MAX_LENGTH, PRIOR_PRECISION
from src.uncertainty import compute_uncertainty_metrics
from approaches.prompt_entropy.prompt_entropy import aggregate_entropy_features, _entropy_from_logits


# =========== intermediate layer entropy (helper functions) ==============

def get_model_components(model):
    """
    Returns (final_norm, lm_head, layers_to_probe, n_layers)
    for any HuggingFace decoder-only model.
    """
    inner = getattr(model, "model", model)

    final_norm = None
    for attr in ["norm", "ln_f", "final_layernorm", "layer_norm"]:
        if hasattr(inner, attr):
            final_norm = getattr(inner, attr)
            break
    if final_norm is None:
        raise ValueError("Can't find final norm. Print model and inspect manually.")

    return final_norm, model.lm_head


def get_layers_to_probe(model, n_probe: int = 8) -> List[int]:
    """Evenly spaced layer indices, always including first and last."""
    n_layers = None
    for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        if hasattr(model.config, attr):
            n_layers = getattr(model.config, attr)
            break
    if n_layers is None:
        raise ValueError(f"Can't determine n_layers from config: {model.config}")

    indices = np.linspace(0, n_layers - 1, n_probe, dtype=int).tolist()
    return sorted(set(indices)), n_layers

# ========================================================

def compute_predictive_entropy(
    model,
    prompts: List[str],
    tokenizer,
    fisher_diag,
    n_samples: int = 20,
    temperature: float = 0.05,
    device: str = "cuda",
    metric: Optional[str] = "intersection_probs_entropy",
    debug: bool = False,
) -> Union[List[float], List[Dict[str, float]]]:
    """
    Compute uncertainty metrics using Laplace approximation.

    Args:
        model: The language model with LoRA parameters
        prompts: List of input prompts
        tokenizer: Tokenizer for the model
        fisher_diag: Diagonal Fisher information for LoRA parameters
        n_samples: Number of posterior samples
        temperature: Temperature scaling for posterior sampling
        device: Device to run computations on
        metric: Specific metric to return, or None for all metrics
            Options: "predictive_entropy", "mutual_information", "variance",
                     "mean_confidence", "intersection_probs_entropy", "mean_entropy"
        debug: Print debug information

    Returns:
        If metric is specified: List of values for that metric
        If metric is None: List of dicts with all metrics
    """
    print(f"Computing uncertainty for {len(prompts)} prompts...")

    model.eval()
    all_metrics = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Computing uncertainty"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        logit_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior: θ ~ N(θ_MAP, (temperature / Fisher))
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std

                        if debug:
                            param_norm = original_state[name].norm().item()
                            noise_norm = noise.norm().item()
                            noise_to_param_ratio = noise_norm / (param_norm + 1e-10)
                            print(f"\n{name}:")
                            print(f"  Param norm: {param_norm:.6f}")
                            print(f"  Noise norm: {noise_norm:.6f}")
                            print(f"  Noise/Param ratio: {noise_to_param_ratio:.4f}")

                        param.data = original_state[name] + noise

                # Forward pass with sampled parameters
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                logit_samples.append(logits.cpu())

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Compute all metrics using shared function
        all_logits = torch.stack(logit_samples, dim=0)  # [n_samples, 1, vocab_size]
        all_probs = F.softmax(all_logits, dim=-1)  # [n_samples, 1, vocab_size]
        metrics = compute_uncertainty_metrics(all_probs)
        all_metrics.append(metrics)

        del inputs, logit_samples, all_logits, all_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if metric is None:
        return all_metrics
    else:
        return [m[metric] for m in all_metrics]


def compute_entropy_trace_features(
    model,
    prompts: List[str],
    tokenizer,
    fisher_diag,
    n_samples: int = 20,
    temperature: float = 0.05,
    device: str = "cuda",
) -> List[Dict[str, float]]:
    """
    Compute per-position predictive-entropy trace using Laplace posterior
    sampling, then aggregate each trace into the same 20 scalar features
    used by the prompt-entropy approach.

    For each token position t, predictive entropy is:
        H[E_samples[p(x_{t+1} | x_{<=t})]]
    where samples are drawn from the Laplace posterior over LoRA weights.

    Args:
        model: The language model with LoRA parameters
        prompts: List of input prompts
        tokenizer: Tokenizer for the model
        fisher_diag: Diagonal Fisher information for LoRA parameters
        n_samples: Number of posterior samples
        temperature: Temperature scaling for posterior sampling
        device: Device to run computations on

    Returns:
        List of dicts (one per prompt), each with the 20 aggregated features.
    """
    print(f"Computing entropy trace features for {len(prompts)} prompts...")

    model.eval()
    all_features = []

    # Get LoRA parameters
    lora_params = {n: p for n, p in model.named_parameters()
                   if 'lora' in n and p.requires_grad}

    # Save original parameters (MAP estimate)
    original_state = {n: p.data.clone() for n, p in lora_params.items()}

    for prompt in tqdm(prompts, desc="Entropy trace features"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        T = inputs["input_ids"].shape[1]
        if T < 2:
            all_features.append(aggregate_entropy_features([]))
            continue

        # Accumulate probability sum in-place — avoids storing [n_samples, T, V]
        prob_sum = torch.zeros(T - 1, model.config.vocab_size, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior
                for name, param in lora_params.items():
                    if name in fisher_diag:
                        precision = fisher_diag[name] + PRIOR_PRECISION
                        std = torch.sqrt(temperature / precision)
                        noise = torch.randn_like(param) * std
                        param.data = original_state[name] + noise

                # Forward pass — keep ALL positions
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(0)[:-1, :].float().cpu()  # [T-1, V]
                prob_sum.add_(F.softmax(logits, dim=-1))
                del outputs, logits

                # Restore original parameters
                for name, param in lora_params.items():
                    param.data = original_state[name]

        # Mean probability across posterior samples at each position
        mean_probs = prob_sum / n_samples              # [T-1, V]

        # Predictive entropy per position
        H = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)  # [T-1]
        entropies = H.tolist()

        features = aggregate_entropy_features(entropies)
        all_features.append(features)

        del inputs, prob_sum, mean_probs, H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_features


# =========== intermediate layer entropy ==============

@dataclass
class IntermediateEntropyResult:
    prompt: str
    n_tokens: int
    layers_probed: List[int]
    H: List[List[float]]          # [n_layers, T-1] as nested list
    features: Dict[str, float]


def aggregate_layer_entropy_surface(
    H: np.ndarray,              # [n_layers, T-1]
    layers_to_probe: List[int],
) -> Dict[str, float]:
    """
    Aggregate [n_layers, T-1] entropy surface into scalar features.

    Three views:
    1. per_layer_*   : all 30 existing metrics applied to each layer's position trace
    2. depth_*       : all 30 existing metrics applied to the layer-mean profile
    3. depth_spread_*: all 30 existing metrics applied to the layer-std profile
    """
    features = {}
    n_layers, T = H.shape

    if T == 0:
        return {}

    # ── View 1: full metrics on each layer's position trace ───────────────
    # This gives you mean/std/slope/total_variation_norm/etc. per layer
    # Prefix: L{actual_layer_index}_{metric}
    for i, layer_idx in enumerate(layers_to_probe):
        layer_feats = aggregate_entropy_features(H[i].tolist())
        for k, v in layer_feats.items():
            features[f"L{layer_idx}_{k}"] = v

    # ── View 2: full metrics on the layer-mean depth profile ─────────────
    # Treats [n_layers] mean-entropy-per-layer as a 1D trace
    # slope here = how steeply entropy decays with depth
    layer_mean = H.mean(axis=1)   # [n_layers]
    for k, v in aggregate_entropy_features(layer_mean.tolist()).items():
        features[f"depth_{k}"] = v

    # ── View 3: full metrics on the layer-std depth profile ──────────────
    # How does the spread of entropy across positions evolve with depth?
    layer_std = H.std(axis=1)     # [n_layers]
    for k, v in aggregate_entropy_features(layer_std.tolist()).items():
        features[f"depth_spread_{k}"] = v

    return features

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
    Compute entropy at each intermediate layer by projecting hidden states
    through the final norm + lm_head.

    Returns:
        H: np.ndarray [n_layers_probed, T-1]
        T: number of prompt tokens
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
        return np.zeros((len(layers_to_probe), 0)), T

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states: tuple of (n_layers+1) tensors, each [1, T, d_model]
    # index 0 = embedding, index k = after transformer layer k-1
    hidden_states = outputs.hidden_states

    H = np.zeros((len(layers_to_probe), T - 1), dtype=np.float32)

    for i, layer_idx in enumerate(layers_to_probe):
        # layer_idx is 0-based transformer layer → hidden_states index is layer_idx+1
        h = hidden_states[layer_idx + 1].squeeze(0)[:-1, :]  # [T-1, d_model]

        # Apply final norm (approximation — see note in docstring)
        h_normed = final_norm(h)                              # [T-1, d_model]

        # Project to vocab space
        logits = lm_head(h_normed).float()                   # [T-1, vocab_size]

        # Entropy
        H[i] = _entropy_from_logits(logits).cpu().numpy()

        del h, h_normed, logits

    del outputs, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return H, T


@torch.no_grad()
def compute_intermediate_entropy_features(
    model,
    tokenizer,
    prompts: List[str],
    *,
    n_probe: int = 8,
    device: Optional[torch.device] = None,
    max_length: Optional[int] = None,
    return_surfaces: bool = False,
) -> List[IntermediateEntropyResult]:
    """
    Compute intermediate layer entropy features for each prompt.
    Model-agnostic: derives layers, norm, and lm_head automatically.
    """
    if device is None:
        device = next(model.parameters()).device

    final_norm, lm_head = get_model_components(model)
    layers_to_probe, _  = get_layers_to_probe(model, n_probe=n_probe)

    results = []
    for p in prompts:
        H, T = compute_intermediate_layer_entropies(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            final_norm=final_norm,
            lm_head=lm_head,
            layers_to_probe=layers_to_probe,
            device=device,
            max_length=max_length,
        )

        if T < 2 or H.shape[1] == 0:
            feats = {}
        else:
            feats = aggregate_layer_entropy_surface(H, layers_to_probe)

        results.append(IntermediateEntropyResult(
            prompt=p,
            n_tokens=T,
            layers_probed=layers_to_probe,
            H=H.tolist() if return_surfaces else [],
            features=feats,
        ))

    return results