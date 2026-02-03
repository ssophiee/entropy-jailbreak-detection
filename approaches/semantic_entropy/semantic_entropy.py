# semantic_entropy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


# --------------------------
# Embedding model cache
# --------------------------
_EMBEDDER_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}


@dataclass
class SemanticEntropyResult:
    prompt: str
    completions: List[str]
    cluster_ids: List[int]
    cluster_sizes: List[int]
    n_clusters: int
    semantic_entropy: float
    semantic_entropy_normalized_logK: float
    semantic_entropy_normalized_logC: float


def _build_chat_prompt(tokenizer, user_prompt: str) -> str:
    """
    Build a prompt compatible with instruct/chat models if a chat template exists.
    Falls back to a simple "User/Assistant" format.
    """
    # Many HF chat tokenizers expose apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": user_prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass
    return f"User: {user_prompt}\nAssistant:"


@torch.no_grad()
def sample_completions(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 20,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Sample multiple completions for a single prompt.
    Uses num_return_sequences for efficiency.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    chat_prompt = _build_chat_prompt(tokenizer, prompt)

    tok = tokenizer(
        chat_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(device)

    gen_kwargs = dict(
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=int(n_samples),
    )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)

    # Make sure pad token exists (common issue with some LLM tokenizers)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    out = model.generate(**tok, **gen_kwargs)

    # Decode all sequences
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    # Extract only the completion part (best-effort)
    completions: List[str] = []
    for s in decoded:
        # Try to split after the assistant marker used by fallback prompt.
        if "Assistant:" in s:
            comp = s.split("Assistant:", 1)[-1].strip()
        else:
            # If chat template was used, the decoded string may include the prompt too.
            # Best-effort: remove the original user_prompt substring if it appears early.
            idx = s.rfind(prompt)
            if idx != -1:
                comp = s[idx + len(prompt):].strip()
            else:
                comp = s.strip()
        completions.append(comp)

    return completions


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over token embeddings using attention mask.
    last_hidden_state: [B, T, H]
    attention_mask:    [B, T]
    returns:           [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
    return summed / denom


def _get_sentence_embedder(
    embed_model_name: str,
    device: torch.device,
) -> SentenceTransformer:
    """
    Cache SentenceTransformer models to avoid re-loading for every prompt.
    Cache key includes model name + device type (cuda/cpu).
    """
    key = (embed_model_name, str(device))
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]

    model = SentenceTransformer(embed_model_name, device=str(device))
    model.eval()
    _EMBEDDER_CACHE[key] = model
    return model


@torch.no_grad()
def embed_texts(
    texts: List[str],
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> torch.Tensor:
    """
    Compute embeddings with sentence-transformers.
    Returns L2-normalized embeddings as torch.Tensor [N, H] on CPU (like your original).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = _get_sentence_embedder(embed_model_name, device=device)

    # Set truncation length if supported (most ST models expose this attr)
    # NOTE: must be set BEFORE encode to take effect; we set it opportunistically.
    if hasattr(embedder, "max_seq_length") and isinstance(max_length, int) and max_length > 0:
        embedder.max_seq_length = max_length

    # sentence-transformers handles tokenization/pooling internally.
    # normalize_embeddings=True gives unit-norm vectors (cosine = dot).
    vecs_np = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        # truncation happens via max_seq_length (below)
    )

    vecs = torch.from_numpy(vecs_np).float()

    # Keep behavior consistent with your old code: return on CPU.
    return vecs.cpu()



def greedy_cosine_clustering(
    embeddings: torch.Tensor,
    threshold: float = 0.82,
) -> Tuple[List[int], List[int]]:
    """
    Greedy clustering:
    - iterate samples in order
    - assign to existing cluster if max cosine(sim to cluster centroid) >= threshold
    - else create new cluster

    embeddings: [N, H], assumed L2-normalized
    returns:
      cluster_ids: length N
      cluster_sizes: sizes for clusters 0..C-1
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [N,H], got {embeddings.shape}")

    embs = embeddings.clone()
    embs = F.normalize(embs, p=2, dim=-1)

    centroids: List[torch.Tensor] = []
    sizes: List[int] = []
    cluster_ids: List[int] = []

    for i in range(embs.size(0)):
        v = embs[i]

        if len(centroids) == 0:
            centroids.append(v.clone())
            sizes.append(1)
            cluster_ids.append(0)
            continue

        C = torch.stack(centroids, dim=0)  # [C, H]
        sims = (C @ v)                     # [C]
        best_sim, best_idx = sims.max(dim=0)

        if float(best_sim) >= float(threshold):
            cid = int(best_idx)
            # Update centroid as running mean (then renormalize)
            new_size = sizes[cid] + 1
            centroids[cid] = F.normalize((centroids[cid] * sizes[cid] + v) / new_size, p=2, dim=-1)
            sizes[cid] = new_size
            cluster_ids.append(cid)
        else:
            centroids.append(v.clone())
            sizes.append(1)
            cluster_ids.append(len(centroids) - 1)

    return cluster_ids, sizes


def semantic_entropy_from_cluster_sizes(cluster_sizes: List[int]) -> Dict[str, float]:
    """
    Compute semantic entropy H = -sum p_c log p_c.
    Also returns:
      - normalized by log(K) (upper bound when all samples are singleton clusters)
      - normalized by log(C) (upper bound given number of clusters)
    """
    K = sum(cluster_sizes)
    if K <= 0:
        return {
            "semantic_entropy": 0.0,
            "semantic_entropy_normalized_logK": 0.0,
            "semantic_entropy_normalized_logC": 0.0,
        }

    ps = [s / K for s in cluster_sizes]
    H = -sum(p * math.log(p + 1e-12) for p in ps)

    H_logK = H / (math.log(K) + 1e-12) if K > 1 else 0.0
    C = len(cluster_sizes)
    H_logC = H / (math.log(C) + 1e-12) if C > 1 else 0.0

    return {
        "semantic_entropy": float(H),
        "semantic_entropy_normalized_logK": float(H_logK),
        "semantic_entropy_normalized_logC": float(H_logC),
    }


@torch.no_grad()
def compute_semantic_entropy(
    model,
    tokenizer,
    prompts: List[str],
    n_samples: int = 20,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: Optional[float] = None,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size: int = 32,
    embed_max_length: int = 256,
    cluster_threshold: float = 0.82,
    device: Optional[torch.device] = None,
    disable_tqdm: bool = False,
) -> List[SemanticEntropyResult]:
    """
    Compute semantic entropy per prompt.

    NOTE: This loads the embedding encoder model inside embed_texts().
          If you want it faster, we can refactor to cache encoder/tokenizer once.
    """
    if device is None:
        device = next(model.parameters()).device

    results: List[SemanticEntropyResult] = []

    for prompt in tqdm(prompts, desc="Semantic entropy", disable=disable_tqdm):
        completions = sample_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )

        # Optional: strip empties to reduce degenerate clusters
        completions = [c.strip() for c in completions]
        if len(completions) == 0:
            results.append(
                SemanticEntropyResult(
                    prompt=prompt,
                    completions=[],
                    cluster_ids=[],
                    cluster_sizes=[],
                    n_clusters=0,
                    semantic_entropy=0.0,
                    semantic_entropy_normalized_logK=0.0,
                    semantic_entropy_normalized_logC=0.0,
                )
            )
            continue

        embs = embed_texts(
            completions,
            embed_model_name=embed_model_name,
            device=device,
            batch_size=embed_batch_size,
            max_length=embed_max_length,
        )

        cluster_ids, cluster_sizes = greedy_cosine_clustering(embs, threshold=cluster_threshold)
        Hs = semantic_entropy_from_cluster_sizes(cluster_sizes)

        results.append(
            SemanticEntropyResult(
                prompt=prompt,
                completions=completions,
                cluster_ids=cluster_ids,
                cluster_sizes=cluster_sizes,
                n_clusters=len(cluster_sizes),
                semantic_entropy=Hs["semantic_entropy"],
                semantic_entropy_normalized_logK=Hs["semantic_entropy_normalized_logK"],
                semantic_entropy_normalized_logC=Hs["semantic_entropy_normalized_logC"],
            )
        )

    return results
