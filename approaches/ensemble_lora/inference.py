"""Ensemble inference with LoRA adapter swapping."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List, Dict, Union, Optional
import os


class EnsembleLoRAInference:
    """
    Efficient ensemble inference using LoRA adapter swapping.

    This class loads a base model once and swaps different LoRA adapters
    for ensemble predictions, dramatically reducing memory usage compared
    to full model ensembles.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_paths: List[str],
        device: str = "cuda",
        torch_dtype=torch.float16
    ):
        """
        Initialize ensemble inference.

        Args:
            base_model_name: HuggingFace model name
            adapter_paths: List of paths to LoRA adapters
            device: Device to run inference on
            torch_dtype: Data type for model weights
        """
        self.base_model_name = base_model_name
        self.adapter_paths = adapter_paths
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded base model with {len(adapter_paths)} adapters available")

    def load_adapter(self, adapter_path: str):
        """Load a specific adapter onto the base model."""
        model = PeftModel.from_pretrained(self.base_model, adapter_path)
        model.to(self.device)
        model.eval()
        return model

    def predict_with_adapter(
        self,
        adapter_path: str,
        prompts: Union[str, List[str]],
        max_length: int = 256,
        return_logits: bool = True
    ):
        """
        Get predictions using a specific adapter.

        Args:
            adapter_path: Path to LoRA adapter
            prompts: Single prompt or list of prompts
            max_length: Maximum sequence length
            return_logits: If True, return logits; otherwise return probs

        Returns:
            logits or probabilities for last token
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        model = self.load_adapter(adapter_path)
        all_outputs = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Last token logits

                if return_logits:
                    all_outputs.append(logits.cpu())
                else:
                    probs = F.softmax(logits, dim=-1)
                    all_outputs.append(probs.cpu())

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return torch.stack(all_outputs, dim=0).squeeze(1)

    def ensemble_predict(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 256,
        aggregation: str = "mean"
    ):
        """
        Get ensemble predictions by swapping through all adapters.

        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum sequence length
            aggregation: How to aggregate predictions ('mean' or 'vote')

        Returns:
            aggregated_probs: Ensemble predictions
            all_probs: Individual predictions from each adapter (for uncertainty)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        all_adapter_probs = []

        print(f"Running ensemble inference with {len(self.adapter_paths)} adapters...")
        for adapter_path in tqdm(self.adapter_paths, desc="Adapters"):
            # Get logits from this adapter
            logits = self.predict_with_adapter(
                adapter_path, prompts, max_length, return_logits=True
            )
            probs = F.softmax(logits, dim=-1)
            all_adapter_probs.append(probs)

        # Stack: [n_adapters, n_prompts, vocab_size]
        all_probs = torch.stack(all_adapter_probs, dim=0)

        # Aggregate
        if aggregation == "mean":
            aggregated_probs = all_probs.mean(dim=0)
        elif aggregation == "vote":
            # Majority voting on predicted tokens
            predicted_tokens = all_probs.argmax(dim=-1)  # [n_adapters, n_prompts]
            # Get most common token per prompt
            aggregated_indices = torch.mode(predicted_tokens, dim=0)[0]
            # Convert back to one-hot
            aggregated_probs = F.one_hot(
                aggregated_indices, num_classes=all_probs.shape[-1]
            ).float()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return aggregated_probs, all_probs

    def generate_ensemble(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ):
        """
        Generate text using ensemble (majority voting per token).

        Args:
            prompts: Input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            generated_texts: List of generated sequences
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        generated_texts = []

        for prompt in tqdm(prompts, desc="Generating"):
            # Start with prompt
            current_text = prompt

            for _ in range(max_new_tokens):
                # Get ensemble predictions for next token
                agg_probs, _ = self.ensemble_predict(
                    current_text, aggregation="mean"
                )

                # Sample next token
                if temperature > 0:
                    next_token_id = torch.multinomial(
                        agg_probs.squeeze() / temperature, num_samples=1
                    )
                else:
                    next_token_id = agg_probs.argmax(dim=-1)

                next_token = self.tokenizer.decode(next_token_id)
                current_text += next_token

                # Stop if EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

            generated_texts.append(current_text)

        return generated_texts


def load_ensemble_from_directory(
    base_model_name: str,
    ensemble_dir: str,
    device: str = "cuda"
) -> EnsembleLoRAInference:
    """
    Load ensemble from a directory containing multiple adapter subdirectories.

    Args:
        base_model_name: HuggingFace model name
        ensemble_dir: Directory containing adapter subdirectories
        device: Device to run on

    Returns:
        EnsembleLoRAInference instance
    """
    # Find all adapter directories
    adapter_paths = [
        os.path.join(ensemble_dir, d)
        for d in os.listdir(ensemble_dir)
        if os.path.isdir(os.path.join(ensemble_dir, d))
        and "adapter" in d.lower()
    ]

    if not adapter_paths:
        raise ValueError(f"No adapter directories found in {ensemble_dir}")

    adapter_paths = sorted(adapter_paths)  # Consistent ordering
    print(f"Found {len(adapter_paths)} adapters in {ensemble_dir}")

    return EnsembleLoRAInference(
        base_model_name=base_model_name,
        adapter_paths=adapter_paths,
        device=device
    )


def swap_and_predict(
    base_model: AutoModelForCausalLM,
    adapter_paths: List[str],
    prompts: List[str],
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_length: int = 256
) -> torch.Tensor:
    """
    Lightweight adapter swapping function (functional API).

    Args:
        base_model: Loaded base model
        adapter_paths: List of adapter paths
        prompts: Input prompts
        tokenizer: Tokenizer
        device: Device
        max_length: Max sequence length

    Returns:
        all_probs: [n_adapters, n_prompts, vocab_size]
    """
    all_probs = []

    for adapter_path in adapter_paths:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()

        prompt_probs = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length
                ).to(device)

                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                prompt_probs.append(probs.cpu())

        all_probs.append(torch.stack(prompt_probs, dim=0).squeeze(1))

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return torch.stack(all_probs, dim=0)
