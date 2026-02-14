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
        torch_dtype=torch.float32  # Changed from float16 to fix NaN issues
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
        model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            is_trainable=False  # Explicitly set inference mode
        )
        model.to(self.device)
        model.eval()
        # Ensure float32 for numerical stability
        model = model.float()
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

                # Debug: Check logits before conversion
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARNING: Logits contain NaN/Inf BEFORE float conversion!")
                    print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                    print(f"  Inf count: {torch.isinf(logits).sum().item()}")
                    print(f"  Logits dtype: {logits.dtype}")
                    print(f"  Logits min: {logits[~torch.isnan(logits) & ~torch.isinf(logits)].min().item() if (~torch.isnan(logits) & ~torch.isinf(logits)).any() else 'all nan/inf'}")
                    print(f"  Logits max: {logits[~torch.isnan(logits) & ~torch.isinf(logits)].max().item() if (~torch.isnan(logits) & ~torch.isinf(logits)).any() else 'all nan/inf'}")

                if return_logits:
                    # Convert to float32 before moving to CPU to avoid float16 overflow
                    all_outputs.append(logits.float().cpu())
                else:
                    # Convert to float32 for numerical stability
                    logits_f32 = logits.float()
                    probs = F.softmax(logits_f32, dim=-1)
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

        # Get logits from first adapter for debugging (before tqdm)
        import sys
        sys.stderr.write("\n[DEBUG] Testing first adapter...\n")
        sys.stderr.flush()
        logits_test = self.predict_with_adapter(
            self.adapter_paths[0], prompts, max_length, return_logits=True
        )
        sys.stderr.write(f"  Logits shape: {logits_test.shape}, dtype: {logits_test.dtype}, device: {logits_test.device}\n")
        sys.stderr.write(f"  Logits has NaN: {torch.isnan(logits_test).any().item()}\n")
        sys.stderr.write(f"  Logits has Inf: {torch.isinf(logits_test).any().item()}\n")
        if not torch.isnan(logits_test).any() and not torch.isinf(logits_test).any():
            sys.stderr.write(f"  Logits range: [{logits_test.min().item():.2f}, {logits_test.max().item():.2f}]\n")
            sys.stderr.write(f"  Logits stats: mean={logits_test.mean().item():.2f}, std={logits_test.std().item():.2f}\n")

        # Test softmax
        logits_f32_test = logits_test.float()
        probs_test = F.softmax(logits_f32_test, dim=-1)
        sys.stderr.write(f"  After softmax:\n")
        sys.stderr.write(f"    Probs has NaN: {torch.isnan(probs_test).any().item()}\n")
        sys.stderr.write(f"    Probs has Inf: {torch.isinf(probs_test).any().item()}\n")
        if not torch.isnan(probs_test).any():
            sys.stderr.write(f"    Probs range: [{probs_test.min().item():.6f}, {probs_test.max().item():.6f}]\n")
            sys.stderr.write(f"    Probs sum (first prompt): {probs_test[0, :].sum().item():.6f}\n\n")
        sys.stderr.flush()

        # Now run the actual ensemble loop
        print(f"Running ensemble inference with {len(self.adapter_paths)} adapters...")
        for adapter_path in tqdm(self.adapter_paths, desc="Adapters"):
            # Get logits from this adapter
            logits = self.predict_with_adapter(
                adapter_path, prompts, max_length, return_logits=True
            )
            # Convert to float32 for numerical stability before softmax
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=-1)
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

    def predict_all_positions(
        self,
        adapter_path: str,
        prompt: str,
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Get logits at ALL token positions for a single prompt using one adapter.

        Returns:
            logits: [1, T, vocab_size] full sequence logits
        """
        model = self.load_adapter(adapter_path)

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            outputs = model(**inputs)
            logits = outputs.logits.float().cpu()  # [1, T, V]

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return logits

    def ensemble_predict_all_positions(
        self,
        prompt: str,
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Get logits at all token positions from every adapter for a single prompt.

        Returns:
            all_logits: [n_adapters, T, vocab_size]
        """
        all_logits = []
        for adapter_path in self.adapter_paths:
            logits = self.predict_all_positions(adapter_path, prompt, max_length)
            all_logits.append(logits.squeeze(0))  # [T, V]
        return torch.stack(all_logits, dim=0)  # [n_adapters, T, V]

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
                # Convert to float32 for numerical stability
                logits_f32 = logits.float()
                probs = F.softmax(logits_f32, dim=-1)
                prompt_probs.append(probs.cpu())

        all_probs.append(torch.stack(prompt_probs, dim=0).squeeze(1))

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return torch.stack(all_probs, dim=0)
