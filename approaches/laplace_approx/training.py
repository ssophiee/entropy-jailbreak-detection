from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
import src.constants as constants

def setup_model_and_lora(model_name, device, lora_rank=8):
    """
    Load model and apply LoRA configuration

    Args:
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        lora_rank: Rank for LoRA adapters

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=constants.LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=constants.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_lora(model, train_loader, epochs=1, lr=3e-4, device=None, save_dir=None, save_name=None, gradient_accumulation_steps=4):
    """
    Train a LoRA-adapted model on provided DataLoader and save the adapter checkpoints.

    Args:
        model: model returned by `get_peft_model` (PEFT model)
        train_loader: PyTorch DataLoader yielding dicts with 'input_ids', 'attention_mask', 'labels'
        epochs: number of training epochs
        lr: learning rate
        device: torch.device or string ('cuda'/'cpu'). If None, will use current model device.
        save_dir: directory to save the fine-tuned adapters. If None, uses `FINE_TUNED_MODEL_DIR` from constants.
        save_name: subdirectory name for this checkpoint. If None, timestamp-based name will be used.

    Returns:
        model, save_path
    """
    import time
    import os
    import torch
    from src.constants import FINE_TUNED_MODEL_DIR

    # Normalize device
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.to(device)
    model.train()

    # Only optimize trainable parameters (LoRA adapters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # Move batch tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            # Huggingface CausalLM returns loss when labels provided
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")

    # Prepare save path
    final_save_dir = save_dir or FINE_TUNED_MODEL_DIR
    os.makedirs(final_save_dir, exist_ok=True)
    if save_name is None:
        save_name = f"lora_finetuned_{int(time.time())}"
    save_path = os.path.join(final_save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)

    # PEFT models support save_pretrained to save adapter weights
    try:
        model.save_pretrained(save_path)
        print(f"Saved fine-tuned model adapters to: {save_path}")
    except Exception as e:
        # Fall back to torch.save of state_dict
        state = {"model_state_dict": model.state_dict()}
        torch.save(state, os.path.join(save_path, "pytorch_model.bin"))
        print(f"Could not use `save_pretrained` - saved state_dict to {save_path}: {e}")

    return model


def load_finetuned(base_model_name_or_model, adapter_path, device=None):
    """
    Load a base model and attach LoRA adapters saved at `adapter_path`.

    Args:
        base_model_name_or_model: HuggingFace model name or an already loaded AutoModelForCausalLM instance.
        adapter_path: path to directory created by `save_pretrained` for the adapter.
        device: torch.device or string. If None, uses CPU or CUDA if available.

    Returns:
        model with adapters loaded and moved to `device`.
    """
    import torch
    import os
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # Load base model if a name is provided
    if isinstance(base_model_name_or_model, str):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_model, device_map="auto" if device.type=="cuda" else None)
    else:
        base_model = base_model_name_or_model

    # Attach adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return model
