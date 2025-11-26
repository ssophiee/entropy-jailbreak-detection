from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

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
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer