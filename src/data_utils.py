from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

def load_safe_prompts(n_samples=100, split="train_sft"):
    """Load helpful, safe prompts"""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    prompts = []
    for item in dataset:
        if len(item['messages']) > 0:
            user_msg = item['messages'][0]['content']
            if len(user_msg) > 20 and len(user_msg) < 200:
                prompts.append(user_msg)
                if len(prompts) >= n_samples:
                    break
    return prompts

def load_benign_prompts(n_samples=100, split="train"):
    """Load general knowledge prompts"""
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
    prompts = []
    for item in dataset:
        instruction = item['instruction']
        if len(instruction) > 20 and len(instruction) < 200:
            prompts.append(instruction)
            if len(prompts) >= n_samples:
                break
    return prompts

def load_harmful_prompts(n_samples=100, split='train'):
    """Load adversarial/harmful prompts"""
    dataset = load_dataset("walledai/AdvBench", split=split)
    prompts = [item['prompt'] for item in dataset][:n_samples]
    return prompts

def load_training_and_test_data(
    n_safe_train=50,
    n_benign_train=50,
    n_test_per_category=20
):
    """
    Load training and test datasets in one convenient function.

    Args:
        n_safe_train: Number of safe prompts for training
        n_benign_train: Number of benign prompts for training
        n_test_per_category: Number of samples per test category (safe and harmful)

    Returns:
        Dictionary with keys:
            - 'train_prompts': Combined training prompts (safe + benign)
            - 'safe_test': Safe test prompts
            - 'harmful_test': Harmful test prompts
    """
    print("Loading datasets...")

    # Training data
    safe_train = load_safe_prompts(n_safe_train)
    benign_train = load_benign_prompts(n_benign_train)
    train_prompts = safe_train + benign_train

    # Test data
    safe_test = load_safe_prompts(n_test_per_category, split="test_sft")[:n_test_per_category]
    harmful_test = load_harmful_prompts(n_test_per_category, split="eval")[:n_test_per_category]

    print(f"Training samples: {len(train_prompts)}")
    print(f"Safe test samples: {len(safe_test)}")
    print(f"Harmful test samples: {len(harmful_test)}")

    print("\nExample safe prompt:", safe_test[0][:100])
    print("\nExample harmful prompt:", harmful_test[0][:100])

    return {
        'train_prompts': train_prompts,
        'safe_test': safe_test,
        'harmful_test': harmful_test
    }

class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=128):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        text = f"User: {prompt}\nAssistant: This is a helpful response."

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def create_dataloader(prompts, tokenizer, max_length=128, batch_size=8, shuffle=True):
    """
    Create a DataLoader from a list of prompts.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    dataset = PromptDataset(prompts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader