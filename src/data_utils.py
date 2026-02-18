import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# ── Legacy loaders (used by existing scripts) ─────────────────────────────────

def load_safe_prompts(n_samples=100, split="train_sft"):
    """Load helpful, safe prompts from ultrachat."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    prompts = []
    for item in dataset:
        if len(item['messages']) > 0:
            prompts.append(item['messages'][0]['content'])
            if len(prompts) >= n_samples:
                break
    return prompts

def load_benign_prompts(n_samples=100, split="train"):
    """Load general knowledge prompts from dolly."""
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
    prompts = []
    for item in dataset:
        prompts.append(item['instruction'])
        if len(prompts) >= n_samples:
            break
    return prompts

def load_harmful_prompts(n_samples=100, split='train'):
    """Load adversarial/harmful prompts from AdvBench."""
    dataset = load_dataset("walledai/AdvBench", split=split, token=True)
    prompts = [item['prompt'] for item in dataset][:n_samples]
    return prompts


def load_training_and_test_data(
    n_safe_train=50,
    n_benign_train=50,
    n_test_per_category=20,
    safe_dataset=None,
    harmful_dataset=None,
    balance=False,
    balance_seed=42,
):
    """
    Load training and test datasets in one convenient function.

    Args:
        n_safe_train: Number of safe prompts for training
        n_benign_train: Number of benign prompts for training
        n_test_per_category: Number of samples per test category (safe and harmful)
        safe_dataset: Registry name for safe test prompts (e.g. "xstest").
                      If None, uses the default ultrachat loader.
        harmful_dataset: Registry name for harmful test prompts (e.g. "strongreject").
                         If None, uses the default AdvBench loader.
        balance: If True, subsample the larger test set to match the smaller (1:1).
        balance_seed: Random seed for balanced subsampling.

    Returns:
        Dictionary with keys:
            - 'train_prompts': Combined training prompts (safe + benign)
            - 'safe_test': Safe test prompts
            - 'harmful_test': Harmful test prompts
    """
    print("Loading datasets...")

    # Training data (always from the default sources)
    safe_train = load_safe_prompts(n_safe_train)
    benign_train = load_benign_prompts(n_benign_train)
    train_prompts = safe_train + benign_train

    # Test data — use registry if dataset names are provided
    if safe_dataset is not None:
        print(f"  Safe test dataset: {safe_dataset}")
        safe_test = load_prompts_by_name(safe_dataset, n_test_per_category)
    else:
        safe_test = load_safe_prompts(n_test_per_category, split="test_sft")[:n_test_per_category]

    if harmful_dataset is not None:
        print(f"  Harmful test dataset: {harmful_dataset}")
        harmful_test = load_prompts_by_name(harmful_dataset, n_test_per_category)
    else:
        harmful_test = load_harmful_prompts(n_test_per_category, split="train")[:n_test_per_category]

    if balance:
        safe_test, harmful_test = balance_prompts(safe_test, harmful_test, seed=balance_seed)

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


# ── Dataset registry loaders ──────────────────────────────────────────────────
# Each loader: (n_samples: int) -> list[str]
# These are used by the multi-dataset runner via DATASET_REGISTRY.

# --- Harmful ---

def load_advbench(n_samples=500):
    """walledai/AdvBench — 520 standard adversarial prompts."""
    ds = load_dataset("walledai/AdvBench", split="train", token=True)
    return [row["prompt"] for row in ds][:n_samples]


def load_strongreject(n_samples=500):
    """walledai/StrongREJECT — 313 prompts across 6 harm categories."""
    ds = load_dataset("walledai/StrongREJECT", split="train", token=True)
    return [row["forbidden_prompt"] for row in ds][:n_samples]


def load_toxic_chat(n_samples=500):
    """lmsys/toxic-chat — real in-the-wild jailbreak attempts (jailbreaking=1)."""
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    prompts = [row["user_input"] for row in ds if row.get("jailbreaking") == 1]
    return prompts[:n_samples]


def load_wildjailbreak_harmful(n_samples=500):
    """allenai/wildjailbreak — adversarial_harmful multi-tactic jailbreaks."""
    ds = load_dataset(
        "allenai/wildjailbreak", "train",
        delimiter="\t", keep_default_na=False, split="train",
    )
    prompts = [row["adversarial"] for row in ds
               if row.get("data_type") == "adversarial_harmful"
               and row.get("adversarial", "") != ""]
    return prompts[:n_samples]


# --- Safe / Benign ---

def load_ultrachat(n_samples=500):
    """HuggingFaceH4/ultrachat_200k — general safe conversations."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    prompts = []
    for item in ds:
        if item["messages"]:
            prompts.append(item["messages"][0]["content"])
            if len(prompts) >= n_samples:
                break
    return prompts


def load_dolly(n_samples=500):
    """databricks/databricks-dolly-15k — general knowledge prompts."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return [row["instruction"] for row in ds][:n_samples]


def load_xstest(n_samples=500):
    """walledai/XSTest — 250 safe-but-superficially-suspicious prompts."""
    ds = load_dataset("walledai/XSTest", split="test", token=True)
    # XSTest contains both safe and unsafe prompts; types starting with
    # "safe" are the over-refusal test set.
    prompts = [row["prompt"] for row in ds
               if row.get("type", "").startswith("safe")]
    return prompts[:n_samples]


def load_jailbreakbench_benign(n_samples=500):
    """JailbreakBench/JBB-Behaviors — 100 benign hard-negative prompts."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    return [row["Goal"] for row in ds][:n_samples]


def load_wildjailbreak_benign(n_samples=500):
    """allenai/wildjailbreak — adversarial_benign (safe in jailbreak framing)."""
    ds = load_dataset(
        "allenai/wildjailbreak", "train",
        delimiter="\t", keep_default_na=False, split="train",
    )
    prompts = [row["adversarial"] for row in ds
               if row.get("data_type") == "adversarial_benign"
               and row.get("adversarial", "") != ""]
    return prompts[:n_samples]


def load_orbench(n_samples=500):
    """bench-llm/or-bench (hard-1k) — 1,320 harder over-refusal prompts."""
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    return [row["prompt"] for row in ds][:n_samples]


# ── Registry ──────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    # Harmful
    "advbench":                load_advbench,
    "strongreject":            load_strongreject,
    "toxic_chat":              load_toxic_chat,
    "wildjailbreak_harmful":   load_wildjailbreak_harmful,
    # Safe / Benign
    "ultrachat":               load_ultrachat,
    "dolly":                   load_dolly,
    "xstest":                  load_xstest,
    "jailbreakbench_benign":   load_jailbreakbench_benign,
    "wildjailbreak_benign":    load_wildjailbreak_benign,
    "orbench":                 load_orbench,
}


def load_prompts_by_name(dataset_name, n_samples=500):
    """Load prompts from the registry by name.

    Args:
        dataset_name: Key in DATASET_REGISTRY.
        n_samples:    Maximum number of prompts to return.

    Raises:
        KeyError: If dataset_name is not in DATASET_REGISTRY.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name](n_samples)


def balance_prompts(safe_prompts, harmful_prompts, seed=42):
    """Subsample the larger list to match the smaller one (1:1 balance).

    Args:
        safe_prompts:    List of safe prompt strings.
        harmful_prompts: List of harmful prompt strings.
        seed:            Random seed for reproducibility.

    Returns:
        (balanced_safe, balanced_harmful) with equal length.
    """
    rng = random.Random(seed)
    n = min(len(safe_prompts), len(harmful_prompts))
    if len(safe_prompts) > n:
        safe_prompts = rng.sample(safe_prompts, n)
    if len(harmful_prompts) > n:
        harmful_prompts = rng.sample(harmful_prompts, n)
    return safe_prompts, harmful_prompts


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
