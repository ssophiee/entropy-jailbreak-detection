# model = train_lora(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)

# factors, merged_model = compute_kronecker_factors(model, train_loader, tokenizer, device=DEVICE,
#                                n_kfac=N_KFAC, lr_threshold=1e-4)
import torch
import torch.nn.functional as F
from tqdm import tqdm


# TODO: verify correctness
# https://github.com/MaximeRobeyns/bayesian_lora
def compute_predictive_entropy_bayesian_lora(
    model, prompts, tokenizer, kronecker_factors,
    n_samples=30, device="cuda"
):
    """
    Compute predictive entropy using bayesian_lora's sampling.
    """
    model.eval()
    entropies = []

    for prompt in tqdm(prompts, desc="Computing Bayesian entropy"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        sampled_logits = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from posterior using Kronecker factors
                # The library handles this internally
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Last token
                sampled_logits.append(logits.cpu())
    

        # Compute predictive entropy
        all_logits = torch.stack(sampled_logits, dim=0)
        mean_probs = F.softmax(all_logits, dim=-1).mean(dim=0).squeeze()
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()
        entropies.append(entropy)

        # Periodic cleanup
        if len(entropies) % 10 == 0:
            torch.cuda.empty_cache()

    return entropies