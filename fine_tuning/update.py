import torch
from typing import Dict, List
from .types import Episode
from .utils import normalize_rewards_per_group, compute_entropy


def prepare_batch(
    episodes: List[Episode],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a batch of episodes for training."""
    # Compute sequence lengths and padding
    lengths = [
        len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in episodes
    ]
    max_len = max(lengths)

    # Initialize lists for batch tensors
    token_ids, masks, advs, old_logprobs = [], [], [], []

    # Process each episode
    for ep in episodes:
        # Combine prefix and generated tokens
        full = ep.prefix_token_ids + ep.generated_token_ids
        # Create mask (0 for prefix, 1 for generated)
        mask = [0] * len(ep.prefix_token_ids) + [1] * len(ep.generated_token_ids)
        pad_len = max_len - len(full)

        # Pad sequences
        token_ids.append(full + [pad_token_id] * pad_len)
        masks.append(mask + [0] * pad_len)
        advs.append(ep.reward)

        # Handle old logprobs (required for GRPO)
        if ep.logprobs_old is not None:
            padded_old = ep.logprobs_old + [0.0] * pad_len
        else:
            padded_old = [0.0] * max_len
        old_logprobs.append(padded_old)

    # Convert to tensors
    return {
        "token_ids": torch.tensor(token_ids, device=device, dtype=torch.long),
        "masks": torch.tensor(masks, device=device, dtype=torch.bool),
        "advantages": torch.tensor(advs, device=device, dtype=torch.float32),
        "logprobs_old": torch.tensor(old_logprobs, device=device, dtype=torch.float32),
        "max_length": max_len,
    }


def compute_policy_gradient(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    method: str = "reinforce",  # or "grpo" or "ppo"
    clip_eps: float = 0.2,
    kl_coef: float = 0.5,  # Coefficient for KL divergence penalty
    value_model: torch.nn.Module | None = None,  # Optional value function for PPO
    value_coef: float = 0.5,  # Coefficient for value loss
) -> Dict[str, torch.Tensor]:
    """Compute policy gradient using REINFORCE, GRPO, or PPO."""
    # Creating sliding windows for next-token prediction
    input_ids = batch["token_ids"][:, :-1]  # All tokens except last
    targets = batch["token_ids"][:, 1:]  # All tokens except first
    mask = batch["masks"][:, 1:]  # 1 for generated tokens, 0 for prefix

    # Forward pass with mixed precision
    with torch.autocast(device_type=device.type, dtype=dtype):
        # Model predicts distribution over next token for each position
        logits = model(input_ids).float()

        # Get value estimates for PPO if value_model is provided
        if method == "ppo" and value_model is not None:
            values = value_model(input_ids).float()  # Shape: (batch_size, seq_len)

    # Compute log-probs of generated tokens
    # For each position, compare predicted distribution with actual next token
    log_probs_new = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # Flatten to (batch*seq_len, vocab_size)
        targets.reshape(-1),  # Flatten to (batch*seq_len)
        ignore_index=pad_token_id,  # Ignore padding tokens
        reduction="none",  # Keep per-token loss
    ).reshape(input_ids.shape[0], -1)  # Reshape back to (batch, seq_len)

    # Compute advantages based on method
    if method == "ppo" and value_model is not None:
        # PPO: Use value function for advantage estimation
        rewards = batch["advantages"][:, None]  # Raw rewards
        gamma = 0.99  # Discount factor
        # Use current values for returns (or implement GAE)
        returns = rewards + gamma * values.detach()
        advantages = returns - values  # Shape: (batch_size, seq_len)
    else:
        # REINFORCE/GRPO: Use raw rewards as advantages
        advantages = batch["advantages"][:, None]  # Expand for broadcasting

    advantages = advantages * mask  # Only consider generated tokens

    # Compute entropy for monitoring (higher = more exploration, lower = more exploitation)
    with torch.no_grad():
        token_entropy = compute_entropy(logits)
        entropy = (
            token_entropy * mask
        ).sum()  # Only count entropy for generated tokens

    if method == "reinforce":
        # REINFORCE: negative expected log-probability weighted by advantages
        # For each generated token, multiply its log-prob by its advantage
        loss = -((log_probs_new * advantages) * mask).sum()

    elif method == "grpo":
        # GRPO: PPO-style clipped objective using old log-probs
        log_probs_old = batch["logprobs_old"][:, 1:]  # Old log-probs for comparison
        # Apply mask before computing ratio to avoid distorted computations
        ratio = torch.exp((log_probs_new - log_probs_old).masked_fill(~mask, 0.0))
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)  # Clip ratio
        obj = torch.min(ratio * advantages, clipped * advantages)  # Take minimum

        # Add KL divergence penalty (KL(new || old))
        kl_div = (log_probs_new - log_probs_old) * mask
        kl_penalty = kl_coef * kl_div.sum()

        # Combine clipped objective with KL penalty
        loss = -(obj * mask).sum() + kl_penalty

    elif method == "ppo":
        # PPO: Clipped objective with value function, entropy bonus, and KL penalty
        log_probs_old = batch["logprobs_old"][:, 1:]  # Old log-probs for comparison
        ratio = torch.exp((log_probs_new - log_probs_old).masked_fill(~mask, 0.0))

        # PPO's clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2) * mask

        # Add entropy bonus to encourage exploration (positive sign)
        entropy_bonus = 0.01 * token_entropy * mask

        # Compute KL divergence between new and old policies (KL(new || old))
        kl_div = (log_probs_new - log_probs_old) * mask
        kl_penalty = kl_coef * kl_div.sum()

        # Compute value loss if value_model is provided
        value_loss = torch.tensor(0.0, device=device)
        if value_model is not None:
            value_loss = value_coef * ((values - returns) * mask).pow(2).sum()

        # Combine losses
        loss = (policy_loss + entropy_bonus + kl_penalty + value_loss).sum()

    else:
        raise ValueError(f"Unknown update method: {method}")

    return {
        "loss": loss,
        "num_tokens": mask.sum(),  # Number of generated tokens
        "entropy": entropy,  # Average entropy of predictions
        "kl_div": kl_div.mean().item()
        if method in ["ppo", "grpo"]
        else 0.0,  # KL divergence for PPO/GRPO
        "value_loss": value_loss.item()
        if method == "ppo" and value_model is not None
        else 0.0,  # Value loss for PPO
        "advantages": advantages.mean().item(),  # Average advantage
    }


def update_policy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    method: str = "reinforce",  # or "grpo" or "ppo"
) -> Dict[str, float]:
    """Update model parameters using either REINFORCE or GRPO."""
    # Normalize rewards and sort episodes by length
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda ep: len(ep.prefix_token_ids) + len(ep.generated_token_ids))

    # Initialize training statistics
    total_loss, total_tokens, total_entropy = 0.0, 0, 0.0

    # Process episodes in micro-batches
    for i in range(0, len(episodes), micro_batch_size):
        # Prepare current micro-batch
        batch_eps = episodes[i : i + micro_batch_size]
        batch = prepare_batch(batch_eps, pad_token_id, device)

        # Compute loss using selected method
        stats = compute_policy_gradient(
            model, batch, pad_token_id, device, dtype, method
        )
        total_loss += stats["loss"].item()
        total_tokens += stats["num_tokens"].item()
        total_entropy += stats["entropy"].item()

        # Backward pass
        stats["loss"].backward()

    # Update parameters with gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / total_tokens,
        "grad_norm": grad_norm.item(),
        "entropy": total_entropy / total_tokens,  # Average entropy per token
    }
