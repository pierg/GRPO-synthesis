import math
import torch
from typing import Dict, List

from .types import Episode
from .utils import normalize_rewards_per_group, compute_entropy

def update_policy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Update model parameters using GRPO algorithm."""
    # Normalize rewards and sort episodes by length
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

    # Initialize training state
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(ep.generated_token_ids) for ep in episodes)
    entropy = 0.0

    # Process episodes in micro-batches
    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )

        # Prepare current micro-batch
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]

        # Compute sequence lengths and padding
        batch_lengths = [
            len(ep.prefix_token_ids) + len(ep.generated_token_ids)
            for ep in batch_episodes
        ]
        batch_max_length = max(batch_lengths)

        # Create padded token sequences and masks
        batch_token_ids = [
            ep.prefix_token_ids +
            ep.generated_token_ids +
            [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, ep in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(ep.prefix_token_ids) +
            [1] * len(ep.generated_token_ids) +
            [0] * (batch_max_length - batch_lengths[k])
            for k, ep in enumerate(batch_episodes)
        ]
        batch_advantages = [ep.reward for ep in batch_episodes]

        # Convert to tensors
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        # Compute log probabilities
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        # Compute entropy for monitoring
        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy += (token_entropy * target_masks).sum() / num_target_tokens

        # Compute policy gradient objective
        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj

        # Backward pass
        loss.backward()

    # Update parameters
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    } 