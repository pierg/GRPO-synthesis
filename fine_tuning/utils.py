from collections import defaultdict
from typing import List
import numpy as np
import torch
import dataclasses

from .types import Episode


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of model predictions.

    This is an optional utility function that can be used for:
    1. Monitoring the model's exploration/exploitation balance
    2. Adding entropy regularization to encourage exploration
    3. Tracking how the model's confidence changes during training

    Higher entropy indicates more uncertainty/exploration, while lower entropy
    indicates more confidence/exploitation in the model's predictions.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards within each group of responses to the same question for fair comparison."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)

        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)

    return output
