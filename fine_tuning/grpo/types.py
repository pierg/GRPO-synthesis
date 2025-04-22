import dataclasses
from typing import List, Dict, Any

@dataclasses.dataclass
class Episode:
    """Represents a single episode of text generation with its metadata."""
    prefix: str  # Input prompt
    text: str  # Complete generated text
    prefix_token_ids: List[int]  # Token IDs of the input prompt
    prefix_tokens: List[str]  # Token strings of the input prompt
    generated_token_ids: List[int]  # Token IDs of the generated text
    is_finished: bool  # Whether generation completed normally
    reward: float  # Scalar reward for the episode
    reward_info: Dict[str, Any]  # Additional reward metadata

@dataclasses.dataclass
class MiniBatch:
    """Represents a batch of input prompts and their associated data."""
    prefix: List[str]  # Input prompts
    prefix_token_ids: List[List[int]]  # Token IDs for each prompt
    prefix_tokens: List[List[str]]  # Token strings for each prompt
    numbers: List[float]  # Numerical inputs for reward computation
    target: List[float]  # Target values for reward computation 