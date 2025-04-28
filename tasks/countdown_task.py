import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from fine_tuning.types import MiniBatch
from tokenizer.tokenizer import Tokenizer

# System message template for the AI assistant
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

# Template for the user's question about creating equations
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

# Initial response prompt for the model
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class CountdownTasksDataset(Dataset):
    """Dataset for countdown tasks - creating equations to reach target values."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        """Loads and splits the countdown task data."""
        data = pd.read_parquet(Path(data_path) / "data")
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns the total number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves and prepares a single example for the model."""
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Encodes the input prefix containing system message, user question, and response prompt."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Combines multiple examples into a batch for the DataLoader."""
        return MiniBatch(
            numbers=[item["nums"] for item in batch],
            target=[item["target"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


class RewardEvaluator:
    """Evaluates format and mathematical correctness of countdown task responses."""

    @staticmethod
    def format_reward(response: str, end_token: Optional[str] = None) -> float:
        """Checks if response follows <think>reasoning</think>\n<answer>solution</answer> format."""
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_regex = r"<think>.*?<\/think>"
        answer_regex = r"<answer>.*?<\/answer>"
        full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

        think_match = re.search(think_regex, response, re.DOTALL)
        answer_match = re.search(answer_regex, response, re.DOTALL)
        full_format_match = re.match(full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0

        reward = 0.0
        if think_match:
            reward += 0.1
        if answer_match:
            reward += 0.5
        return reward

    @staticmethod
    def answer_reward(
        response: str, numbers: List[int] = None, target: int = None
    ) -> float:
        """Checks if answer uses all numbers once and evaluates to target value."""
        answer_match = re.search(r"<answer>(.*?)<\/answer>", response, re.DOTALL)
        if not answer_match:
            return 0.0

        answer_content = answer_match.group(1)
        if not answer_content or not re.match(r"^[0-9+\-*/() ]+$", answer_content):
            return 0.0

        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0

        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0
        except:
            pass
        return 0.0

    @classmethod
    def evaluate(
        cls,
        response: str,
        numbers: List[int] = None,
        target: int = None,
        end_token: str = None,
    ) -> Dict[str, Any]:
        """Combines format reward (0.1) and answer reward (1.0) into total score."""
        format_reward = cls.format_reward("<think>" + response, end_token)
        answer_reward = cls.answer_reward(response, numbers, target)

        return {
            "reward": format_reward * 0.1 + answer_reward,
            "reward_info": {
                "format_reward": format_reward,
                "answer_reward": answer_reward,
            },
        }
