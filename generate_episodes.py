"""
Generate episodes from the countdown dataset using the Qwen model.

This script processes the countdown dataset in batches, generates multiple responses per question,
and saves the episodes to JSON files. It supports resuming from interrupted runs and provides
detailed logging of the generation process.

Output:
    The script creates JSON files in the output directory:
    - episodes_batch_0000.json, episodes_batch_0001.json, etc.
    - error_batch_XXXX.txt for any failed batches

Usage:
    uv run generate_episodes.py --batch_size 4 --num_answers 4 --max_gen_len 512 --resume --print_generation
"""

import argparse
from pathlib import Path
import torch
from typing import List, Optional
import sys

from utils.logging import configure_logging
from utils.loading import load_model_and_tokenizer
from utils.logging import logger
from utils.serialization import save_dataclass_list
from tasks.countdown_task import CountdownTasksDataset, RewardEvaluator
from fine_tuning.grpo.rollout import rollout
from fine_tuning.grpo.types import Episode


def print_generated_text(tokens: torch.Tensor, tokenizer, prefix_length: int) -> None:
    """Print the generated text in real-time."""
    # Convert tokens to text
    generated_tokens = tokens[prefix_length:].tolist()
    text = tokenizer.detokenize(generated_tokens)

    # Print with carriage return to overwrite the previous line
    sys.stdout.write(f"\rGenerated text: {text}")
    sys.stdout.flush()


def get_last_batch_index(output_dir: Path) -> int:
    """Get the index of the last processed batch."""
    if not output_dir.exists():
        return -1

    batch_files = list(output_dir.glob("episodes_batch_*.json"))
    if not batch_files:
        return -1

    # Extract batch numbers and find the maximum
    batch_numbers = [int(f.stem.split("_")[-1]) for f in batch_files]
    return max(batch_numbers)


def main():
    """Main function to generate episodes from the countdown dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate episodes from countdown dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/Qwen2.5-3B-Instruct",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Countdown-Tasks-3to4",
        help="Path to the countdown dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/training",
        help="Directory to save generated episodes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of examples to process in each batch",
    )
    parser.add_argument(
        "--num_answers",
        type=int,
        default=4,
        help="Number of answers to generate per question",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=512,
        help="Maximum length of generated responses",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the last processed batch"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model computations",
    )
    parser.add_argument(
        "--print_generation",
        action="store_true",
        help="Print generated text in real-time",
    )
    args = parser.parse_args()

    configure_logging()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device and dtype configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check if dtype is supported on the current device
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        logger.warning("bfloat16 not supported on this device, falling back to float16")
        dtype = torch.float16

    logger.info(f"Using device: {device} with dtype: {dtype}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model = model.to(device=device, dtype=dtype)
    logger.info("Model loaded successfully!")

    # Initialize dataset
    logger.info("Loading countdown dataset...")
    dataset = CountdownTasksDataset(
        tokenizer=tokenizer, data_path=args.data_path, split="train"
    )
    logger.info(f"Dataset loaded with {len(dataset)} examples")

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=CountdownTasksDataset.collate_fn,
        shuffle=True,
    )

    # Determine starting batch
    start_batch = 0
    if args.resume:
        start_batch = get_last_batch_index(output_dir) + 1
        logger.info(f"Resuming from batch {start_batch}")

    # Process batches and generate episodes
    total_episodes = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_batch:
            continue

        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        try:
            # Generate episodes using rollout
            episodes = rollout(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                max_gen_len=args.max_gen_len,
                num_answer_per_question=args.num_answers,
                reward_function=RewardEvaluator.evaluate,
                device=device,
                dtype=dtype,
                print_generation=args.print_generation,
            )

            # Save episodes to file
            output_path = output_dir / f"episodes_batch_{batch_idx:04d}.json"
            save_dataclass_list(episodes, output_path)

            total_episodes += len(episodes)
            logger.info(f"Generated {len(episodes)} episodes (total: {total_episodes})")

            # Print some statistics
            rewards = [episode.reward for episode in episodes]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            logger.info(f"Mean reward for this batch: {mean_reward:.4f}")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            # Save error information
            error_file = output_dir / f"error_batch_{batch_idx:04d}.txt"
            with open(error_file, "w") as f:
                f.write(f"Error in batch {batch_idx}:\n{str(e)}")
            continue


if __name__ == "__main__":
    main()
