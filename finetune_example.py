"""
Example script demonstrating fine-tuning using different RL methods.

This script shows how to:
1. Load a pretrained model
2. Generate episodes for training
3. Fine-tune using REINFORCE, GRPO, or PPO
4. Save the fine-tuned model

Usage:
    uv run finetuning_example.py --method reinforce --num_epochs 3
    uv run finetuning_example.py --method grpo --num_epochs 3 --clip_eps 0.2
    uv run finetuning_example.py --method ppo --num_epochs 3 --clip_eps 0.2 --kl_coef 0.5
"""

import argparse
from pathlib import Path
import torch

from utils.logging import configure_logging, logger
from utils.loading import load_model_and_tokenizer
from tasks.countdown_task import CountdownTasksDataset, RewardEvaluator
from fine_tuning.rollout import rollout
from fine_tuning.update import update_policy


def main():
    """Main function to demonstrate fine-tuning with different RL methods."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune model using RL methods")
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
        help="Directory to save generated episodes and fine-tuned model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="reinforce",
        choices=["reinforce", "grpo", "ppo"],
        help="RL method to use for fine-tuning",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
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
        "--clip_eps",
        type=float,
        default=0.2,
        help="Clipping parameter for GRPO/PPO",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.5,
        help="KL divergence coefficient for GRPO/PPO",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model computations",
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

    # Check if dtype is supported
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

    # Training loop
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            try:
                # Generate episodes
                episodes = rollout(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    max_gen_len=args.max_gen_len,
                    num_answer_per_question=args.num_answers,
                    reward_function=RewardEvaluator.evaluate,
                    device=device,
                    dtype=dtype,
                )

                # Update policy
                stats = update_policy(
                    model=model,
                    episodes=episodes,
                    tokenizer=tokenizer,
                    method=args.method,
                    clip_eps=args.clip_eps,
                    kl_coef=args.kl_coef,
                    device=device,
                    dtype=dtype,
                )

                # Log statistics
                logger.info(f"Batch {batch_idx + 1} statistics:")
                logger.info(f"  Loss: {stats['loss']:.4f}")
                logger.info(f"  Reward: {stats['reward']:.4f}")
                if args.method in ["grpo", "ppo"]:
                    logger.info(f"  KL Divergence: {stats['kl_div']:.4f}")
                if args.method == "ppo":
                    logger.info(f"  Value Loss: {stats['value_loss']:.4f}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    # Save fine-tuned model
    model_save_path = output_dir / f"model_finetuned_{args.method}"
    model.save_pretrained(model_save_path)
    logger.info(f"Fine-tuned model saved to {model_save_path}")


if __name__ == "__main__":
    main()
