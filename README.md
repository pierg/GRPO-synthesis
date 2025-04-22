# GRPO-synthesis

A Python implementation of the Qwen2.5-3B-Instruct model for text generation and fine-tuning using various Reinforcement Learning algorithms.

## Prerequisites

1. Download the pretrained model:
```bash
mkdir -p data
cd data
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
cd ..
```

2. Download the countdown dataset:
```bash
cd data
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
cd ..
```

## Installation

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

## Usage

Run inference on a sample chat:
```bash
uv run inference_example.py
```
This script demonstrates the model's text generation capabilities with a simple chat example.

Generate training episodes for fine-tuning:
```bash
uv run generate_episodes.py --batch_size 4 --num_answers 4 --max_gen_len 512 --resume --print_generation
```
This script generates multiple responses per question and evaluates them using the countdown task reward function.

## Fine-tuning Methods

The codebase supports three different Reinforcement Learning algorithms for fine-tuning:

1. **REINFORCE**
   - Basic policy gradient method
   - Uses raw advantages for policy updates
   - Simple and effective for basic tasks

2. **GRPO (Generalized REINFORCE Policy Optimization)**
   - Extends REINFORCE with ratio clipping
   - Includes KL divergence penalty for policy stability
   - Uses group-based advantages for better credit assignment

3. **PPO (Proximal Policy Optimization)**
   - Value function (critic) for advantage estimation
   - Includes KL divergence penalty and entropy bonus

Each method can be selected by passing the appropriate `method` parameter to the `compute_policy_gradient` function, along with method-specific parameters like `clip_eps`, `kl_coef`, and `value_model`.
