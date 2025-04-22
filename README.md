# GRPO-synthesis

A Python implementation of the Qwen2.5-3B-Instruct model for text generation and fine-tuning using GRPO.

## Prerequisites

1. Download the pretrained model:
```bash
mkdir -p data
cd data
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
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

Run the inference example:
```bash
uv run inference_example.py
```

The example will:
1. Load the Qwen2.5-3B-Instruct model
2. Process a sample chat conversation
3. Generate and display the model's response token by token

## Project Structure

- `qwen/`: Core model implementation
  - `config.py`: Model configuration
  - `model.py`: Transformer model implementation
  - `modules.py`: Model components (attention, feed-forward, etc.)
  - `utils.py`: Helper functions
- `tokenizer/`: Tokenizer implementation
- `utils/`: Utility functions for loading, inference, and logging
- `inference_example.py`: Example usage of the model
