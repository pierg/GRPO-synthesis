import torch
from pathlib import Path
from loguru import logger
from qwen import Transformer
from tokenizer.tokenizer import Tokenizer


def load_model_and_tokenizer(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load the model and tokenizer from the specified path."""
    torch.set_default_dtype(torch.bfloat16)
    logger.debug(f"Default dtype: {torch.get_default_dtype()}")
    
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.json"))
    logger.debug(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    model = Transformer.from_pretrained(model_path, device=torch.device(device))
    model.eval()
    logger.debug(f"Model loaded on {device}")
    
    return model, tokenizer
