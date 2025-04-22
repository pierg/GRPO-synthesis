import json
from pathlib import Path
from typing import Union

import torch
from torch import nn

from .config import Qwen2Config
from .modules import TransformerBlock, RMSNorm, Qwen2RotaryEmbedding


class Transformer(nn.Module):
    """The complete Qwen2 transformer model.

    This model implements a transformer-based language model that can be used for both
    training and inference. The key differences between these modes are:

    Training:
    - Processes full sequences in one forward pass
    - No KV cache needed
    - Returns predictions for all positions

    Inference:
    - Processes sequences incrementally
    - Uses KV cache for efficiency
    - Only returns prediction for next token
    """

    def __init__(self, params: Qwen2Config, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        self.embed_tokens = torch.nn.Embedding(params.vocab_size, params.hidden_size)
        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config=params, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def output_proj(self, x):
        # Input: (batch_size, seq_len, hidden_size)
        # Output: (batch_size, seq_len, vocab_size)
        if self.params.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        else:
            return self.lm_head(x)

    def forward(self, tokens: torch.Tensor):
        # Input: (batch_size, seq_len)
        # Output: (batch_size, seq_len, vocab_size)
        _bsz, seqlen = tokens.shape

        # Token embedding
        h = self.embed_tokens(tokens)

        # Position embeddings
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        pos_emb = self.rotary_emb(h, pos)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, pos_emb)

        # Final normalization and projection
        h = self.norm(h)
        output = self.output_proj(h)

        return output

    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        # Input: (batch_size, seq_len)
        # Output: (batch_size, 1, vocab_size)
        _bsz, seqlen = tokens.shape

        # Token embedding
        h = self.embed_tokens(tokens)

        # Position embeddings
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = pos + start_pos[:, None]
        else:
            pos.add_(start_pos)
        pos_emb = self.rotary_emb(h, pos)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, pos_emb, start_pos=start_pos)

        # Extract last token and project
        h = h[:, -1:, :]
        h = self.norm(h)
        output = self.output_proj(h)

        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize KV cache for all attention layers."""
        for layer in self.layers:
            layer.self_attn.init_kv_cache(
                max_batch_size, max_seq_len, dtype=dtype, device=device
            )

    def del_kv_cache(self):
        """Clear KV cache from all attention layers."""
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        """Loads a pretrained model from checkpoint."""
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        args = Qwen2Config(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            initializer_range=config["initializer_range"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            max_window_layers=config["max_window_layers"],
            model_type=config["model_type"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            sliding_window=config["sliding_window"],
            use_sliding_window=config["use_sliding_window"],
            use_cache=config["use_cache"],
            tie_word_embeddings=config["tie_word_embeddings"],
            torch_dtype=config["torch_dtype"],
        )
        with torch.device("meta"):
            model = cls(params=args, device=device)

        import safetensors.torch

        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True, assign=True)
        return model.to(device)
