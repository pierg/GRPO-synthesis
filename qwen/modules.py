import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union

from .config import Qwen2Config
from .utils import apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    A variant of layer normalization that:
    1. Computes the root mean square of the input
    2. Normalizes the input by this value
    3. Applies a learned scale parameter
    
    This is more efficient than standard layer norm and works well with
    transformer architectures.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Input: (batch_size, seq_len, hidden_size)
        # Output: (batch_size, seq_len, hidden_size)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_dtype)
        return x


class Attention(nn.Module):
    """Multi-head attention layer with KV caching.
    
    This layer implements the core attention mechanism:
    1. Projects input into query, key, value vectors
    2. Applies rotary position embeddings
    3. Computes attention scores
    4. Applies attention to value vectors
    
    KV caching is used during generation to store previous key-value states
    and avoid recomputing them for each new token.
    """
    def __init__(self, args: Qwen2Config):
        super().__init__()
        self.n_kv_heads = args.num_key_value_heads or args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.args = args

    def init_kv_cache(self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        """Initialize KV cache for attention."""
        # Shape: (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        """Clear KV cache."""
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, pos_embed: Tuple[torch.Tensor, torch.Tensor], start_pos: Optional[Union[int, torch.Tensor]] = None):
        # Input: (batch_size, seq_len, hidden_size)
        # Output: (batch_size, seq_len, hidden_size)
        bsz, seqlen, _ = x.shape
        
        # Project to query, key, value vectors
        # Shape: (batch_size, seq_len, n_heads * head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply rotary position embeddings
        cos, sin = pos_embed
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        
        if start_pos is not None:
            # Generation mode: use KV cache
            end_pos = start_pos + seqlen
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=self.cache_k[:bsz, :end_pos].transpose(1, 2),
                value=self.cache_v[:bsz, :end_pos].transpose(1, 2),
                is_causal=True if seqlen > 1 else False,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            # Training mode: full attention
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        
        # Reshape and project output
        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.
    
    This layer implements a gated linear unit (GLU) with SiLU activation,
    which has been shown to work well in transformer architectures.
    """
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        # Input: (batch_size, seq_len, hidden_size)
        # Output: (batch_size, seq_len, hidden_size)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    """A single transformer layer.
    
    This layer consists of:
    1. Self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Layer normalization before each sub-layer
    """
    def __init__(self, layer_id: int, args: Qwen2Config):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(dim=args.hidden_size, intermediate_size=args.intermediate_size)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, x: torch.Tensor, pos_embed: Tuple[torch.Tensor, torch.Tensor], start_pos: Optional[Union[int, torch.Tensor]] = None):
        # Input: (batch_size, seq_len, hidden_size)
        # Output: (batch_size, seq_len, hidden_size)
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2RotaryEmbedding(nn.Module):
    """Rotary position embeddings for transformer.
    
    This layer generates position embeddings using rotation matrices.
    The embeddings are computed using sine and cosine functions of different
    frequencies, which helps the model learn position-dependent patterns.
    """
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        # Input shapes:
        # x: (batch_size, seq_len, hidden_size)
        # pos: (batch_size, seq_len)
        # Output shapes:
        # cos, sin: (batch_size, seq_len, head_dim)
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        pos = pos[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype) 