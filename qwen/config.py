from dataclasses import dataclass


@dataclass
class Qwen2Config:
    """Model configuration parameters."""
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    max_window_layers: int = 70
    model_type: str = "qwen2"
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936 