import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input.

    This function is used in rotary position embeddings (RoPE) to encode position information.
    The rotation helps the model understand the relative positions of tokens in a sequence.

    For a hidden dimension of size d, we split it into two halves of size d/2.
    We then rotate the second half by 180 degrees (multiply by -1) and swap the halves.

    This creates a rotation matrix that depends on the position, allowing the model to
    learn position-dependent attention patterns.

    Args:
        x: Tensor of shape (..., hidden_size)

    Returns:
        Rotated tensor of same shape (..., hidden_size)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """Applies rotary position embeddings to query and key tensors.

    Rotary Position Embeddings (RoPE) are a type of position encoding that:
    1. Uses rotation matrices to encode position information
    2. Preserves relative position information in attention scores
    3. Allows for efficient computation of attention scores

    The rotation is applied using sine and cosine functions of different frequencies,
    which helps the model learn position-dependent patterns.

    Args:
        q, k: Query and key tensors of shape (batch_size, seq_len, num_heads, head_dim)
        cos, sin: Position embeddings of shape (batch_size, seq_len, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        q_embed, k_embed: Position-encoded tensors of same shape as input
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
