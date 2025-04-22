import torch
from tokenizer.tokenizer import Tokenizer
from qwen.model import Transformer


def generate_response(
    model: Transformer,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Generate a response from the model given chat messages."""
    prompt = tokenizer.encode_chat_with_response_prompt(messages, "")
    encoding = tokenizer.tokenize(prompt)
    input_ids = torch.tensor(encoding.ids, device=device, dtype=torch.long).unsqueeze(0)

    # Print the full prompt
    for token_id in input_ids[0]:
        token_text = tokenizer.convert_ids_to_tokens(token_id.item())
        print(tokenizer.convert_token_to_readable(token_text), end="", flush=True)
    max_batch_size = 1
    max_seq_len = input_ids.shape[1] + max_new_tokens
    model.init_kv_cache(
        max_batch_size, max_seq_len, device=torch.device(device), dtype=torch.bfloat16
    )

    generated_ids = input_ids
    response = ""

    for i in range(max_new_tokens):
        with torch.no_grad():
            logits = model.inference(generated_ids, start_pos=0)

        # Convert logits to float32 for numerical stability
        logits = logits.to(torch.float32)
        # Apply temperature scaling to control randomness (higher = more random)
        logits = logits[:, -1, :] / temperature
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Top-p sampling (nucleus sampling) - keeps only the most likely tokens
        # that sum up to top_p probability mass. This:
        # 1. Prevents sampling of very unlikely tokens
        # 2. Adapts the number of kept tokens based on the distribution
        # 3. Maintains coherence while allowing controlled randomness
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        probs[indices_to_remove] = 0

        # Sample next token from remaining probabilities
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        token_text = tokenizer.convert_ids_to_tokens(token_id)

        # # Handle special tokens
        # if token_text == "<|im_end|>":
        #     break

        # Print token as it's generated
        readable_token = tokenizer.convert_token_to_readable(token_text)
        print(readable_token, end="", flush=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if token_id == tokenizer.eos_token_id:
            break
        response += readable_token

    model.del_kv_cache()
    print()  # New line after generation
    return response
