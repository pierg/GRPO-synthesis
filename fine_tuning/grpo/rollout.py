import gc
import torch
from typing import Callable, List, Optional
import re

from .types import Episode, MiniBatch
from qwen.model import Transformer
from tokenizer.tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    print_generation: bool = False,
) -> List[Episode]:
    """Generate multiple responses per question and compute rewards."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Prepare batch for generation
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len

    # Initialize KV cache for efficient generation
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )

    # Initialize tokens tensor with padding
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    
    # Fill in input prompts
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    # Initialize generation state
    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    # Print initial context if requested
    if print_generation:
        initial_context = tokenizer.detokenize(prefix_token_ids[0])
        print(initial_context, end="", flush=True)

    # Generate tokens one by one
    for cur_pos in range(min_prompt_len, total_len):
        # Get next token probabilities
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        
        # Sample next token and handle padding/finished sequences
        next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        next_token = torch.where(is_finished, pad_token_id, next_token)
        
        tokens[:, cur_pos] = next_token
        
        # Print new token if requested
        if print_generation:
            new_token = tokens[0][cur_pos].item()
            token = tokenizer.convert_ids_to_tokens([new_token])[0]
            readable_token = tokenizer.convert_token_to_readable(token)
            print(readable_token, end="", flush=True)
        
        # Check for end of sequence
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        
        prev_pos = cur_pos
        if is_finished.all():
            break

    # Clean up
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to lists for processing
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # Process generated sequences into episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            
            # Compute rewards
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            
            # Create episode
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    
    print("\r", end=" " * 100, flush=True)
    return episodes 