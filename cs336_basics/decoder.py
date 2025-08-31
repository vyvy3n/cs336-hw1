import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM

def decode(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_tokens_length: int=32, temperature: float=0.7, top_p: float=0.9):

    stop_token = tokenizer.encode("<|endoftext|>")[0]

    input_ids = tokenizer.encode(prompt)
    device = model.device
    context_length = model.context_length

    generated_ids = input_ids.copy()
    model.eval()

    with torch.no_grad():

        for _ in range(max_tokens_length):

            # prepare input
            current_input_ids = generated_ids[-context_length:] if len(generated_ids) > context_length else generated_ids
            input_tensor = torch.tensor(current_input_ids, device=device).unsqueeze(0)
            logits = model(input_tensor)

            next_token_logits = logits[:, -1, :]

            # Apply temperature scaling and softmax
            if temperature == 0:
                # Greedy decoding: pick the token with the highest logit
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            else:
                scaled_logits = next_token_logits / temperature
                probabilities = torch.softmax(scaled_logits, dim=-1)

                # Apply Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Create a mask to keep only the tokens in the nucleus
                    # All tokens whose cumulative probability (excluding themselves) is less than top_p are kept.
                    # This correctly includes the token that pushes the sum over top_p.
                    mask = cumulative_probs - sorted_probs < top_p
                    
                    # Filter and re-normalize probabilities
                    filtered_probs = sorted_probs * mask.float()
                    # Re-normalize to ensure sum is 1.0 (important after filtering)
                    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

                    # Sample from the filtered distribution
                    next_token_id = sorted_indices[torch.multinomial(filtered_probs, num_samples=1)].squeeze().item()
                else:
                    # If top_p is 1.0, just sample from the temperature-scaled probabilities (no top-p filtering)
                    next_token_id = torch.multinomial(probabilities, num_samples=1).squeeze().item()
            
            token_id = next_token_id
            if token_id == stop_token:
                break

            generated_ids.append(token_id)

    model.train()
    return tokenizer.decode(generated_ids)
                

            
