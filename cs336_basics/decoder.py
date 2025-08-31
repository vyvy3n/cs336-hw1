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
            if len(generated_ids) > context_length:
                input_ids = generated_ids[-context_length:]
            else:
                input_ids = generated_ids

            input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
            logits = model(input_tensor)

            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            probabilities = torch.softmax(next_token_logits, dim=-1)

            if top_p < 1.0:

                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                 # mask out tokens beyond nucleus
                cutoff = cum_probs > top_p

                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False

                sorted_probs[cutoff] = 0.0
                probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # sample from nucleus
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            else:
                # multinomial sampling from full distribution
                next_token = torch.multinomial(probabilities, num_samples=1)

            token_id = next_token.item()
            if token_id == stop_token:
                break

            generated_ids.append(token_id)

        return tokenizer.decode(generated_ids)
                

            
