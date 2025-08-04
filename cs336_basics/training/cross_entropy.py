from torch import nn
import torch 
from einops import rearrange, reduce, einsum



def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_flat = rearrange(logits, '... vocab -> (...) vocab')
    targets_flat = rearrange(targets, '... -> (...)')
    logits_max = reduce(logits_flat, 'batch vocab -> batch 1', 'max')
    logits_stable = logits_flat - logits_max
    log_softmax = torch.log_softmax(logits_stable, dim=-1)
    target_log_probs = log_softmax.gather(dim=-1, index=targets_flat.unsqueeze(-1))
    target_log_probs = rearrange(target_log_probs, 'batch 1 -> batch')
    return reduce(-target_log_probs, 'batch -> ', 'mean')


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    our_loss = cross_entropy(logits, targets)
    criterion = nn.CrossEntropyLoss()
    pytorch_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    print(f"Наша функция: {our_loss:.6f}")
    print(f"PyTorch CrossEntropyLoss: {pytorch_loss:.6f}")
    print(f"Разность: {abs(our_loss - pytorch_loss):.8f}")
    