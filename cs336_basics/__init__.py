try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except Exception:
    __version__ = "0.0.0"

# Export main components
from .models import TransformerLM, TransformerBlock, MultiheadSelfAttention
from .tokenizer import Tokenizer
from .generation import generate, generate_batch, sample_from_logits

__all__ = [
    "TransformerLM",
    "TransformerBlock",
    "MultiheadSelfAttention",
    "Tokenizer",
    "generate",
    "generate_batch",
    "sample_from_logits",
]
