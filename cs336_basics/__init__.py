import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .pretokenization_example import find_chunk_boundaries