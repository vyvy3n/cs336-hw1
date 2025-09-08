try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except Exception:
    __version__ = "0.0.0"
