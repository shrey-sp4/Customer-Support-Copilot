"""Utility: deterministic seeding."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] Global seed set to {seed}")
