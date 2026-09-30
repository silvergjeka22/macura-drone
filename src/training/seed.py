import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed=0, deterministic=True):
    """Seed every random generator (Python, numpy, torch, cuDNN)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return seed
