import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed=0, deterministic=True):
    """
    Seed every RNG the pipeline uses (Python, numpy, torch, cuDNN).

    Call once after imports and again before each training run, so a result does not
    depend on how many cells ran beforehand.
    """
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
