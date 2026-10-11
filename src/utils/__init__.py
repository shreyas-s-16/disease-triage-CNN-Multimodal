"""
Shared utilities for the disease-triage pipeline.
"""

import torch
import os
import random
import numpy as np


def get_device() -> torch.device:
    """
    Select the best available compute device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> str:
    """Return the absolute path to the disease-triage project root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
