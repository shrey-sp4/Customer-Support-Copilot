"""Utility: device selection."""
import torch
import warnings


def get_device(cfg_device: str = "auto") -> torch.device:
    """Return the best available torch device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[device] CUDA available: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
        else:
            device = torch.device("cpu")
            warnings.warn("[device] CUDA not available — running on CPU. Training will be slow.")
    elif cfg_device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("[device] CUDA requested but not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif cfg_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)
    return device


def use_fp16(cfg_fp16: bool, device: torch.device) -> bool:
    """Return True only when fp16 is configured AND CUDA is present."""
    return cfg_fp16 and device.type == "cuda"
