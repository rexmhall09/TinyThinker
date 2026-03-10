from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        if requested == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("MPS requested but no MPS device is available")
        return requested

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: str, requested: str) -> torch.dtype | None:
    if requested == "fp32" or device != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: str, amp_dtype: torch.dtype | None):
    if device != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def describe_device(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return device
