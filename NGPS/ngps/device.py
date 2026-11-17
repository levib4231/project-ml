"""Device selection, synchronization, and accelerator telemetry."""
from __future__ import annotations

import platform

import torch


_SUPPORTED = {"cpu", "cuda", "mps"}


def as_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def require_device(device) -> torch.device:
    """Validate and return a supported CPU/CUDA/MPS device."""
    resolved = as_device(device)
    if resolved.type not in _SUPPORTED:
        raise ValueError(f"Unsupported device {resolved}; choose cpu, cuda, or mps")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but unavailable. Select a CUDA runtime.")
    if resolved.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_built() or not backend.is_available():
            raise RuntimeError("Apple MPS is unavailable. Use a Mac PyTorch build with MPS support.")
    return resolved


def is_accelerator(device) -> bool:
    return as_device(device).type in {"cuda", "mps"}


def preferred_device() -> str:
    """Choose CUDA, then Apple MPS, then CPU for interactive commands."""
    if torch.cuda.is_available():
        return "cuda"
    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_built() and backend.is_available():
        return "mps"
    return "cpu"


def synchronize(device) -> None:
    """Synchronize an accelerator before/after timing or telemetry."""
    resolved = as_device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device) -> None:
    """Reset peak memory where the backend exposes that telemetry."""
    resolved = as_device(device)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)


def memory_stats(device) -> dict:
    """Return optional device memory telemetry.

    CUDA exposes peak allocation. MPS currently exposes current and driver
    allocation, but not a portable peak-allocation API, so its peak is null.
    """
    resolved = as_device(device)
    if resolved.type == "cuda":
        return {
            "backend": "cuda",
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)),
            "current_memory_bytes": int(torch.cuda.memory_allocated(resolved)),
            "driver_memory_bytes": None,
        }
    if resolved.type == "mps":
        return {
            "backend": "mps",
            "peak_memory_bytes": None,
            "current_memory_bytes": int(torch.mps.current_allocated_memory()),
            "driver_memory_bytes": int(torch.mps.driver_allocated_memory()),
        }
    return {
        "backend": "cpu",
        "peak_memory_bytes": None,
        "current_memory_bytes": None,
        "driver_memory_bytes": None,
    }


def empty_cache(device) -> None:
    resolved = as_device(device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
    elif resolved.type == "mps":
        torch.mps.empty_cache()


def device_name(device) -> str:
    resolved = as_device(device)
    if resolved.type == "cuda":
        return torch.cuda.get_device_name(resolved)
    if resolved.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    return platform.processor() or platform.machine()
