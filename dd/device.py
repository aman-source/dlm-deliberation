"""Device, dtype, and memory guards.

CPU is for smoke tests only, and only when the weights fit. An 8B model in
bf16 is about 16GB of weights before activations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# Architecture estimate from the published LLaDA-8B config
# (vocab embeddings untied + 32 layers). Not a measured numel.
LLADA_N_PARAMS_ESTIMATE = 8_000_000_000
FIT_OVERHEAD = 1.15


@dataclass(frozen=True)
class DeviceChoice:
    device: torch.device
    dtype: torch.dtype
    budget_hours: float | None
    reason: str


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def budget_hours_for(device: torch.device, override: float | None) -> float | None:
    """Section 9.2. CPU has no full-sweep budget; smoke only."""
    if override is not None:
        return float(override)
    if device.type == "cuda":
        return 48.0
    if device.type == "mps":
        return 24.0
    return None


def available_ram_bytes() -> int | None:
    # /proc/meminfo is Linux only. Without this, Windows and macOS returned None and
    # model_fits(None) let a CPU run download and load 16GB of weights unchecked.
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def weight_bytes(n_params: int, dtype: torch.dtype) -> int:
    return int(n_params) * dtype_nbytes(dtype)


def dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    if dtype == torch.float64:
        return 8
    raise ValueError(f"unsupported dtype {dtype}")


def model_fits(
    available_bytes: int | None,
    *,
    n_params: int = LLADA_N_PARAMS_ESTIMATE,
    dtype: torch.dtype = torch.bfloat16,
    overhead: float = FIT_OVERHEAD,
) -> bool:
    """True when resident memory can hold the weights plus a small overhead.

    Activations are not included. A False result is a hard stop before download.
    None available memory is treated as unknown and does not block a CUDA load
    by itself; callers still check CUDA free memory when present.
    """
    if available_bytes is None:
        return True
    needed = weight_bytes(n_params, dtype) * overhead
    return available_bytes >= needed


def cuda_free_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def preferred_dtype(device: torch.device, preference: str = "bfloat16") -> list[torch.dtype]:
    """Order to try. The first dtype that runs without NaNs is frozen for the sweep."""
    named = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    first = named[preference]
    if device.type == "cuda":
        order = [first, torch.bfloat16, torch.float16, torch.float32]
    elif device.type == "mps":
        order = [first, torch.bfloat16, torch.float16, torch.float32]
    else:
        # CPU bf16 is often emulated and still ~16GB. fp32 is the honest fallback.
        order = [first, torch.float32]
    seen: list[torch.dtype] = []
    for dtype in order:
        if dtype not in seen:
            seen.append(dtype)
    return seen


def describe_environment() -> dict:
    device = select_device()
    info = {
        "device": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "available_ram_bytes": available_ram_bytes(),
        "llada_n_params_estimate": LLADA_N_PARAMS_ESTIMATE,
        "bf16_weight_bytes_estimate": weight_bytes(LLADA_N_PARAMS_ESTIMATE, torch.bfloat16),
    }
    if device.type == "cuda":
        info["cuda_name"] = torch.cuda.get_device_name(device)
        info["cuda_free_bytes"] = cuda_free_bytes(device)
    return info


def write_environment(path: Path, extra: dict | None = None) -> dict:
    payload = describe_environment()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    return payload
