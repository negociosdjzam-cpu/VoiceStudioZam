"""Loader prerequisites, checked before importing or initializing any model."""
import importlib
import json
from pathlib import Path

from ..types import AureoEngineError


def require_files(*paths):
    if any(path is None or not Path(path).is_file() or Path(path).stat().st_size == 0 for path in paths):
        raise AureoEngineError("assets_missing", "Required local checkpoints/configuration are missing or empty")


def read_json(path):
    require_files(path)
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AureoEngineError("invalid_assets", "Model configuration must be a JSON object")
    return value


def import_sdk(module):
    try:
        return importlib.import_module(module)
    except ImportError:
        raise AureoEngineError("sdk_missing", "SDK or dependency is missing; install it in the engine's isolated environment") from None


def torch_for_device(device):
    torch = import_sdk("torch")
    if device.startswith("cuda"):
        index = int(device.partition(":")[2] or "0")
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise AureoEngineError("gpu_unavailable", "The requested CUDA device is unavailable")
    elif device == "mps" and not torch.backends.mps.is_available():
        raise AureoEngineError("gpu_unavailable", "The requested MPS device is unavailable")
    return torch
