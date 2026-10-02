"""Environment contract for a dedicated Chatterbox GPU process."""
from dataclasses import dataclass
import os
from pathlib import Path


def _integer(env, name, default, lower, upper):
    value = int(env.get(name, default))
    if not lower <= value <= upper:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class GPUConfig:
    storage: Path
    model_dir: Path
    device: str = "cuda:0"
    cache_entries: int = 3
    minimum_vram_gib: int = 8
    timeout_seconds: int = 600
    max_text_chars: int = 2000
    max_reference_bytes: int = 12 * 1024 * 1024
    max_inline_bytes: int = 8 * 1024 * 1024
    worker_python: Path | None = None

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        root = Path(env.get("AUREO_STORAGE_DIR", "/workspace/aureo")).expanduser().resolve()
        model = Path(env.get("AUREO_MODEL_DIR", str(root / "models/chatterbox"))).expanduser().resolve()
        device = env.get("AUREO_DEVICE", "cuda:0")
        import re
        if not re.fullmatch(r"cuda:\d+", device):
            raise ValueError("This deployment requires an explicit NVIDIA CUDA device")
        executable = env.get("AUREO_WORKER_PYTHON")
        if executable and ("://" in executable or not Path(executable).is_absolute()):
            raise ValueError("AUREO_WORKER_PYTHON must be an absolute local executable path")
        return cls(root, model, device,
                   _integer(env, "AUREO_REFERENCE_CACHE_ENTRIES", 3, 1, 16),
                   _integer(env, "AUREO_MIN_VRAM_GIB", 8, 8, 192),
                   _integer(env, "AUREO_JOB_TIMEOUT_SECONDS", 600, 10, 3600),
                   _integer(env, "AUREO_MAX_TEXT_CHARS", 2000, 1, 10000),
                   worker_python=None if not executable else Path(executable))

    @property
    def references(self):
        return self.storage / "references"

    @property
    def results(self):
        return self.storage / "results"

    def environment(self):
        variables = {
            "AUREO_STORAGE_DIR": str(self.storage), "AUREO_MODEL_DIR": str(self.model_dir),
            "AUREO_DEVICE": self.device, "AUREO_REFERENCE_CACHE_ENTRIES": str(self.cache_entries),
            "AUREO_MIN_VRAM_GIB": str(self.minimum_vram_gib),
            "AUREO_JOB_TIMEOUT_SECONDS": str(self.timeout_seconds),
            "AUREO_MAX_TEXT_CHARS": str(self.max_text_chars),
            "HF_HOME": str(self.storage / "hf"), "HF_HUB_CACHE": str(self.storage / "hf/hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMNIVOICE_DATA_DIR": str(self.storage / "watermark-compat"),
            "OMNIVOICE_ENV_FILE": str(self.storage / "watermark-compat/user-env"),
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        }
        if self.worker_python is not None:
            variables["AUREO_WORKER_PYTHON"] = str(self.worker_python)
        return variables


def verify_gpu(config):
    from ..runtimes.common import torch_for_device
    from ..types import AureoEngineError
    torch = torch_for_device(config.device)
    device = torch.cuda.get_device_properties(config.device)
    if device.total_memory < config.minimum_vram_gib * 1024**3:
        raise AureoEngineError("insufficient_vram", "This deployment requires at least 8 GiB VRAM (or the configured higher floor)")
    return {"name": device.name, "total_vram_bytes": device.total_memory,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
