"""Explicit, local-only runtime settings; no discovery or provisioning."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ENGINE_IDS = ("chatterbox-multilingual", "qwen3-tts-1.7b-base", "openvoice-v2")
MELO_LANGUAGES = {"en": "EN", "es": "ES", "fr": "FR", "zh": "ZH", "ja": "JP", "ko": "KR"}
PATH_FIELDS = ("python", "checkpoint_dir", "converter_config", "converter_checkpoint", "melo_config", "melo_checkpoint", "source_embedding", "watermark_checkpoint")


@dataclass(frozen=True)
class LocalRuntimeConfig:
    python: Path | None = None
    device: str = "cuda:0"
    checkpoint_dir: Path | None = None
    dtype: str = "auto"
    converter_config: Path | None = None
    converter_checkpoint: Path | None = None
    melo_config: Path | None = None
    melo_checkpoint: Path | None = None
    source_embedding: Path | None = None
    watermark_checkpoint: Path | None = None
    language: str = "es"
    speaker: str = "ES"

    def __post_init__(self):
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|mps|cuda(?::\d+)?", self.device):
            raise ValueError("Select an explicit cpu, mps or cuda device")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("Invalid runtime dtype")
        if self.language not in MELO_LANGUAGES or not isinstance(self.speaker, str) or not self.speaker.strip():
            raise ValueError("Invalid local source voice")
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                if not str(value).strip() or "://" in str(value):
                    raise ValueError("Runtime assets must be local paths")
                object.__setattr__(self, name, Path(value))

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value, *, base: Path | None = None):
        if not isinstance(value, dict) or set(value) - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown runtime configuration fields")
        config = cls(**value)
        if base is not None:
            fields = config.to_dict()
            for key in PATH_FIELDS:
                path = getattr(config, key)
                if path is not None and not path.is_absolute():
                    fields[key] = str(base / path)
            config = cls(**fields)
        return config


def read_runtime_config(path: Path) -> dict[str, LocalRuntimeConfig]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"version", "engines"} or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported local runtime configuration")
    if not isinstance(value["engines"], dict) or set(value["engines"]) - set(ENGINE_IDS):
        raise ValueError("Unknown benchmark engine")
    allowed = {
        "chatterbox-multilingual": {"python", "device", "checkpoint_dir"},
        "qwen3-tts-1.7b-base": {"python", "device", "checkpoint_dir", "dtype"},
        "openvoice-v2": set(PATH_FIELDS) - {"checkpoint_dir"} | {"device", "language", "speaker"},
    }
    for key, item in value["engines"].items():
        if not isinstance(item, dict) or set(item) - allowed[key]:
            raise ValueError("Runtime settings do not apply to this engine")
    return {key: LocalRuntimeConfig.from_dict(item, base=source.parent) for key, item in value["engines"].items()}
