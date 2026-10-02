"""Portable AUREO profile bindings; no legacy profile/SQLite mutation."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .types import QualityMode, SynthesisRequest, VoiceReference, valid_identifier


@dataclass(frozen=True)
class AureoVoiceProfile:
    profile_id: str
    engine_id: str
    mode: QualityMode = QualityMode.PRO
    reference: VoiceReference | None = None
    language: str | None = None
    seed: int | None = None
    variation: float | None = None
    energy: float | None = None
    expression: str | None = None
    mode_engines: Mapping[QualityMode, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_identifier(self.profile_id)
        valid_identifier(self.engine_id)
        bindings = {QualityMode(mode): valid_identifier(engine) for mode, engine in self.mode_engines.items()}
        object.__setattr__(self, "mode", QualityMode(self.mode))
        object.__setattr__(self, "mode_engines", MappingProxyType(bindings))
        self.request("validation")  # validate all stored synthesis controls

    def select_engine(self, mode: QualityMode | None = None) -> str:
        selected = self.mode if mode is None else QualityMode(mode)
        return self.mode_engines.get(selected, self.engine_id)

    def request(self, text: str, *, mode: QualityMode | None = None, streaming: bool = False) -> SynthesisRequest:
        return SynthesisRequest(
            text=text, reference=self.reference, mode=self.mode if mode is None else QualityMode(mode),
            language=self.language, seed=self.seed, variation=self.variation,
            energy=self.energy, expression=self.expression, streaming=streaming,
        )

    def to_dict(self) -> dict:
        return {
            "version": 1, "profile_id": self.profile_id, "engine_id": self.engine_id,
            "mode": self.mode.value,
            "reference": None if self.reference is None else {
                "path": str(self.reference.path), "transcript": self.reference.transcript,
            },
            "language": self.language, "seed": self.seed, "variation": self.variation,
            "energy": self.energy, "expression": self.expression,
            "mode_engines": {mode.value: engine for mode, engine in self.mode_engines.items()},
        }

    @classmethod
    def from_dict(cls, value: dict) -> AureoVoiceProfile:
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("Unsupported AUREO profile format")
        allowed = {"version", "profile_id", "engine_id", "mode", "reference", "language", "seed", "variation", "energy", "expression", "mode_engines"}
        if set(value) - allowed:
            raise ValueError("Unknown AUREO profile fields")
        fields = {key: item for key, item in value.items() if key != "version"}
        reference = fields.get("reference")
        if reference is not None:
            if not isinstance(reference, dict) or set(reference) - {"path", "transcript"}:
                raise ValueError("Invalid reference configuration")
            fields["reference"] = VoiceReference(**reference)
        return cls(**fields)

    @classmethod
    def read(cls, path: Path) -> AureoVoiceProfile:
        source = Path(path).resolve()
        profile = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        if profile.reference is not None and not profile.reference.path.is_absolute():
            profile = replace(profile, reference=replace(
                profile.reference, path=source.parent / profile.reference.path,
            ))
        return profile

    def write(self, path: Path) -> None:
        """Atomically replace one explicit profile file, never a shared database."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                value = self.to_dict()
                if self.reference is not None:
                    value["reference"]["path"] = str(self.reference.path.resolve())
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
