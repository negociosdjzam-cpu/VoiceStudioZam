"""Independent engine lifecycle; no inheritance from the legacy TTSBackend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict
from threading import RLock
from typing import Iterator

from .types import AudioChunk, AureoEngineError, EngineCapabilities, SynthesisRequest


class AureoEngine(ABC):
    engine_id: str
    display_name: str
    capabilities = EngineCapabilities()

    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @contextmanager
    def session(self) -> Iterator[None]:
        """Serialize a whole load/render/unload operation for this instance."""
        with self._lock:
            yield

    def describe(self) -> dict:
        return {
            "id": self.engine_id, "name": self.display_name,
            "loaded": self.loaded, "capabilities": asdict(self.capabilities),
        }

    def validate(self, request: SynthesisRequest) -> None:
        if not isinstance(request, SynthesisRequest):
            raise TypeError("Expected a SynthesisRequest")
        caps = self.capabilities
        if caps.requires_voice_reference and request.reference is None:
            raise AureoEngineError("reference_required", "This engine requires a voice reference")
        for control in ("seed", "variation", "energy", "expression"):
            if getattr(request, control) is not None and not getattr(caps, control):
                raise AureoEngineError("unsupported_control", f"This engine does not support {control}")
        if request.streaming and not caps.streaming:
            raise AureoEngineError("streaming_unsupported", "This engine does not support real audio streaming")
        if request.reference is not None:
            if not caps.voice_reference:
                raise AureoEngineError("reference_unsupported", "This engine does not support voice references")
            if not request.reference.path.is_file():
                raise AureoEngineError("reference_missing", "The local voice reference file is missing")
        if request.language is not None and caps.languages and request.language not in caps.languages:
            raise AureoEngineError("language_unsupported", "This engine does not support the requested language")

    def load(self) -> None:
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True

    def unload(self) -> None:
        with self._lock:
            try:
                self._unload()  # also cleans up a partially failed load
            finally:
                self._loaded = False

    def generate(self, request: SynthesisRequest) -> AudioChunk:
        with self._lock:
            self.validate(request)
            if request.streaming:
                raise AureoEngineError("use_stream", "Use stream() for a streaming request")
            self.load()
            chunk = self._generate(request)
            if not isinstance(chunk, AudioChunk):
                raise AureoEngineError("invalid_audio", "The engine returned invalid audio")
            return chunk

    def stream(self, request: SynthesisRequest) -> Iterator[AudioChunk]:
        """The lease lasts until exhaustion or explicit iterator.close()."""
        with self._lock:
            self.validate(request)
            if not self.capabilities.streaming:
                raise AureoEngineError("streaming_unsupported", "This engine does not support real audio streaming")
            self.load()
            iterator = self._stream(request)
            sample_rate = None
            try:
                for chunk in iterator:
                    if not isinstance(chunk, AudioChunk):
                        raise AureoEngineError("invalid_audio", "The engine returned invalid audio")
                    if sample_rate is not None and chunk.sample_rate != sample_rate:
                        raise AureoEngineError("sample_rate_changed", "The stream changed its sample rate")
                    sample_rate = chunk.sample_rate
                    yield chunk
                if sample_rate is None:
                    raise AureoEngineError("empty_audio", "The engine produced no audio")
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _generate(self, request: SynthesisRequest) -> AudioChunk: ...

    def _stream(self, request: SynthesisRequest) -> Iterator[AudioChunk]:
        raise AureoEngineError("streaming_unsupported", "Streaming is not implemented")

    @abstractmethod
    def _unload(self) -> None: ...
