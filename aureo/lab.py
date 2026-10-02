"""Profile-based dispatch below the application's audio/export policy boundary."""
from __future__ import annotations

from typing import Callable, Iterator

from .profiles import AureoVoiceProfile
from .registry import AureoEngineRegistry
from .types import AudioChunk, AureoEngineError, QualityMode


class AureoEngineLab:
    def __init__(self, registry: AureoEngineRegistry, *, audio_processor: Callable[[AudioChunk], AudioChunk]):
        # The host supplies its provenance/postprocessing policy explicitly.
        # VoiceStudio hosts must delegate to services.watermark.mark_synthetic.
        if not callable(audio_processor):
            raise TypeError("An explicit host audio processor is required")
        self.registry = registry
        self.audio_processor = audio_processor

    def _process(self, chunk: AudioChunk) -> AudioChunk:
        output = self.audio_processor(chunk)
        if not isinstance(output, AudioChunk):
            raise AureoEngineError("invalid_audio", "The host audio processor returned invalid audio")
        return output

    def generate(self, profile: AureoVoiceProfile, text: str, *, mode: QualityMode | None = None) -> AudioChunk:
        engine = self.registry.get(profile.select_engine(mode))
        with engine.session():
            return self._process(engine.generate(profile.request(text, mode=mode)))

    def stream(self, profile: AureoVoiceProfile, text: str, *, mode: QualityMode | None = None) -> Iterator[AudioChunk]:
        engine = self.registry.get(profile.select_engine(mode))
        with engine.session():
            iterator = engine.stream(profile.request(text, mode=mode, streaming=True))
            try:
                for chunk in iterator:
                    yield self._process(chunk)
            finally:
                iterator.close()
