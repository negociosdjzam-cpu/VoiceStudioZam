"""Sequential, metadata-only comparisons. Audio never leaves this runner."""
from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

from .metrics import MemorySampler, MemoryStats
from .registry import AureoEngineRegistry
from .types import AudioChunk, AureoEngineError, SynthesisRequest, VoiceReference


@dataclass(frozen=True)
class BenchmarkError:
    stage: str
    code: str
    message: str


@dataclass
class BenchmarkResult:
    engine_id: str
    mode: str
    streaming: bool
    load_seconds: float = 0.0
    time_to_first_audio_seconds: float | None = None
    generation_time_to_first_audio_seconds: float | None = None
    generation_seconds: float = 0.0
    total_seconds: float = 0.0
    generated_duration_seconds: float = 0.0
    real_time_factor: float | None = None
    end_to_end_real_time_factor: float | None = None
    sample_rate: int | None = None
    chunks: int = 0
    text_sha256: str | None = None
    reference_sha256: str | None = None
    memory: MemoryStats = field(default_factory=MemoryStats)
    errors: list[BenchmarkError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {**asdict(self), "success": self.success, "partial_audio": bool(self.errors and self.chunks)}


def error_report(stage: str, error: Exception) -> BenchmarkError:
    if isinstance(error, AureoEngineError):
        return BenchmarkError(stage, error.code, str(error))
    # SDK exception messages can contain tokens, transcripts and local paths.
    return BenchmarkError(stage, "engine_error", f"Engine operation failed ({type(error).__name__})")


class AureoBenchmark:
    def __init__(self, registry: AureoEngineRegistry, *, clock: Callable[[], float] = time.perf_counter,
                 memory_factory: Callable[[], MemorySampler] = MemorySampler):
        self.registry = registry
        self.clock = clock
        self.memory_factory = memory_factory

    def run(self, request: SynthesisRequest, engine_ids: Iterable[str], *, cold: bool = True) -> list[BenchmarkResult]:
        """Use the identical immutable request for every engine; continue on errors.

        cold=True unloads before and after each run. cold=False retains the
        configured instances for warm comparisons; their owner must unload them.
        """
        if not isinstance(request, SynthesisRequest):
            raise TypeError("Expected a SynthesisRequest")
        identifiers = tuple(engine_ids)
        if not identifiers or any(not isinstance(item, str) for item in identifiers):
            raise ValueError("Choose at least one engine identifier")
        # Freeze the reference once, so replacing the original file during a
        # comparison cannot feed different recordings to different engines.
        with tempfile.TemporaryDirectory(prefix="aureo-benchmark-") as directory:
            digest = None
            if request.reference is not None:
                target = Path(directory) / ("reference" + request.reference.path.suffix)
                try:
                    hasher = hashlib.sha256()
                    with request.reference.path.open("rb") as source, target.open("wb") as output:
                        while block := source.read(1024 * 1024):
                            hasher.update(block)
                            output.write(block)
                    digest = hasher.hexdigest()
                except OSError:
                    return [BenchmarkResult(
                        identifier, request.mode.value, request.streaming,
                        errors=[BenchmarkError("reference", "reference_unreadable", "The local voice reference cannot be read")],
                    ) for identifier in identifiers]
                request = replace(request, reference=VoiceReference(target, request.reference.transcript))
            results = [self._run_one(request, identifier, cold=cold) for identifier in identifiers]
            text_digest = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
            for result in results:
                result.text_sha256 = text_digest
                result.reference_sha256 = digest
            return results

    def _run_one(self, request: SynthesisRequest, engine_id: str, *, cold: bool) -> BenchmarkResult:
        result = BenchmarkResult(engine_id, request.mode.value, request.streaming)
        started = self.clock()
        stage = "resolve"
        try:
            sampler = self.memory_factory()
        except Exception:
            sampler = None
        try:
            engine = self.registry.get(engine_id)
            # Keep ownership through cleanup; a concurrent profile render may
            # not unload or use the same engine halfway through a benchmark.
            with engine.session():
                stage = "validate"
                try:
                    engine.validate(request)
                    if cold:
                        stage = "reset"
                        engine.unload()
                    if sampler is not None:
                        try:
                            sampler.start()
                        except Exception:
                            pass
                    stage = "load"
                    load_started = self.clock()
                    try:
                        engine.load()
                    finally:
                        result.load_seconds = self.clock() - load_started
                    stage = "generate"
                    generation_started = self.clock()
                    iterator = None
                    try:
                        iterator = engine.stream(request) if request.streaming else iter((engine.generate(request),))
                        for chunk in iterator:
                            if not isinstance(chunk, AudioChunk):
                                raise AureoEngineError("invalid_audio", "The engine returned invalid audio")
                            if result.sample_rate is not None and result.sample_rate != chunk.sample_rate:
                                raise AureoEngineError("sample_rate_changed", "The stream changed its sample rate")
                            observed = self.clock()
                            if result.time_to_first_audio_seconds is None:
                                result.time_to_first_audio_seconds = observed - started
                                result.generation_time_to_first_audio_seconds = observed - generation_started
                            result.sample_rate = chunk.sample_rate
                            result.chunks += 1
                            result.generated_duration_seconds += chunk.duration_seconds
                        if result.chunks == 0:
                            raise AureoEngineError("empty_audio", "The engine produced no audio")
                    finally:
                        result.generation_seconds = self.clock() - generation_started
                        close = getattr(iterator, "close", None)
                        if close is not None:
                            close()
                except Exception as error:
                    result.errors.append(error_report(stage, error))
                finally:
                    if cold or result.errors:
                        try:
                            engine.unload()
                        except Exception as error:
                            result.errors.append(error_report("cleanup", error))
        except Exception as error:
            result.errors.append(error_report(stage, error))
        finally:
            if sampler is not None:
                try:
                    result.memory = sampler.stop()
                except Exception:
                    pass
            result.total_seconds = self.clock() - started
        if result.success and result.generated_duration_seconds:
            result.real_time_factor = result.generation_seconds / result.generated_duration_seconds
            result.end_to_end_real_time_factor = result.total_seconds / result.generated_duration_seconds
        return result
