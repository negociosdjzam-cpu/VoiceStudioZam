import hashlib
import json

import pytest

from aureo import AureoEngineRegistry, SynthesisRequest, VoiceReference
from aureo.adapters import create_default_registry
from aureo.benchmark import AureoBenchmark
from aureo.metrics import MemorySampler, MemoryStats


class NoMemory:
    def start(self):
        pass

    def stop(self):
        return MemoryStats()


def runner(clock, *engines, memory_factory=NoMemory):
    registry = AureoEngineRegistry()
    for engine in engines:
        registry.register(engine)
    return AureoBenchmark(registry, clock=clock, memory_factory=memory_factory)


def test_buffered_timing_includes_real_load_and_cleanup(clock, make_engine):
    engine = make_engine()
    result = runner(clock, engine).run(SynthesisRequest("Hola"), ["test"])[0]
    assert result.success
    assert result.load_seconds == 2
    assert result.time_to_first_audio_seconds == 5
    assert result.generation_time_to_first_audio_seconds == 3
    assert result.generation_seconds == 3 and result.total_seconds == 6
    assert result.generated_duration_seconds == 1
    assert result.real_time_factor == 3 and result.end_to_end_real_time_factor == 6
    assert result.sample_rate == 10 and result.chunks == 1
    assert engine.loads == engine.unloads == 1 and not engine.loaded
    value = result.to_dict()
    assert value["memory"]["ram_peak_bytes"] is None
    assert not value["partial_audio"] and "samples" not in value


def test_stream_timing_observes_first_chunk_before_completion(clock, make_engine):
    engine = make_engine()
    result = runner(clock, engine).run(SynthesisRequest("Hola", streaming=True), ["test"])[0]
    assert result.success and result.chunks == 2
    assert result.time_to_first_audio_seconds == 3
    assert result.generation_time_to_first_audio_seconds == 1
    assert result.generation_seconds == 5 and result.total_seconds == 8
    assert result.generated_duration_seconds == 2 and result.real_time_factor == 2.5
    assert engine.stream_closed and not engine.loaded


@pytest.mark.parametrize("failure,stage,chunks", [
    ("load", "load", 0), ("generate", "generate", 0), ("partial", "generate", 1),
    ("empty", "generate", 0), ("rate", "generate", 1), ("cleanup", "cleanup", 1),
])
def test_failed_engine_does_not_hide_errors_or_stop_next(clock, make_engine, failure, stage, chunks):
    failed, good = make_engine("failed", fail=failure), make_engine("good")
    streaming = failure in ("partial", "empty", "rate")
    results = runner(clock, failed, good).run(SynthesisRequest("private-text", streaming=streaming), ["failed", "missing", "good"])
    assert not results[0].success and results[0].errors[0].stage == stage
    assert results[0].chunks == chunks
    assert results[0].real_time_factor is None
    assert results[0].to_dict()["partial_audio"] == bool(chunks)
    assert results[1].errors[0].code == "unknown_engine"
    assert results[2].success
    encoded = json.dumps([result.to_dict() for result in results])
    assert "private-token" not in encoded and "private-text" not in encoded
    assert not failed.loaded and not good.loaded
    if failure == "generate":
        assert results[0].generation_seconds == 3
    if failure == "load":
        assert results[0].load_seconds == 2 and results[0].time_to_first_audio_seconds is None


def test_warm_runs_keep_instance_without_reloading(clock, make_engine):
    engine = make_engine()
    benchmark = runner(clock, engine)
    first = benchmark.run(SynthesisRequest("Hola"), ["test"], cold=False)[0]
    second = benchmark.run(SynthesisRequest("Hola"), ["test"], cold=False)[0]
    assert first.load_seconds == 2 and first.total_seconds == 5
    assert second.load_seconds == 0 and second.total_seconds == 3
    assert engine.loaded and engine.loads == 1 and engine.unloads == 0
    engine.unload()


def test_shared_reference_is_snapshotted_once_for_all_engines(clock, make_engine, tmp_path):
    original = tmp_path / "private-reference.wav"
    original.write_bytes(b"same original voice")
    observations = []
    def first_hook(request):
        observations.append(request.reference.path.read_bytes())
        original.write_bytes(b"replacement during comparison")
    def second_hook(request):
        observations.append(request.reference.path.read_bytes())
    first, second = make_engine("first", hook=first_hook), make_engine("second", hook=second_hook)
    request = SynthesisRequest("Mismo texto", reference=VoiceReference(original, "Referencia"), seed=8, variation=0.2)
    results = runner(clock, first, second).run(request, ["first", "second"])
    assert observations == [b"same original voice"] * 2
    assert first.requests[0] is second.requests[0]
    copied = first.requests[0].reference
    assert copied.path != original and not copied.path.exists()
    assert copied.transcript == "Referencia" and first.requests[0].seed == 8
    assert request.reference.path == original
    assert all(result.reference_sha256 == hashlib.sha256(b"same original voice").hexdigest() for result in results)
    assert all(result.text_sha256 == hashlib.sha256("Mismo texto".encode()).hexdigest() for result in results)
    assert "private-reference" not in json.dumps([result.to_dict() for result in results])


def test_missing_reference_never_loads_any_engine(clock, make_engine, tmp_path):
    engine = make_engine()
    results = runner(clock, engine).run(SynthesisRequest("Hola", reference=VoiceReference(tmp_path / "missing.wav")), ["test", "unknown"])
    assert all(result.errors[0].code == "reference_unreadable" for result in results)
    assert engine.loads == 0


def test_default_adapters_are_honestly_unconfigured(tmp_path):
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"local reference")
    registry = create_default_registry()
    results = AureoBenchmark(registry, memory_factory=NoMemory).run(
        SynthesisRequest("Hola", reference=VoiceReference(reference), seed=1, variation=0.5),
        [item["id"] for item in registry.describe()],
    )
    assert len(results) == 3
    assert all(result.errors[0].code == "runtime_not_configured" for result in results)
    assert all(result.time_to_first_audio_seconds is None and result.chunks == 0 for result in results)
    assert reference.read_bytes() == b"local reference"


@pytest.mark.parametrize("where", ["factory", "start", "stop"])
def test_memory_measurement_failure_never_fails_audio(clock, make_engine, where):
    class BrokenMemory(NoMemory):
        def start(self):
            if where == "start":
                raise OSError("unavailable")
        def stop(self):
            if where == "stop":
                raise OSError("unavailable")
            return super().stop()
    def factory():
        if where == "factory":
            raise OSError("unavailable")
        return BrokenMemory()
    result = runner(clock, make_engine(), memory_factory=factory).run(SynthesisRequest("Hola"), ["test"])[0]
    assert result.success and result.memory.ram_peak_bytes is None


def test_memory_sampler_retains_maximum_without_requiring_gpu():
    samples = iter([(10, None), (40, 20), (25, 12)])
    sampler = MemorySampler(reader=lambda: next(samples), interval=1)
    sampler.start()
    sampler._sample()
    stats = sampler.stop()
    assert stats.ram_peak_bytes == 40 and stats.vram_peak_bytes == 20
    with pytest.raises(RuntimeError):
        sampler.start()
    unavailable = MemorySampler(reader=lambda: (_ for _ in ()).throw(OSError()))
    unavailable.start()
    assert unavailable.stop().ram_peak_bytes is None


@pytest.mark.parametrize("interval", [True, 0, 2, float("nan"), float("inf")])
def test_memory_interval_validation(interval):
    with pytest.raises(ValueError):
        MemorySampler(interval=interval)
