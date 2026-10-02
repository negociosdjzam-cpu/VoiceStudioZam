import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from threading import Event, Thread

import pytest

from aureo import (
    AudioChunk, AureoEngineError, AureoEngineLab, AureoEngineRegistry,
    AureoVoiceProfile, EngineCapabilities, QualityMode, SynthesisRequest, VoiceReference,
)


@pytest.mark.parametrize("options", [
    {"text": " "}, {"seed": True}, {"seed": -1}, {"seed": 2**32},
    {"variation": float("nan")}, {"energy": float("inf")}, {"energy": True},
    {"variation": -0.1}, {"variation": 1.01}, {"expression": ""},
    {"language": 17}, {"streaming": 1}, {"mode": "INVALID"}, {"reference": "voice.wav"},
])
def test_invalid_requests(options):
    with pytest.raises(ValueError):
        SynthesisRequest(**{"text": "Hola", **options})


@pytest.mark.parametrize("samples,rate", [
    ((), 10), ((float("nan"),), 10), ((True,), 10), ((0.1,), 0),
    ((0.1,), True), (([0.1],), 10),
])
def test_invalid_audio(samples, rate):
    with pytest.raises(ValueError):
        AudioChunk(samples, rate)


def test_requests_and_audio_are_immutable():
    request = SynthesisRequest("Hola", mode="FAST", seed=0, variation=0, energy=1)
    assert request.mode is QualityMode.FAST
    with pytest.raises(FrozenInstanceError):
        request.seed = 3
    assert AudioChunk([0.1] * 50, 100).duration_seconds == 0.5
    with pytest.raises(ValueError):
        VoiceReference("https://example.com/voice.wav")


def test_capability_validation_happens_before_load(make_engine, tmp_path):
    engine = make_engine()
    engine.capabilities = EngineCapabilities()
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"reference")
    requests = [
        (SynthesisRequest("Hola", seed=0), "unsupported_control"),
        (SynthesisRequest("Hola", energy=0), "unsupported_control"),
        (SynthesisRequest("Hola", expression="calm"), "unsupported_control"),
        (SynthesisRequest("Hola", streaming=True), "streaming_unsupported"),
        (SynthesisRequest("Hola", reference=VoiceReference(reference)), "reference_unsupported"),
    ]
    for request, code in requests:
        with pytest.raises(AureoEngineError) as error:
            engine.generate(request)
        assert error.value.code == code
    assert engine.loads == 0
    engine.capabilities = EngineCapabilities(voice_reference=True, requires_voice_reference=True)
    with pytest.raises(AureoEngineError, match="requires a voice reference"):
        engine.generate(SynthesisRequest("Hola"))
    with pytest.raises(AureoEngineError) as error:
        engine.generate(SynthesisRequest("Hola", reference=VoiceReference(tmp_path / "missing.wav")))
    assert error.value.code == "reference_missing"
    assert engine.loads == 0


def test_registry_owns_explicit_isolated_instances(make_engine):
    registry, other = AureoEngineRegistry(), AureoEngineRegistry()
    engine = make_engine()
    registry.register(engine)
    assert registry.get("test") is engine
    with pytest.raises(ValueError):
        registry.register(make_engine())
    with pytest.raises(TypeError):
        registry.register(object())
    with pytest.raises(AureoEngineError) as error:
        other.get("test")
    assert error.value.code == "unknown_engine"
    engine.generate(SynthesisRequest("one"))
    engine.generate(SynthesisRequest("two"))
    assert engine.loads == 1
    engine.unload()
    assert not engine.loaded
    engine.generate(SynthesisRequest("three"))
    assert engine.loads == 2


def test_stream_cancellation_closes_runtime_generator(make_engine):
    engine = make_engine()
    iterator = engine.stream(SynthesisRequest("Hola", streaming=True))
    assert next(iterator).duration_seconds == 1
    assert not engine.stream_closed
    iterator.close()
    assert engine.stream_closed
    engine.unload()


@pytest.mark.parametrize("failure,code", [("empty", "empty_audio"), ("rate", "sample_rate_changed")])
def test_invalid_stream_fails_and_closes(make_engine, failure, code):
    engine = make_engine(fail=failure)
    with pytest.raises(AureoEngineError) as error:
        list(engine.stream(SynthesisRequest("Hola", streaming=True)))
    assert error.value.code == code
    assert engine.stream_closed


def test_render_blocks_concurrent_unload(make_engine):
    entered, release, unloaded = Event(), Event(), Event()
    def hook(request):
        entered.set()
        assert release.wait(3)
    engine = make_engine(hook=hook)
    outputs = []
    render = Thread(target=lambda: outputs.append(engine.generate(SynthesisRequest("Hola"))))
    render.start()
    assert entered.wait(3)
    cleanup = Thread(target=lambda: (engine.unload(), unloaded.set()))
    cleanup.start()
    try:
        assert not unloaded.wait(0.03)
    finally:
        release.set()
        render.join(3)
        cleanup.join(3)
    assert len(outputs) == 1 and unloaded.is_set() and not engine.loaded


def test_lab_dispatches_profile_modes_and_host_audio_policy(make_engine):
    registry = AureoEngineRegistry()
    for name in ("fast", "pro", "ultra"):
        registry.register(make_engine(name))
    processed = []
    def processor(chunk):
        processed.append(chunk)
        return AudioChunk((0.7,) * len(chunk.samples), chunk.sample_rate)
    lab = AureoEngineLab(registry, audio_processor=processor)
    profile = AureoVoiceProfile("voice", "pro", mode_engines={"FAST": "fast", "ULTRA": "ultra"}, seed=42)
    assert lab.generate(profile, "Hola", mode=QualityMode.FAST).samples[0] == 0.7
    assert registry.get("fast").requests[0].seed == 42
    assert registry.get("pro").loads == 0
    assert len(list(lab.stream(profile, "Hola", mode=QualityMode.ULTRA))) == 2
    assert len(processed) == 3
    with pytest.raises(TypeError):
        AureoEngineLab(registry, audio_processor=None)
    bad = AureoEngineLab(registry, audio_processor=lambda chunk: None)
    iterator = bad.stream(profile, "Hola", mode=QualityMode.ULTRA)
    with pytest.raises(AureoEngineError, match="host audio processor"):
        next(iterator)
    assert registry.get("ultra").stream_closed


def test_discovery_in_fresh_process_imports_no_framework_or_legacy():
    script = '''
import json, sys
from aureo.adapters import create_default_registry
from aureo.benchmark import AureoBenchmark
print(json.dumps({"engines": create_default_registry().describe(), "modules": sorted(sys.modules)}))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    forbidden = ("torch", "numpy", "chatterbox", "qwen_tts", "openvoice", "omnivoice", "services", "core")
    assert not any(name.split(".")[0] in forbidden for name in value["modules"])
    assert [item["id"] for item in value["engines"]] == ["chatterbox", "qwen3-tts", "openvoice-v2"]
    assert all(not item["configured"] and not item["loaded"] for item in value["engines"])
