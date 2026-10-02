import random
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from aureo import AudioChunk, AureoEngineError, QualityMode, SynthesisRequest, VoiceReference
from aureo.adapters import (
    ChatterboxAdapter, OpenVoiceRuntime, OpenVoiceSource, OpenVoiceV2Adapter,
    Qwen3TTSAdapter, create_default_registry,
)
from aureo.adapters.audio import mono_chunk
from aureo.adapters.seed import seed_scope
from aureo.metrics import read_memory


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"existing local reference")
    return VoiceReference(path, "Hola referencia")


@pytest.fixture
def python_rng_only(monkeypatch):
    # Fake native SDKs below use Python RNG only. Framework-specific scopes
    # are exercised separately without importing or initializing a device.
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "torch", None)


def test_chatterbox_native_controls_and_reference_do_not_leak(reference, python_rng_only):
    calls = []
    default_conditioning = object()
    class Runtime:
        sr = 24000
        conds = default_conditioning
        def generate(self, text, **kwargs):
            calls.append((text, kwargs, random.random()))
            self.conds = "new voice"
            return [[0.1, 0.2, 0.3]]
    runtime = Runtime()
    disposed = []
    adapter = ChatterboxAdapter(lambda: runtime, runtime_disposer=disposed.append)
    request = SynthesisRequest("Hello", reference=reference, mode=QualityMode.ULTRA, seed=11, variation=0.6, energy=0.8, language="en")
    first, second = adapter.generate(request), adapter.generate(request)
    assert first == second and first.sample_rate == 24000
    assert calls[0][1] == {
        "cfg_weight": 0.7, "temperature": pytest.approx(0.64),
        "exaggeration": 0.8, "audio_prompt_path": str(reference.path),
    }
    assert calls[0][2] == calls[1][2]
    assert runtime.conds is default_conditioning
    adapter.generate(SynthesisRequest("Default", mode=QualityMode.FAST))
    assert calls[-1][1]["audio_prompt_path"] is None
    assert calls[-1][1]["cfg_weight"] == 0.3
    adapter.unload()
    adapter.unload()
    assert disposed == [runtime]


def test_chatterbox_restores_conditioning_even_on_sdk_error(reference, python_rng_only):
    class Runtime:
        sr = 10
        conds = "original"
        def generate(self, *args, **kwargs):
            self.conds = "leaked voice"
            raise RuntimeError("failure")
    runtime = Runtime()
    with pytest.raises(RuntimeError):
        ChatterboxAdapter(lambda: runtime).generate(SynthesisRequest("Hola", reference=reference, seed=0))
    assert runtime.conds == "original"


def test_qwen_clone_api_uses_transcript_and_supported_sampling(reference, python_rng_only):
    calls = []
    class Runtime:
        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs)
            return [[0.1] * 20], 24000
    adapter = Qwen3TTSAdapter(lambda: Runtime())
    adapter.generate(SynthesisRequest("Hola", reference=reference, mode="FAST", language="es", variation=0, seed=9))
    assert calls[0] == {
        "text": "Hola", "language": "Spanish", "ref_audio": str(reference.path),
        "ref_text": "Hola referencia", "x_vector_only_mode": False,
        "non_streaming_mode": True, "top_k": 20, "top_p": 0.8, "temperature": 0.1,
    }
    adapter.generate(SynthesisRequest("Hola", reference=VoiceReference(reference.path), mode="ULTRA"))
    assert calls[1]["x_vector_only_mode"] and calls[1]["language"] == "Auto"
    assert calls[1]["top_k"] == 80 and calls[1]["top_p"] == 0.97
    assert "instruct" not in calls[1] and "temperature" not in calls[1]


@pytest.mark.parametrize("adapter_class,kwargs,code", [
    (ChatterboxAdapter, {"language": "es"}, "language_unsupported"),
    (ChatterboxAdapter, {"expression": "calm"}, "unsupported_control"),
    (Qwen3TTSAdapter, {"energy": 0.4}, "unsupported_control"),
    (Qwen3TTSAdapter, {"expression": "happy"}, "unsupported_control"),
    (OpenVoiceV2Adapter, {"expression": "happy"}, "unsupported_control"),
    (ChatterboxAdapter, {"streaming": True}, "streaming_unsupported"),
    (Qwen3TTSAdapter, {"streaming": True}, "streaming_unsupported"),
    (OpenVoiceV2Adapter, {"streaming": True}, "streaming_unsupported"),
])
def test_unsupported_native_controls_never_load_models(adapter_class, kwargs, code, reference):
    loads = []
    adapter = adapter_class(lambda: loads.append("loaded"))
    with pytest.raises(AureoEngineError) as error:
        adapter.generate(SynthesisRequest("Hola", reference=reference, **kwargs))
    assert error.value.code == code and loads == []


@pytest.mark.parametrize("fails", [False, True])
def test_openvoice_needs_explicit_source_and_cleans_intermediate_audio(reference, python_rng_only, fails):
    calls, paths = [], []
    source_embedding, target_embedding = object(), object()
    class Converter:
        hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=22050))
        def extract_se(self, paths):
            assert paths == [str(reference.path)]
            return target_embedding
        def convert(self, **kwargs):
            paths.append(Path(kwargs["audio_src_path"]))
            assert kwargs["src_se"] is source_embedding and kwargs["tgt_se"] is target_embedding
            assert kwargs["output_path"] is None and kwargs["tau"] == pytest.approx(0.5)
            with wave.open(kwargs["audio_src_path"], "rb") as audio:
                assert audio.getframerate() == 16000 and audio.getnframes() == 160
                assert audio.getnchannels() == 1
            if fails:
                raise RuntimeError("conversion failed")
            return [0.1] * 220
    def synthesize_source(request):
        calls.append(request)
        return OpenVoiceSource(AudioChunk((0.3,) * 160, 16000), source_embedding)
    adapter = OpenVoiceV2Adapter(lambda: OpenVoiceRuntime(Converter(), synthesize_source))
    request = SynthesisRequest("Hola", reference=reference, mode="ULTRA", seed=0, variation=0.5)
    if fails:
        with pytest.raises(RuntimeError):
            adapter.generate(request)
    else:
        assert adapter.generate(request).sample_rate == 22050
    assert calls == [request]
    assert paths and not paths[0].exists() and not paths[0].parent.exists()


def test_invalid_native_audio_is_rejected():
    class Tensor:
        def detach(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return [[0.1, 0.2]]
    assert mono_chunk(Tensor(), 10) == AudioChunk((0.1, 0.2), 10)
    for invalid in ([[0.1], [0.2]], [float("nan")], [], object()):
        with pytest.raises(AureoEngineError) as error:
            mono_chunk(invalid, 10)
        assert error.value.code == "invalid_audio"


def test_explicit_factories_validate_configuration_and_release_runtime():
    with pytest.raises(ValueError):
        create_default_registry({"unknown": lambda: None})
    with pytest.raises(TypeError):
        ChatterboxAdapter(True)
    with pytest.raises(TypeError):
        ChatterboxAdapter(runtime_disposer=False)
    adapter = ChatterboxAdapter(lambda: None)
    with pytest.raises(AureoEngineError) as error:
        adapter.load()
    assert error.value.code == "invalid_runtime" and not adapter.loaded


@pytest.mark.parametrize("raises", [False, True])
def test_seed_scope_restores_python_rng_on_success_and_failure(python_rng_only, raises):
    before = random.getstate()
    values = []
    for _ in range(2):
        try:
            with seed_scope(17):
                values.append(random.random())
                if raises:
                    raise RuntimeError("SDK failure")
        except RuntimeError:
            pass
        assert random.getstate() == before
    assert values[0] == values[1]


def test_seed_scope_restores_numpy_torch_cuda_and_mps_without_imports(monkeypatch):
    class RNG:
        def __init__(self):
            self.state = "original"
            self.default_generator = self
        def get_state(self):
            return self.state
        get_rng_state = get_rng_state_all = get_state
        def set_state(self, state):
            self.state = state
        set_rng_state = set_rng_state_all = set_state
        def seed(self, value):
            self.state = value
        manual_seed = manual_seed_all = seed
        def is_initialized(self):
            return True
    numpy_rng, cpu_rng, cuda_rng, mps_rng = RNG(), RNG(), RNG(), RNG()
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=numpy_rng))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(random=cpu_rng, cuda=cuda_rng, mps=mps_rng))
    with pytest.raises(RuntimeError):
        with seed_scope(7, device="mps:0"):
            assert all(rng.state == 7 for rng in (numpy_rng, cpu_rng, cuda_rng, mps_rng))
            raise RuntimeError("render failed")
    assert all(rng.state == "original" for rng in (numpy_rng, cpu_rng, cuda_rng, mps_rng))


def test_memory_reader_does_not_initialize_cuda_or_require_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    class Cuda:
        initialized = False
        def is_initialized(self):
            return self.initialized
        def device_count(self):
            assert self.initialized
            return 2
        def memory_allocated(self, index):
            assert self.initialized
            return [10, 20][index]
    cuda = Cuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert read_memory() == (None, None)
    cuda.initialized = True
    assert read_memory() == (None, 30)
