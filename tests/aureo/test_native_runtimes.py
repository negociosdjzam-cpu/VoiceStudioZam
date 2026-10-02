import json
import socket
import sys
from types import SimpleNamespace

import pytest

from aureo import AudioChunk, AureoEngineError, SynthesisRequest, VoiceReference
from aureo.adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
from aureo.adapters.openvoice_melo import OpenVoiceMeloAdapter
from aureo.adapters.openvoice_v2 import OpenVoiceRuntime, OpenVoiceSource
from aureo.runtimes import LocalRuntimeConfig, create_benchmark_registry, read_runtime_config
from aureo.runtimes.chatterbox import load_chatterbox
from aureo.runtimes.openvoice import load_openvoice
from aureo.runtimes.qwen3 import load_qwen3
from aureo.runtimes.offline import offline_assets_only


@pytest.fixture
def voice_reference(tmp_path):
    path = tmp_path / "voice.wav"
    path.write_bytes(b"shared reference fixture")
    return VoiceReference(path, "Reference transcript")


@pytest.fixture
def without_framework_rng(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "numpy", None)


def test_multilingual_native_controls_and_conditioning(voice_reference, without_framework_rng):
    calls = []
    class Runtime:
        device = "cpu"
        sr = 24000
        conds = "original voice"
        def generate(self, text, **kwargs):
            calls.append((text, kwargs))
            self.conds = "different reference"
            return [[0.1, 0.2]]
    runtime = Runtime()
    engine = ChatterboxMultilingualAdapter(lambda: runtime)
    audio = engine.generate(SynthesisRequest("Hola", reference=voice_reference, language="es", variation=0.3, energy=0.85, fidelity=1, seed=0))
    assert audio.sample_rate == 24000
    assert calls[0][1] == {
        "language_id": "es", "audio_prompt_path": str(voice_reference.path),
        "cfg_weight": pytest.approx(0.9), "temperature": pytest.approx(0.37), "exaggeration": 0.85,
    }
    assert runtime.conds == "original voice"
    with pytest.raises(AureoEngineError) as error:
        engine.generate(SynthesisRequest("Hola", reference=voice_reference, speed=1.2))
    assert error.value.code == "unsupported_control" and len(calls) == 1
    with pytest.raises(AureoEngineError) as error:
        engine.generate(SynthesisRequest("Hola"))
    assert error.value.code == "reference_required"


def test_openvoice_fidelity_interpolates_embeddings_without_disabling_watermark(voice_reference, without_framework_rng):
    calls = []
    class Converter:
        hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=22050))
        def extract_se(self, paths):
            return 10.0
        def convert(self, **kwargs):
            calls.append(kwargs)
            assert "message" not in kwargs
            return [0.1] * 100
    engine = OpenVoiceMeloAdapter(lambda: OpenVoiceRuntime(
        Converter(), lambda request: OpenVoiceSource(AudioChunk((0.1,) * 20, 16000), 2.0),
    ), language="es")
    engine.generate(SynthesisRequest("Hola", reference=voice_reference, language="es", fidelity=0.75, speed=1.2))
    assert calls[0]["tgt_se"] == 8.0
    assert engine.capabilities.speed and engine.capabilities.fidelity
    with pytest.raises(AureoEngineError) as error:
        engine.generate(SynthesisRequest("Hello", reference=voice_reference, language="en"))
    assert error.value.code == "language_unsupported"


def test_discovery_and_missing_assets_do_not_import_sdks(monkeypatch):
    imports = []
    monkeypatch.setattr("aureo.runtimes.common.import_sdk", lambda module: imports.append(module))
    configs = {"chatterbox-multilingual": LocalRuntimeConfig(checkpoint_dir="/missing/model", device="cpu")}
    registry = create_benchmark_registry(configs)
    assert len(registry.describe()) == 3
    assert imports == []
    with pytest.raises(AureoEngineError) as error:
        registry.get("chatterbox-multilingual").load()
    assert error.value.code == "assets_missing" and imports == []


def test_chatterbox_loader_uses_only_from_local(tmp_path, monkeypatch):
    for name in ("ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"):
        (tmp_path / name).write_bytes(b"local asset fixture")
    calls = []
    marker = object()
    sdk = SimpleNamespace(ChatterboxMultilingualTTS=SimpleNamespace(from_local=lambda *args, **kwargs: (calls.append((args, kwargs)), marker)[1]))
    monkeypatch.setattr("aureo.runtimes.chatterbox.import_sdk", lambda name: sdk)
    monkeypatch.setattr("aureo.runtimes.chatterbox.torch_for_device", lambda device: None)
    assert load_chatterbox(LocalRuntimeConfig(checkpoint_dir=tmp_path, device="cpu")) is marker
    assert calls == [((str(tmp_path),), {"device": "cpu"})]


@pytest.mark.parametrize("model_type,size", [("custom_voice", "1.7B"), ("base", "0.6B"), ("voice_design", "1.7B")])
def test_qwen_rejects_wrong_model_before_imports(tmp_path, monkeypatch, model_type, size):
    (tmp_path / "config.json").write_text(json.dumps({"tts_model_type": model_type, "tts_model_size": size}))
    monkeypatch.setattr("aureo.runtimes.qwen3.import_sdk", lambda name: pytest.fail("Must not import SDK"))
    with pytest.raises(AureoEngineError) as error:
        load_qwen3(LocalRuntimeConfig(checkpoint_dir=tmp_path))
    assert error.value.code == "wrong_model"


def test_qwen_local_base_loading_and_cpu_dtype(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"tts_model_type": "base", "tts_model_size": "1.7B"}))
    (tmp_path / "model.safetensors").write_bytes(b"local weight fixture")
    calls = []
    runtime = SimpleNamespace(model=SimpleNamespace(tts_model_type="base", tts_model_size="1.7B"))
    def from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return runtime
    monkeypatch.setattr("aureo.runtimes.qwen3.import_sdk", lambda module: SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setattr("aureo.runtimes.qwen3.torch_for_device", lambda device: SimpleNamespace(float32="float32"))
    assert load_qwen3(LocalRuntimeConfig(checkpoint_dir=tmp_path, device="cpu")) is runtime
    assert calls == [((str(tmp_path),), {"device_map": "cpu", "dtype": "float32", "attn_implementation": "sdpa", "local_files_only": True})]


def test_openvoice_local_components_and_native_melo_speed(tmp_path, monkeypatch, voice_reference):
    paths = {}
    for name in ("converter_config", "converter_checkpoint", "melo_config", "melo_checkpoint", "source_embedding", "watermark_checkpoint"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(b"asset fixture")
    paths["converter_config"].write_text(json.dumps({"_version_": "v2"}))
    paths["melo_config"].write_text(json.dumps({"data": {"spk2id": {"ES": 0}}}))
    calls = []
    embedding = SimpleNamespace(to=lambda device: "source embedding")
    torch = SimpleNamespace(load=lambda path, **kwargs: (calls.append(("embedding", path, kwargs)), embedding)[1])
    watermark = SimpleNamespace(load_model=lambda path="default": (calls.append(("watermark", path)), "native watermark")[1])
    original_watermark_load = watermark.load_model
    class Converter:
        version = "v2"
        def __init__(self, path, **kwargs):
            calls.append(("converter", path, kwargs))
            assert watermark.load_model() == "native watermark"
        def load_ckpt(self, path):
            calls.append(("checkpoint", path))
    class Source:
        hps = SimpleNamespace(data=SimpleNamespace(spk2id={"ES": 0}, sampling_rate=16000))
        def __init__(self, **kwargs):
            calls.append(("melo", kwargs))
        def tts_to_file(self, *args, **kwargs):
            calls.append(("source", args, kwargs))
            return [0.1] * 20
    modules = {"wavmark": watermark, "openvoice.api": SimpleNamespace(ToneColorConverter=Converter), "melo.api": SimpleNamespace(TTS=Source)}
    monkeypatch.setattr("aureo.runtimes.openvoice.import_sdk", modules.__getitem__)
    monkeypatch.setattr("aureo.runtimes.openvoice.torch_for_device", lambda device: torch)
    runtime = load_openvoice(LocalRuntimeConfig(device="cpu", **paths))
    source = runtime.synthesize_source(SynthesisRequest("Hola", reference=voice_reference, speed=1.25, variation=0.5))
    assert source.audio.sample_rate == 16000 and source.speaker_embedding == "source embedding"
    assert watermark.load_model is original_watermark_load
    assert ("watermark", str(paths["watermark_checkpoint"])) in calls
    assert next(call for call in calls if call[0] == "melo")[1] == {
        "language": "ES", "device": "cpu", "use_hf": False,
        "config_path": str(paths["melo_config"]), "ckpt_path": str(paths["melo_checkpoint"]),
    }
    assert next(call for call in calls if call[0] == "source")[2] == {"output_path": None, "speed": 1.25, "quiet": True, "noise_scale": 0.5}
    assert next(call for call in calls if call[0] == "embedding")[2] == {"map_location": "cpu", "weights_only": True}


def test_offline_scope_blocks_internet_and_restores_host(monkeypatch):
    calls = []
    def original(sock, address):
        calls.append(address)
        return 0
    monkeypatch.setattr(socket.socket, "connect", original)
    monkeypatch.setattr(socket.socket, "connect_ex", original)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with offline_assets_only():
        import os
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with socket.socket(socket.AF_INET) as sock:
            for function in (sock.connect, sock.connect_ex):
                with pytest.raises(AureoEngineError) as error:
                    function(("127.0.0.1", 9))
                assert error.value.code == "offline_asset_missing"
        if hasattr(socket, "AF_UNIX"):
            with socket.socket(socket.AF_UNIX) as sock:
                sock.connect("local-driver-socket")
    assert socket.socket.connect is original
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert calls == (["local-driver-socket"] if hasattr(socket, "AF_UNIX") else [])


def test_config_relative_paths_and_engine_specific_fields(tmp_path):
    path = tmp_path / "runtimes.json"
    path.write_text(json.dumps({"version": 1, "engines": {"chatterbox-multilingual": {"python": "env/python", "checkpoint_dir": "models", "device": "cpu"}}}))
    config = read_runtime_config(path)["chatterbox-multilingual"]
    assert config.python == tmp_path / "env/python" and config.checkpoint_dir == tmp_path / "models"
    value = json.loads(path.read_text())
    value["engines"]["chatterbox-multilingual"]["dtype"] = "float16"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_runtime_config(path)


@pytest.mark.parametrize("fields", [
    {"device": "auto"}, {"device": "https://gpu"}, {"dtype": "int4"},
    {"checkpoint_dir": "https://example.com/model"}, {"language": "invalid"}, {"speaker": " "},
])
def test_invalid_runtime_settings(fields):
    with pytest.raises(ValueError):
        LocalRuntimeConfig(**fields)
