"""CPU infrastructure tests; native speech quality/performance is not claimed."""
import base64
from dataclasses import replace
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import wave

import pytest

from aureo.adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
from aureo.gpu.cache import CachedChatterboxRuntime
from aureo.gpu.config import GPUConfig
from aureo.gpu.export import artifact_path, with_inline_audio, write_wav
from aureo.gpu.provision import MODEL_FILES, provision, verify_model
from aureo.gpu.reference import save_reference, voice_path
from aureo.gpu.worker import GPUHost, validate_payload
from aureo.types import AureoEngineError


def reference_bytes(value=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes([value, 0]) * 48000)
    return output.getvalue()


@pytest.fixture
def gpu_config(tmp_path):
    return GPUConfig(tmp_path / "storage", tmp_path / "models")


class Native:
    sr = 24000
    device = "cpu"

    def __init__(self):
        self.conds = None
        self.preparations = 0
        self.calls = []
        self.fail = False

    def prepare_conditionals(self, path, exaggeration):
        self.preparations += 1
        random.random()  # prove preparation's RNG does not perturb a seed
        self.conds = SimpleNamespace(t3=SimpleNamespace(energy=exaggeration), gen={"reference": Path(path).read_bytes()[:16]})

    def generate(self, text, **options):
        import torch
        self.calls.append(options)
        assert options["audio_prompt_path"] is None
        assert self.conds is not None
        if self.fail:
            raise RuntimeError("PRIVATE-SDK-TOKEN")
        self.conds.t3.energy = -1  # cache containers must not be corrupted
        return torch.tensor([[random.uniform(-0.2, 0.2) for _ in range(2400)]])


@pytest.fixture
def host(gpu_config):
    native = Native()
    runtime = CachedChatterboxRuntime(native, capacity=3)
    loads = []
    def factory():
        loads.append(1)
        return runtime
    engine = ChatterboxMultilingualAdapter(factory)
    marking = []
    def marker(chunk):
        marking.append(chunk)
        return chunk, {"native_perth": False, "audioseal_applied": False, "test_only": True}
    worker = GPUHost(gpu_config, engine=engine, gpu_check=lambda config: {"test_only": "CPU fake"}, marker=marker)
    worker.test_native, worker.test_loads, worker.test_marking = native, loads, marking
    return worker


def payload(config, **overrides):
    voice_id = save_reference(config, reference_bytes())
    return {"text": "Hola, ¿cómo estás?", "voice_id": voice_id, **overrides}


def test_benchmark_nine_wavs_three_seeds_single_load(host, gpu_config):
    host.preload()
    result = host.execute(payload(gpu_config, operation="benchmark"))
    assert result["success"] and len(result["results"]) == 9
    assert len(host.test_loads) == 1
    assert host.test_native.preparations == 3
    assert len(host.test_marking) == 9
    assert host.test_native.conds is None
    assert {row["style"] for row in result["results"]} == {"NATURAL", "ENERGÉTICO", "FIDELIDAD"}
    assert sum(row["first_inference"] for row in result["results"]) == 1
    assert sum(row["reference_cache_hit"] for row in result["results"]) == 6
    directory = gpu_config.results / result["run_id"]
    assert json.loads((directory / "metrics.json").read_text())["run_id"] == result["run_id"]
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "Persistent worker" in report and "First PCM is measured after preload" in report
    assert "cold engine per trial" not in report
    for style in ("NATURAL", "ENERGÉTICO", "FIDELIDAD"):
        rows = [row for row in result["results"] if row["style"] == style]
        assert {row["seed"] for row in rows} == {42, 43, 44}
        assert len({artifact_path(gpu_config, result["run_id"], row["wav"]).read_bytes() for row in rows}) == 3
    for row in result["results"]:
        assert row["load_seconds"] == 0
        assert row["time_to_first_audio_seconds"] >= row["generation_seconds"] > 0
        assert row["total_seconds"] >= row["generation_seconds"]
        assert row["generated_duration_seconds"] == 0.1
        assert row["memory"]["ram_peak_bytes"] > 0
        assert row["real_time_factor"] > 0
        with wave.open(str(directory / row["wav"])) as audio:
            assert audio.getnframes() == 2400 and audio.getsampwidth() == 2
    # Same voice/text/seed: warm cache remains deterministic, without reloading.
    repeated = host.execute(payload(gpu_config, operation="benchmark"))
    assert len(host.test_loads) == 1 and host.test_native.preparations == 3
    for before, after in zip(result["results"], repeated["results"]):
        assert artifact_path(gpu_config, result["run_id"], before["wav"]).read_bytes() == artifact_path(gpu_config, repeated["run_id"], after["wav"]).read_bytes()


def test_reference_cache_separates_voice_energy_and_evicts(host, gpu_config):
    for value in (1, 2):
        voice_id = save_reference(gpu_config, reference_bytes(value))
        host.execute({"operation": "benchmark", "text": "Hola", "voice_id": voice_id})
    assert host.test_native.preparations == 6
    assert host.engine._runtime.stats()["entries"] == 3


def test_failed_generation_restores_voice_and_persists_errors(host, gpu_config):
    host.test_native.fail = True
    result = host.execute(payload(gpu_config))
    assert not result["success"] and len(result["results"]) == 3
    assert host.test_native.conds is None and len(host.test_loads) == 1
    assert all(row["wav"] is None and row["real_time_factor"] is None for row in result["results"])
    assert "PRIVATE-SDK-TOKEN" not in json.dumps(result)
    assert not list((gpu_config.results / result["run_id"]).glob("*.wav"))


@pytest.mark.parametrize("controls", [{"speed": 1}, {"expression": "energetic"}, {"streaming": True}])
def test_explicit_unsupported_controls_fail_before_loading(host, gpu_config, controls):
    with pytest.raises(AureoEngineError):
        host.execute(payload(gpu_config, **controls))
    assert not host.test_loads


@pytest.mark.parametrize("overrides", [{"seeds": [1, 1, 2]}, {"seeds": [True, 2, 3]}, {"seeds": [1, 2]}, {"seeds": [-1, 2, 3]}, {"seeds": [1, 2, 2**32]}, {"text": ""}, {"text": "x" * 2001}, {"operation": "other"}, {"include_audio": 1}, {"unknown": 2}, {"preset": "wrong"}])
def test_invalid_gpu_inputs(gpu_config, overrides):
    with pytest.raises((ValueError, TypeError)):
        validate_payload(gpu_config, {"text": "Hola", **overrides})


def test_content_addressed_reference_dedup_integrity_and_bounds(gpu_config):
    first = save_reference(gpu_config, reference_bytes())
    assert first == save_reference(gpu_config, reference_bytes())
    assert len(list(gpu_config.references.glob("*.wav"))) == 1
    voice_path(gpu_config, first).write_bytes(reference_bytes(2))
    with pytest.raises(ValueError, match="integrity"):
        voice_path(gpu_config, first)
    for bad in (b"bad", b"", b"x" * (gpu_config.max_reference_bytes + 1)):
        with pytest.raises(ValueError):
            save_reference(gpu_config, bad)
    with pytest.raises(ValueError):
        voice_path(gpu_config, "../../private")


def test_inline_audio_limits_and_no_artifact_traversal(host, gpu_config):
    result = host.execute(payload(gpu_config))
    inline = with_inline_audio(gpu_config, result)
    assert len(inline["audio_wav_base64"]) == 3
    assert all(base64.b64decode(value).startswith(b"RIFF") for value in inline["audio_wav_base64"].values())
    with pytest.raises(ValueError):
        with_inline_audio(replace(gpu_config, max_inline_bytes=1), result)
    for run_id, filename in (("../../private", "metrics.json"), (result["run_id"], "../secret"), (result["run_id"], "NATURAL-seed999.wav")):
        with pytest.raises(ValueError):
            artifact_path(gpu_config, run_id, filename)


def test_failed_export_does_not_publish_wav(host, gpu_config):
    def failed_marker(chunk):
        raise RuntimeError("private")
    host.marker = failed_marker
    result = host.execute(payload(gpu_config))
    assert all(row["errors"][0]["stage"] == "export" for row in result["results"])
    assert not list((gpu_config.results / result["run_id"]).glob("*.wav"))


def test_provision_requires_gpu_before_any_weight_download(gpu_config):
    downloads = []
    def no_gpu(config):
        raise AureoEngineError("gpu_unavailable", "No GPU")
    with pytest.raises(AureoEngineError):
        provision(gpu_config, "a" * 40, gpu_check=no_gpu, downloader=lambda **kw: downloads.append(kw))
    assert not downloads and not gpu_config.model_dir.exists()


def test_pinned_provisioning_repeatability_resume_and_checksums(gpu_config):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        for name in MODEL_FILES:
            (gpu_config.model_dir / name).write_bytes(b"dummy fixture, not a model")
    first = provision(gpu_config, "a" * 40, gpu_check=lambda config: None, downloader=download)
    assert provision(gpu_config, "a" * 40, gpu_check=lambda config: None, downloader=download) == first
    assert len(calls) == 1 and calls[0]["revision"] == "a" * 40
    assert verify_model(gpu_config) == first
    with pytest.raises(ValueError):
        provision(gpu_config, "main", gpu_check=lambda config: None, downloader=download)
    (gpu_config.model_dir / "ve.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_model(gpu_config)


@pytest.mark.parametrize("settings", [{"AUREO_DEVICE": "cpu"}, {"AUREO_MIN_VRAM_GIB": "2"}, {"AUREO_REFERENCE_CACHE_ENTRIES": "0"}, {"AUREO_JOB_TIMEOUT_SECONDS": "-1"}, {"AUREO_WORKER_PYTHON": "http://remote"}, {"AUREO_WORKER_PYTHON": "python -m anything"}])
def test_gpu_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        GPUConfig.from_env(settings)


def test_export_invokes_canonical_mark_synthetic(monkeypatch, tmp_path):
    import torch
    from aureo.gpu.export import mark_audio
    from aureo.types import AudioChunk
    import services.watermark as watermark
    calls = []
    def mark(waveform, sample_rate, **kwargs):
        calls.append((sample_rate, kwargs))
        return waveform + 0.01
    monkeypatch.setattr(watermark, "mark_synthetic", mark)
    marked, status = mark_audio(AudioChunk((0.1, 0.2), 24000))
    assert calls == [(24000, {"context": "aureo.gpu.export", "force": True})]
    assert status["audioseal_applied"]
    assert marked.samples == pytest.approx((0.11, 0.21))
