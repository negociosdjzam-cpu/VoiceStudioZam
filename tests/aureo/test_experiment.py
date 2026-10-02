import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aureo import AureoEngineRegistry, SynthesisRequest, VoiceReference
from aureo.benchmark import AureoBenchmark
from aureo.experiment import BenchmarkExperiment
from aureo.metrics import MemoryStats
from aureo.presets import BenchmarkStyle, request_from_dict, style_request
from aureo.runtimes import ENGINE_IDS, LocalRuntimeConfig, create_benchmark_registry


class NoMemory:
    def start(self):
        pass
    def stop(self):
        return MemoryStats()


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "private-recording.wav"
    path.write_bytes(b"identical voice reference")
    return VoiceReference(path, "Private reference transcript")


@pytest.fixture
def trial_runner(clock, make_engine):
    def run(payload, config, timeout):
        registry = AureoEngineRegistry()
        registry.register(make_engine(payload["engine_id"]))
        result = AureoBenchmark(registry, clock=clock, memory_factory=NoMemory).run(request_from_dict(payload["request"]), [payload["engine_id"]])[0]
        return {"version": 1, "result": result.to_dict(), "runtime": {"fixture": True}}
    return run


def test_presets_negotiate_explicitly_and_caller_overrides_remain_strict(reference):
    caps = create_benchmark_registry().get("qwen3-tts-1.7b-base").capabilities
    base = SynthesisRequest("Hola", reference=reference, seed=42)
    effective, requested, omitted = style_request(base, "ENERGETICO", caps)
    assert requested["energy"] == 0.85 and requested["speed"] == 1.1
    assert effective.variation == 0.65 and effective.energy is None
    assert set(omitted) == {"energy", "expression", "fidelity", "speed"}
    explicit, _, omitted = style_request(SynthesisRequest("Hola", reference=reference, energy=0.9), "NATURAL", caps)
    assert explicit.energy == 0.9 and "energy" not in omitted
    strict, _, omitted = style_request(base, "FIDELIDAD", caps, strict=True)
    assert strict.fidelity == 1 and not omitted
    assert BenchmarkStyle.parse("ENERGETICO") is BenchmarkStyle.ENERGETICO


def test_same_reference_and_text_across_all_styles_repeats_and_engines(reference, trial_runner):
    observations = []
    original = reference.path.read_bytes()
    def run(payload, config, timeout):
        path = Path(payload["request"]["reference"]["path"])
        observations.append((path, path.read_bytes(), payload["request"]["text"], payload["request"]["seed"]))
        reference.path.write_bytes(b"replacement while trial is running")
        return trial_runner(payload, config, timeout)
    receipt = BenchmarkExperiment(runner=run).run(SynthesisRequest("Mismo texto", reference=reference, seed=9), repeats=2)
    assert len(receipt["results"]) == 18
    assert all(row["success"] for row in receipt["results"])
    assert {item[0] for item in observations} == {observations[0][0]}
    assert all(item[1:] == (original, "Mismo texto", 9) for item in observations)
    assert not observations[0][0].exists()
    assert receipt["reference_sha256"] == hashlib.sha256(original).hexdigest()
    assert {row["style"] for row in receipt["results"]} == {style.value for style in BenchmarkStyle}
    assert all(row["time_to_first_audio_seconds"] == 5 and row["total_seconds"] == 6 for row in receipt["results"])
    encoded = json.dumps(receipt)
    assert "Mismo texto" not in encoded and "Private reference transcript" not in encoded and "private-recording" not in encoded


def test_one_config_selects_only_one_engine_by_default(reference, trial_runner):
    receipt = BenchmarkExperiment({"chatterbox-multilingual": LocalRuntimeConfig(device="cpu")}, runner=trial_runner).run(SynthesisRequest("Hola", reference=reference))
    assert len(receipt["results"]) == 3
    assert {row["engine_id"] for row in receipt["results"]} == {"chatterbox-multilingual"}


def test_worker_failure_does_not_stop_other_engines(reference, trial_runner):
    def run(payload, config, timeout):
        if payload["engine_id"] == "qwen3-tts-1.7b-base":
            raise subprocess.TimeoutExpired("private-command-text", timeout)
        return trial_runner(payload, config, timeout)
    receipt = BenchmarkExperiment(runner=run).run(SynthesisRequest("Hola", reference=reference))
    failed = [row for row in receipt["results"] if not row["success"]]
    assert len(failed) == 3 and all(row["errors"][0]["code"] == "worker_timeout" for row in failed)
    assert all(row["real_time_factor"] is None for row in failed)
    assert sum(row["success"] for row in receipt["results"]) == 6
    assert "private-command-text" not in json.dumps(receipt)


@pytest.mark.parametrize("mutation", ["wrong_hash", "wrong_engine", "no_audio", "nan", "extra_audio", "inconsistent_errors", "negative_memory"])
def test_invalid_worker_reply_never_claims_success(reference, trial_runner, mutation):
    def run(payload, config, timeout):
        reply = copy.deepcopy(trial_runner(payload, config, timeout))
        result = reply["result"]
        if mutation == "wrong_hash":
            result["reference_sha256"] = "different reference"
        elif mutation == "wrong_engine":
            result["engine_id"] = "another engine"
        elif mutation == "no_audio":
            result["chunks"] = 0
        elif mutation == "nan":
            result["total_seconds"] = float("nan")
        elif mutation == "extra_audio":
            result["samples"] = [0.1]
        elif mutation == "inconsistent_errors":
            result["errors"] = [{"stage": "load", "code": "failure", "message": "failed"}]
        elif mutation == "negative_memory":
            result["memory"]["ram_peak_bytes"] = -1
        return reply
    receipt = BenchmarkExperiment(runner=run).run(SynthesisRequest("Hola", reference=reference), engine_ids=[ENGINE_IDS[0]], styles=["NATURAL"])
    row = receipt["results"][0]
    assert not row["success"] and row["errors"][0]["code"] == "worker_error"


def test_unreadable_shared_reference_does_not_spawn_workers(tmp_path):
    def run(*args):
        pytest.fail("Must not spawn a worker for unreadable input")
    receipt = BenchmarkExperiment(runner=run).run(SynthesisRequest("Hola", reference=VoiceReference(tmp_path / "missing.wav")))
    assert len(receipt["results"]) == 9
    assert all(row["errors"][0]["code"] == "reference_unreadable" for row in receipt["results"])


def test_real_worker_timeout_kills_and_reaps_process(reference, monkeypatch):
    processes = []
    original = subprocess.Popen
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr("aureo.experiment.subprocess.Popen", spawn)
    receipt = BenchmarkExperiment(timeout=0.001).run(SynthesisRequest("Hola", reference=reference), engine_ids=[ENGINE_IDS[0]], styles=["NATURAL"])
    assert receipt["results"][0]["errors"][0]["code"] == "worker_timeout"
    assert processes and all(process.poll() is not None for process in processes)


def test_real_process_pipeline_native_sdk_contract_without_model_weights(tmp_path, monkeypatch, reference):
    # A fake SDK in its own import directory exercises real subprocess IPC,
    # local loading and cleanup. This is explicitly not model inference.
    sdk_root = tmp_path / "sdk"
    package = sdk_root / "chatterbox"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "mtl_tts.py").write_text('''
import os
class ChatterboxMultilingualTTS:
    sr = 24000
    device = "cpu"
    conds = None
    @classmethod
    def from_local(cls, path, device):
        print("SDK fixture loading")
        return cls()
    def generate(self, text, **kwargs):
        assert kwargs["language_id"] == "es"
        return [[0.1] * 2400]
''')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(sdk_root), os.environ.get("PYTHONPATH")))))
    # Simulate a host configured with ASCII pipes; worker IPC must preserve
    # Unicode text on Windows and other non-UTF8 environments.
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    weights = tmp_path / "models"
    weights.mkdir()
    for name in ("ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"):
        (weights / name).write_bytes(b"fixture, not actual weights")
    config = LocalRuntimeConfig(python=Path(sys.executable), checkpoint_dir=weights, device="cpu")
    text = "Hola, ¿cómo estás? Mañana probaremos otra voz."
    receipt = BenchmarkExperiment({ENGINE_IDS[0]: config}, timeout=45).run(SynthesisRequest(text, reference=reference, language="es", seed=42), styles=["NATURAL"])
    row = receipt["results"][0]
    assert row["success"], row["errors"]
    assert row["generated_duration_seconds"] == 0.1 and row["sample_rate"] == 24000
    assert row["load_seconds"] > 0 and row["time_to_first_audio_seconds"] >= row["load_seconds"]
    assert row["process_wall_seconds"] >= row["total_seconds"]
    assert row["runtime"]["device"] == "cpu" and row["runtime"]["gpu_names"] == []
    assert row["memory"]["ram_peak_bytes"] > 0 and row["memory"]["vram_peak_bytes"] is None
    assert "SDK fixture loading" not in json.dumps(receipt)
    assert row["text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert os.environ["PYTHONIOENCODING"] == "ascii"


@pytest.mark.parametrize("options", [{"repeats": 0}, {"repeats": True}, {"styles": []}, {"engine_ids": []}, {"engine_ids": [ENGINE_IDS[0]] * 2}])
def test_experiment_argument_validation(reference, options):
    with pytest.raises(ValueError):
        BenchmarkExperiment().run(SynthesisRequest("Hola", reference=reference), **options)
