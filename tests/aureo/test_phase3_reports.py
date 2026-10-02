import json
import subprocess
import sys

import pytest

from aureo import AureoVoiceProfile, SynthesisRequest, VoiceReference
from aureo.cli import main
from aureo.experiment import BenchmarkExperiment
from aureo.report import render_report, save_receipt


@pytest.fixture
def failed_receipt(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"reference fixture")
    def failed(*args):
        raise RuntimeError("secret-native-error")
    return BenchmarkExperiment(runner=failed).run(SynthesisRequest("Private text", reference=VoiceReference(path, "Private transcript")))


@pytest.mark.parametrize("controls", [
    {"fidelity": -0.1}, {"fidelity": 1.1}, {"fidelity": float("nan")}, {"fidelity": True},
    {"speed": 0}, {"speed": 4.1}, {"speed": float("inf")}, {"speed": True},
])
def test_new_controls_are_validated(controls):
    with pytest.raises(ValueError):
        SynthesisRequest("Hola", **controls)


def test_profile_new_controls_round_trip_and_old_format_compatibility(tmp_path):
    profile = AureoVoiceProfile("voice", "openvoice-v2", fidelity=0.9, speed=1.2)
    path = tmp_path / "profile.json"
    profile.write(path)
    restored = AureoVoiceProfile.read(path)
    assert restored == profile and restored.request("Hola").speed == 1.2
    old = profile.to_dict()
    old.pop("fidelity")
    old.pop("speed")
    restored_old = AureoVoiceProfile.from_dict(old)
    assert restored_old.fidelity is None and restored_old.speed is None


def test_json_and_markdown_preserve_errors_and_no_fabricated_quality(tmp_path, failed_receipt):
    output, report = save_receipt(failed_receipt, tmp_path / "results.json")
    assert json.loads(output.read_text()) == failed_receipt
    text = report.read_text()
    assert "NATURAL" in text and "ENERGÉTICO" in text and "FIDELIDAD" in text
    assert "0/1" in text and "worker/worker_error" in text
    assert "FAST, PRO and ULTRA remain unassigned" in text
    assert "Private text" not in text and "Private transcript" not in text and "secret-native-error" not in text
    assert "—" in text
    with pytest.raises(FileExistsError):
        save_receipt(failed_receipt, output)
    save_receipt(failed_receipt, output, overwrite=True)


def test_stage_failure_keeps_previous_results_and_cleans_tempfiles(tmp_path, failed_receipt, monkeypatch):
    output, report = save_receipt(failed_receipt, tmp_path / "result.json")
    before = output.read_bytes(), report.read_bytes()
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("aureo.report.os.fsync", fail)
    with pytest.raises(OSError):
        save_receipt(failed_receipt, output, overwrite=True)
    assert (output.read_bytes(), report.read_bytes()) == before
    assert {path.name for path in tmp_path.iterdir()} == {"result.json", "result.md", "ref.wav"}


def test_results_cannot_overwrite_inputs_or_model_directories(tmp_path, failed_receipt):
    reference = tmp_path / "ref.wav"
    before = reference.read_bytes()
    with pytest.raises(ValueError):
        save_receipt(failed_receipt, reference, overwrite=True, protected_paths=[reference])
    assert reference.read_bytes() == before
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(ValueError):
        save_receipt(failed_receipt, models / "config.json", protected_paths=[models])
    with pytest.raises(ValueError):
        save_receipt(failed_receipt, tmp_path / "results.md")


def test_cli_experiment_real_unconfigured_workers_write_both_artifacts(tmp_path, capsys):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"local fixture")
    output = tmp_path / "results.json"
    status = main(["experiment", "--text", "Hola", "--reference", str(reference), "--output", str(output)])
    receipt = json.loads(capsys.readouterr().out)
    assert status == 1 and len(receipt["results"]) == 9
    assert all(row["errors"][0]["code"] == "runtime_not_configured" for row in receipt["results"])
    assert json.loads(output.read_text()) == receipt and output.with_suffix(".md").is_file()
    assert reference.read_bytes() == b"local fixture"


def test_cli_input_overwrite_rejected_before_worker_launch(tmp_path, capsys, monkeypatch):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"preserve me")
    monkeypatch.setattr("aureo.experiment.BenchmarkExperiment.run", lambda *args, **kwargs: pytest.fail("Preflight must reject first"))
    status = main(["experiment", "--text", "Hola", "--reference", str(reference), "--output", str(reference), "--overwrite"])
    assert status == 2 and reference.read_bytes() == b"preserve me"
    assert json.loads(capsys.readouterr().err)["error"] == "lab_request_failed"


def test_native_catalog_in_fresh_process_imports_no_sdks_or_torch():
    script = '''
import sys, json
from aureo.runtimes import create_benchmark_registry
print(json.dumps({"engines": create_benchmark_registry().describe(), "modules": list(sys.modules)}))
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    value = json.loads(completed.stdout)
    assert [engine["id"] for engine in value["engines"]] == ["chatterbox-multilingual", "qwen3-tts-1.7b-base", "openvoice-v2"]
    assert not any(module.split(".")[0] in {"torch", "numpy", "chatterbox", "qwen_tts", "openvoice", "melo", "services", "omnivoice"} for module in value["modules"])
    assert all(not engine["loaded"] and not engine["configured"] for engine in value["engines"])
