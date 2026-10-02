import json
import subprocess
import sys
from types import SimpleNamespace

from aureo import AureoEngineRegistry, AureoVoiceProfile
from aureo.cli import main


def test_default_cli_is_offline_and_returns_unconfigured_errors(tmp_path):
    listing = subprocess.run([sys.executable, "-m", "aureo", "list"], capture_output=True, text=True, check=True)
    assert len(json.loads(listing.stdout)["engines"]) == 3
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"local audio")
    benchmark = subprocess.run([
        sys.executable, "-m", "aureo", "benchmark", "--text", "Hola",
        "--reference", str(reference), "--seed", "42",
    ], capture_output=True, text=True)
    assert benchmark.returncode == 1
    results = json.loads(benchmark.stdout)["results"]
    assert len(results) == 3
    assert all(result["errors"][0]["code"] == "runtime_not_configured" for result in results)
    assert all(result["time_to_first_audio_seconds"] is None for result in results)
    assert list(tmp_path.iterdir()) == [reference]


def test_cli_profile_dispatch_and_sdk_stdout_do_not_corrupt_json(tmp_path, make_engine, monkeypatch, capsys):
    registry = AureoEngineRegistry()
    def noisy(request):
        print("SDK progress")
    registry.register(make_engine("fast", hook=noisy))
    registry.register(make_engine("pro"))
    monkeypatch.setitem(sys.modules, "local_aureo_factory", SimpleNamespace(build=lambda: registry))
    path = tmp_path / "voice.json"
    AureoVoiceProfile("voice", "pro", mode_engines={"FAST": "fast"}, seed=7).write(path)
    status = main([
        "--registry-factory", "local_aureo_factory:build", "benchmark",
        "--profile", str(path), "--mode", "FAST", "--text", "Hola",
    ])
    output = capsys.readouterr()
    assert status == 0 and "SDK progress" in output.err
    results = json.loads(output.out)["results"]
    assert len(results) == 1 and results[0]["engine_id"] == "fast"
    request = registry.get("fast").requests[0]
    assert request.seed == 7 and request.mode.value == "FAST"
    assert registry.get("pro").loads == 0


def test_cli_reports_configuration_error_without_leaking_details(capsys):
    status = main(["benchmark", "--text", "Hola", "--reference-text", "Private transcript"])
    output = capsys.readouterr()
    assert status == 2 and not output.out
    assert json.loads(output.err) == {"error": "lab_request_failed", "type": "ValueError"}
