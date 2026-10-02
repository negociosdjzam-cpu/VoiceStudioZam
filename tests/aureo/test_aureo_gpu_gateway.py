import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aureo.gpu.client import PersistentGPUClient
from aureo.gpu.config import GPUConfig
from aureo.gpu.gateway import Gateway
from aureo.gpu.http import create_app
from aureo.gpu.runpod import create_handler
from aureo.types import AureoEngineError
from test_aureo_gpu_worker import reference_bytes


@pytest.fixture
def fake_gpu_client(tmp_path):
    config = GPUConfig(tmp_path / "storage", tmp_path / "models", timeout_seconds=20)
    child = tmp_path / "test_model_child.py"
    child.write_text('''
import random
from types import SimpleNamespace
import aureo.gpu.worker as worker
from aureo.gpu.cache import CachedChatterboxRuntime
from aureo.adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
class Native:
    sr=24000
    device="cpu"
    conds=None
    def prepare_conditionals(self,path,exaggeration):
        self.conds=SimpleNamespace(t3=SimpleNamespace(energy=exaggeration),gen={})
    def generate(self,text,**kwargs):
        print("SDK progress goes to stderr, not JSON")
        return [[random.uniform(-0.1,0.1) for _ in range(2400)]]
class Host(worker.GPUHost):
    def __init__(self,config):
        native=CachedChatterboxRuntime(Native())
        native.provenance={"test_only":True,"loads":0}
        def factory():
            native.provenance["loads"]+=1
            return native
        super().__init__(config,engine=ChatterboxMultilingualAdapter(factory),gpu_check=lambda _: {"test_only":"CPU fake"},marker=lambda audio:(audio,{"test_only":True}))
worker.GPUHost=Host
raise SystemExit(worker.main())
''', encoding="utf-8")
    client = PersistentGPUClient(config, command=[sys.executable, str(child)])
    try:
        yield client
    finally:
        client.close()


def test_real_persistent_ipc_unicode_single_model_and_restart(fake_gpu_client):
    client = fake_gpu_client
    gateway = Gateway(client.config, client)
    ready = client.start()
    process = client.process
    data = {"text": "Hola, ¿cómo estás?", "reference_wav_base64": base64.b64encode(reference_bytes()).decode()}
    first = gateway.execute(data)
    second = gateway.execute({"text": "Mañana", "voice_id": first["reference_sha256"], "preset": "ENERGETICO"})
    assert first["success"] and second["success"] and client.process is process
    assert second["worker_after_run"]["successful_generations"] == 6
    assert second["worker_after_run"]["model_provenance"]["loads"] == 1
    assert client.request({"operation": "health"})["ready"]
    client.close()
    assert process.poll() is not None
    client.start()
    assert client.process.pid != process.pid


def test_actual_http_api_auth_upload_generate_download_and_benchmark(fake_gpu_client):
    from fastapi.testclient import TestClient
    config = fake_gpu_client.config
    token = "test-only-private-token-0001"
    gateway = Gateway(config, fake_gpu_client)
    app = create_app(config, gateway=gateway, token=token)
    headers = {"Authorization": "Bearer " + token}
    with TestClient(app) as api:
        assert api.get("/healthz").json()["alive"]
        assert api.get("/readyz").status_code == 200
        assert api.post("/v1/aureo/test", json={"text": "Hola"}).status_code == 401
        uploaded = api.post("/v1/aureo/reference", headers=headers, json={"reference_wav_base64": base64.b64encode(reference_bytes()).decode()})
        assert uploaded.status_code == 200
        payload = {"text": "Hola, ¿cómo estás?", "voice_id": uploaded.json()["voice_id"]}
        response = api.post("/v1/aureo/test", headers=headers, json={**payload, "include_audio": True})
        assert response.status_code == 200
        receipt = response.json()
        assert receipt["success"] and len(receipt["audio_wav_base64"]) == 3
        filename = receipt["results"][0]["wav"]
        url = f"/v1/aureo/runs/{receipt['run_id']}/{filename}"
        assert api.get(url).status_code == 401
        wav = api.get(url, headers=headers)
        assert wav.status_code == 200 and wav.content.startswith(b"RIFF")
        assert api.get(f"/v1/aureo/runs/{receipt['run_id']}/metrics.json", headers=headers).json()["run_id"] == receipt["run_id"]
        benchmark = api.post("/v1/aureo/benchmark", headers=headers, json=payload)
        assert benchmark.status_code == 200 and len(benchmark.json()["results"]) == 9
        for invalid in ({**payload, "speed": 1.2}, {**payload, "seeds": [1, 1, 2]}, {**payload, "operation": "benchmark"}, {**payload, "voice_id": "../private"}):
            assert api.post("/v1/aureo/test", headers=headers, json=invalid).status_code == 422
        assert api.post("/v1/aureo/reference", headers=headers, json={"reference_wav_base64": "not-base64"}).status_code == 422
        assert api.get(f"/v1/aureo/runs/{receipt['run_id']}/secret.json", headers=headers).status_code == 404
    assert fake_gpu_client.process is None


def test_pod_requires_token_and_readiness_tracks_dead_process(fake_gpu_client):
    from fastapi.testclient import TestClient
    with pytest.raises(ValueError):
        create_app(fake_gpu_client.config, token="")
    gateway = Gateway(fake_gpu_client.config, fake_gpu_client)
    with TestClient(create_app(fake_gpu_client.config, gateway=gateway, token="test-auth-token-123", preload=False)) as api:
        assert api.get("/readyz").status_code == 503
        fake_gpu_client.start()
        assert api.get("/readyz").status_code == 200
        fake_gpu_client.close()
        assert api.get("/readyz").status_code == 503


def test_runpod_handler_same_gateway_fetch_and_safe_errors(fake_gpu_client):
    gateway = Gateway(fake_gpu_client.config, fake_gpu_client)
    handler = create_handler(gateway)
    result = handler({"input": {"text": "Hola", "reference_wav_base64": base64.b64encode(reference_bytes()).decode(), "preset": "FIDELIDAD"}})
    assert result["success"] and len(result["results"]) == 3
    artifact = handler({"input": {"operation": "fetch", "run_id": result["run_id"], "filename": result["results"][0]["wav"]}})
    assert base64.b64decode(artifact["content_base64"]).startswith(b"RIFF")
    for bad in ({}, {"input": []}, {"input": {"text": "private"}}):
        error = handler(bad)
        assert error["error"]["code"] == "invalid_request" and "private" not in json.dumps(error)


def protocol_child(tmp_path, response):
    script = tmp_path / "protocol.py"
    script.write_text('''import json,sys,time
print(json.dumps({"version":1,"ready":{"ready":True}}),flush=True)
for line in sys.stdin:
    message=json.loads(line)
''' + response, encoding="utf-8")
    return [sys.executable, str(script)]


def test_ipc_timeout_kills_and_reaps_worker(tmp_path):
    config = GPUConfig(tmp_path / "s", tmp_path / "m", timeout_seconds=1)
    command = protocol_child(tmp_path, '    time.sleep(10)\n')
    client = PersistentGPUClient(config, command=command)
    client.start()
    process = client.process
    with pytest.raises(AureoEngineError) as error:
        client.request({"operation": "health"})
    assert error.value.code == "worker_timeout"
    assert process.poll() is not None and client.process is None


@pytest.mark.parametrize("reply", ['{"version":True,"id":message["id"],"result":{}}', '{"version":1,"id":"wrong","result":{}}', '{"version":1,"id":message["id"],"result":{},"audio":[]}', '{"version":1,"id":"wrong","error":{"code":"test","message":"test"}}'])
def test_malformed_reply_terminates_worker(tmp_path, reply):
    config = GPUConfig(tmp_path / "s", tmp_path / "m", timeout_seconds=10)
    client = PersistentGPUClient(config, command=protocol_child(tmp_path, f'    print(json.dumps({reply}),flush=True)\n'))
    client.start()
    process = client.process
    with pytest.raises(AureoEngineError) as error:
        client.request({"operation": "health"})
    assert error.value.code == "worker_error" and process.poll() is not None


def test_no_gpu_probe_and_installer_never_download_models(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "deploy/aureo/install.sh"
    dry = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True)
    assert dry.returncode == 0 and "no model downloads" in dry.stdout
    # An explicit empty CUDA visibility tests the actual CLI guard, not a stub.
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "AUREO_STORAGE_DIR": str(tmp_path / "s")}
    probe = subprocess.run([sys.executable, "-m", "aureo.gpu", "probe"], env=environment, capture_output=True, text=True)
    assert probe.returncode == 2 and "gpu_unavailable" in probe.stderr
    assert not (tmp_path / "s/models").exists()


def test_network_gateway_imports_neither_torch_nor_native_sdks():
    script = 'import sys; import aureo.gpu.gateway,aureo.gpu.http,aureo.gpu.runpod; assert not any(name in sys.modules for name in ("torch","chatterbox","runpod","services.watermark"))'
    subprocess.run([sys.executable, "-c", script], check=True)
