"""One isolated, offline, cold native-engine trial. Private input via stdin."""
from contextlib import redirect_stdout
import importlib.metadata
import json
import platform
import sys

from .benchmark import AureoBenchmark
from .presets import request_from_dict
from .runtimes import ENGINE_IDS, MODEL_IDS, LocalRuntimeConfig, create_benchmark_registry
from .runtimes.offline import offline_assets_only

SDK_DISTRIBUTIONS = ("chatterbox-tts", "qwen-tts", "OpenVoice", "MeloTTS", "wavmark", "transformers")


def runtime_metadata(engine_id, config):
    versions = {}
    for name in SDK_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    torch = sys.modules.get("torch")
    gpu = []
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except (AttributeError, RuntimeError, OSError):
            pass
    return {
        "model": MODEL_IDS[engine_id], "python": platform.python_version(),
        "platform": sys.platform, "device": None if config is None else config.device,
        "dtype_policy": None if config is None else config.dtype,
        "packages": versions, "torch": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "gpu_names": gpu, "execution": "isolated_native_worker",
    }


def execute(payload):
    if not isinstance(payload, dict) or set(payload) != {"version", "engine_id", "config", "request"} or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported worker protocol")
    engine_id = payload["engine_id"]
    if engine_id not in ENGINE_IDS:
        raise ValueError("Unknown native engine")
    request = request_from_dict(payload["request"])
    config = None if payload["config"] is None else LocalRuntimeConfig.from_dict(payload["config"])
    with offline_assets_only():
        registry = create_benchmark_registry({engine_id: config} if config is not None else {})
        result = AureoBenchmark(registry).run(request, [engine_id], cold=True)[0]
    return {"version": 1, "result": result.to_dict(), "runtime": runtime_metadata(engine_id, config)}


def main():
    try:
        payload = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            response = execute(payload)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False))
        return 0 if response["result"]["success"] else 1
    except Exception as error:
        print(json.dumps({"error": "worker_request_failed", "type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
