"""Persistent GPU host. JSON-line IPC; model and audio never enter the gateway."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from ..adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
from ..benchmark import BenchmarkResult, error_report
from ..metrics import MemorySampler
from ..presets import BenchmarkStyle, style_request
from ..runtimes.chatterbox import load_chatterbox
from ..runtimes.config import LocalRuntimeConfig
from ..runtimes.offline import offline_assets_only
from ..types import SynthesisRequest, VoiceReference
from .cache import CachedChatterboxRuntime
from .config import GPUConfig, verify_gpu
from .export import mark_audio, write_wav
from .reference import resolve_reference
from .report import render_report


def native_engine(config):
    def factory():
        from unittest.mock import patch
        from ..runtimes.common import torch_for_device
        from .provision import verify_model
        provenance = verify_model(config)
        from chatterbox.models.tokenizers import tokenizer
        original_hub = tokenizer.hf_hub_download
        def local_mapping(*args, **kwargs):
            if kwargs.get("repo_id") == "ResembleAI/chatterbox" and kwargs.get("filename") == "Cangjie5_TC.json":
                return str(config.model_dir / "Cangjie5_TC.json")
            return original_hub(*args, **kwargs)
        # SDK 0.1.7 otherwise requests a Hub cache lookup for this already-local
        # dictionary. Keep the pinned local asset and the worker offline guard.
        with patch.object(tokenizer, "hf_hub_download", local_mapping):
            native = load_chatterbox(LocalRuntimeConfig(device=config.device, checkpoint_dir=config.model_dir))
        torch = torch_for_device(config.device)
        synchronize = lambda: torch.cuda.synchronize(config.device)
        synchronize()
        runtime = CachedChatterboxRuntime(native, capacity=config.cache_entries, synchronize=synchronize)
        runtime.provenance = provenance
        return runtime
    return ChatterboxMultilingualAdapter(factory)


def validate_payload(config, payload):
    allowed = {"operation", "text", "language", "voice_id", "reference_wav_base64", "preset", "seeds", "variation", "energy", "fidelity", "expression", "speed", "streaming", "include_audio"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown GPU request fields")
    operation = payload.get("operation", "synthesize")
    if operation not in ("synthesize", "benchmark"):
        raise ValueError("Unknown synthesis operation")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > config.max_text_chars:
        raise ValueError("Text must be non-empty and within the configured limit")
    seeds = payload.get("seeds", [42, 43, 44])
    if not isinstance(seeds, list) or len(seeds) != 3 or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds) or len(set(seeds)) != 3:
        raise ValueError("Supply exactly three different unsigned 32-bit seeds")
    if "include_audio" in payload and type(payload["include_audio"]) is not bool:
        raise ValueError("include_audio must be a boolean")
    style = BenchmarkStyle.parse(payload.get("preset", "NATURAL"))
    base = SynthesisRequest(text, language=payload.get("language", "es"), seed=seeds[0],
                            **{key: payload[key] for key in ("variation", "energy", "fidelity", "expression", "speed", "streaming") if key in payload})
    return base, seeds, tuple(BenchmarkStyle) if operation == "benchmark" else (style,)


class GPUHost:
    def __init__(self, config, *, engine=None, gpu_check=verify_gpu, marker=mark_audio, memory_factory=MemorySampler):
        self.config = config
        self.engine = native_engine(config) if engine is None else engine
        self.gpu_check = gpu_check
        self.marker = marker
        self.memory_factory = memory_factory
        self.load_seconds = None
        self.load_memory = None
        self.hardware = None
        self.generations = 0

    def preload(self):
        if self.engine.loaded:
            return
        started = time.perf_counter()
        sampler = self.memory_factory()
        sampler.start()
        try:
            self.hardware = self.gpu_check(self.config)
            with offline_assets_only():
                self.engine.load()
            self.load_seconds = time.perf_counter() - started
        except Exception:
            self.engine.unload()
            raise
        finally:
            self.load_memory = asdict(sampler.stop())

    def status(self):
        runtime = getattr(self.engine, "_runtime", None)
        return {"ready": self.engine.loaded, "engine_id": self.engine.engine_id,
                "load_seconds": self.load_seconds, "load_memory": self.load_memory,
                "hardware": self.hardware, "successful_generations": self.generations,
                "model_provenance": getattr(runtime, "provenance", None),
                "reference_cache": runtime.stats() if isinstance(runtime, CachedChatterboxRuntime) else None}

    def execute(self, payload):
        base, seeds, styles = validate_payload(self.config, payload)
        voice_id, reference = resolve_reference(self.config, payload)
        from dataclasses import replace
        base = replace(base, reference=VoiceReference(reference))
        requests = []
        for style in styles:
            effective, requested, omitted = style_request(base, style, self.engine.capabilities)
            self.engine.validate(effective)  # explicit unsupported controls fail before loading
            for seed in seeds:
                requests.append((style, replace(effective, seed=seed), requested, omitted))
        started = time.perf_counter()
        self.preload()
        run_id = uuid.uuid4().hex
        directory = self.config.results / run_id
        directory.mkdir(parents=True, exist_ok=False)
        receipt = {"version": 1, "run_id": run_id, "engine_id": self.engine.engine_id,
                   "text_sha256": hashlib.sha256(base.text.encode("utf-8")).hexdigest(),
                   "reference_sha256": voice_id, "language": base.language,
                   "model_startup": self.status(), "cold": False, "persistent_worker": True,
                   "results": []}
        for style, request, requested, omitted in requests:
            result = BenchmarkResult(self.engine.engine_id, request.mode.value, False)
            result.text_sha256, result.reference_sha256 = receipt["text_sha256"], voice_id
            beginning = time.perf_counter()
            sampler = self.memory_factory()
            sampler.start()
            filename, marking = None, None
            export_seconds = None
            first_inference = self.generations == 0
            stage = "generate"
            try:
                with self.engine.session(), offline_assets_only():
                    generation = time.perf_counter()
                    chunk = self.engine.generate(request)
                    result.generation_seconds = time.perf_counter() - generation
                    result.time_to_first_audio_seconds = time.perf_counter() - beginning
                    result.generation_time_to_first_audio_seconds = result.generation_seconds
                    result.chunks, result.sample_rate = 1, chunk.sample_rate
                    result.generated_duration_seconds = chunk.duration_seconds
                    stage = "export"
                    export = time.perf_counter()
                    marked, marking = self.marker(chunk)
                    filename = f"{style.name}-seed{request.seed}.wav"
                    write_wav(marked, directory / filename)
                    export_seconds = time.perf_counter() - export
                    self.generations += 1
            except Exception as error:
                result.errors.append(error_report(stage, error))
                if stage == "generate":
                    result.generation_seconds = time.perf_counter() - beginning
                if filename is not None:
                    (directory / filename).unlink(missing_ok=True)
                filename = None
            finally:
                result.memory = sampler.stop()
                result.total_seconds = time.perf_counter() - beginning
            if result.success:
                result.real_time_factor = result.generation_seconds / result.generated_duration_seconds
                result.end_to_end_real_time_factor = result.total_seconds / result.generated_duration_seconds
            runtime = getattr(self.engine, "_runtime", None)
            receipt["results"].append({**result.to_dict(), "style": style.value, "trial": seeds.index(request.seed) + 1,
                "seed": request.seed, "first_inference": first_inference, "wav": filename,
                "export_seconds": export_seconds, "marking": marking,
                "requested_controls": requested, "effective_controls": {key: getattr(request, key) for key in requested},
                "omitted_controls": omitted,
                "reference_cache_hit": runtime.last_hit if isinstance(runtime, CachedChatterboxRuntime) else None,
                "reference_prepare_seconds": runtime.last_prepare_seconds if isinstance(runtime, CachedChatterboxRuntime) else None})
        receipt["batch_seconds"] = time.perf_counter() - started
        receipt["worker_after_run"] = self.status()
        receipt["success"] = all(row["success"] for row in receipt["results"])
        # Each run has a new directory; metrics are the final completion marker.
        (directory / "report.md").write_text(render_report(receipt), encoding="utf-8")
        temporary = directory / "metrics.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(directory / "metrics.json")
        return receipt


def main():
    from contextlib import redirect_stdout
    config = GPUConfig.from_env()
    host = GPUHost(config)
    try:
        with redirect_stdout(sys.stderr):
            host.preload()
        print(json.dumps({"version": 1, "ready": host.status()}), flush=True)
        for line in sys.stdin:
            message = None
            try:
                message = json.loads(line)
                if set(message) != {"id", "payload"}:
                    raise ValueError("Invalid GPU protocol")
                with redirect_stdout(sys.stderr):
                    result = host.status() if message["payload"] == {"operation": "health"} else host.execute(message["payload"])
                reply = {"version": 1, "id": message["id"], "result": result}
            except Exception as error:
                reply = {"version": 1, "id": message.get("id") if isinstance(message, dict) else None,
                         "error": asdict(error_report("request", error))}
            print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"version": 1, "error": asdict(error_report("startup", error))}), flush=True)
        return 1
    finally:
        with redirect_stdout(sys.stderr):
            host.engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
