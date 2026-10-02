"""Sequential isolated trials with a single reference snapshot and no audio IPC."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from .benchmark import BenchmarkError, BenchmarkResult
from .presets import BenchmarkStyle, CONTROLS, request_to_dict, style_request
from .runtimes import ENGINE_IDS, create_benchmark_registry
from .types import VoiceReference


def run_worker(payload, config, timeout):
    environment = dict(os.environ)
    # Only AUREO is imported from the checkout; SDKs come from the chosen env.
    root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (root, environment.get("PYTHONPATH"))))
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    python = sys.executable if config is None or config.python is None else str(config.python)
    completed = subprocess.run(
        [python, "-m", "aureo.worker"], input=json.dumps(payload, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        timeout=timeout, check=False,  # subprocess.run kills/reaps timed-out workers
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("Native worker exited unsuccessfully")
    response = json.loads(completed.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite worker data")))
    return response


def validate_reply(reply, engine_id, request, text_digest, reference_digest):
    if not isinstance(reply, dict) or set(reply) != {"version", "result", "runtime"} or type(reply["version"]) is not int or reply["version"] != 1:
        raise ValueError("Invalid worker response")
    result = reply["result"]
    expected = BenchmarkResult(engine_id, request.mode.value, request.streaming).to_dict()
    if not isinstance(result, dict) or set(result) != set(expected) or not isinstance(reply["runtime"], dict):
        raise ValueError("Invalid worker result schema")
    if result["engine_id"] != engine_id or result["mode"] != request.mode.value or result["streaming"] is not request.streaming:
        raise ValueError("Worker identity mismatch")
    errors = result["errors"]
    if not isinstance(errors, list) or any(not isinstance(error, dict) or set(error) != {"stage", "code", "message"} or any(not isinstance(value, str) for value in error.values()) for error in errors):
        raise ValueError("Invalid worker errors")
    if type(result["success"]) is not bool or result["success"] != (not errors):
        raise ValueError("Inconsistent worker success")
    if type(result["chunks"]) is not int or result["chunks"] < 0 or type(result["partial_audio"]) is not bool or result["partial_audio"] != bool(errors and result["chunks"]):
        raise ValueError("Invalid worker chunks")
    for name in ("load_seconds", "generation_seconds", "total_seconds", "generated_duration_seconds", "time_to_first_audio_seconds", "generation_time_to_first_audio_seconds", "real_time_factor", "end_to_end_real_time_factor"):
        value = result[name]
        nullable = name in ("time_to_first_audio_seconds", "generation_time_to_first_audio_seconds", "real_time_factor", "end_to_end_real_time_factor")
        if value is None and nullable:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid worker measurement")
    rate = result["sample_rate"]
    if rate is not None and (type(rate) is not int or rate <= 0):
        raise ValueError("Invalid worker sample rate")
    memory = result["memory"]
    if not isinstance(memory, dict) or set(memory) != {"ram_peak_bytes", "vram_peak_bytes", "sample_interval_seconds"}:
        raise ValueError("Invalid worker memory")
    for name in ("ram_peak_bytes", "vram_peak_bytes"):
        if memory[name] is not None and (type(memory[name]) is not int or memory[name] < 0):
            raise ValueError("Invalid worker memory measurement")
    interval = memory["sample_interval_seconds"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or not 0 < interval <= 1:
        raise ValueError("Invalid worker memory interval")
    if result["success"] and (not result["chunks"] or not rate or result["generated_duration_seconds"] <= 0):
        raise ValueError("Successful worker returned no audio")
    if errors and (result["real_time_factor"] is not None or result["end_to_end_real_time_factor"] is not None):
        raise ValueError("Failed worker cannot claim real-time factors")
    if result["text_sha256"] != text_digest or result["reference_sha256"] != reference_digest:
        raise ValueError("Worker input fingerprint mismatch")
    return result


class BenchmarkExperiment:
    def __init__(self, configs=None, *, runner=run_worker, timeout=300):
        self.configs = {} if configs is None else dict(configs)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("Worker timeout must be between 0 and 3600 seconds")
        self.timeout = timeout
        self.runner = runner
        self.registry = create_benchmark_registry(self.configs)

    def run(self, request, *, engine_ids=None, styles=None, repeats=1, strict_presets=False):
        identifiers = tuple((self.configs or dict.fromkeys(ENGINE_IDS)) if engine_ids is None else engine_ids)
        selected_styles = tuple(BenchmarkStyle.parse(style) for style in (tuple(BenchmarkStyle) if styles is None else styles))
        if not identifiers or set(identifiers) - set(ENGINE_IDS) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Choose unique native engine identifiers")
        if not selected_styles or len(set(selected_styles)) != len(selected_styles):
            raise ValueError("Choose unique benchmark styles")
        if type(repeats) is not int or not 1 <= repeats <= 100:
            raise ValueError("Benchmark repeats must be between 1 and 100")
        if request.reference is None:
            raise ValueError("A shared local voice reference is required")
        text_digest = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
        receipt = {
            "version": 1, "run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "text_sha256": text_digest, "reference_sha256": None,
            "reference_transcript_sha256": None if request.reference.transcript is None else hashlib.sha256(request.reference.transcript.encode("utf-8")).hexdigest(),
            "language": request.language, "mode": request.mode.value, "streaming": request.streaming,
            "cold": True, "isolated_process_per_trial": True, "repeats": repeats,
            "strict_presets": bool(strict_presets), "results": [],
        }
        with tempfile.TemporaryDirectory(prefix="aureo-experiment-") as directory:
            target = Path(directory) / ("reference" + request.reference.path.suffix)
            reference_error = None
            try:
                digest = hashlib.sha256()
                with request.reference.path.open("rb") as source, target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
                receipt["reference_sha256"] = digest.hexdigest()
                shared = replace(request, reference=VoiceReference(target, request.reference.transcript))
            except OSError:
                reference_error = "The shared local voice reference cannot be read"
                shared = request
            # Keep all styles/repeats for one engine together; only one worker
            # exists at a time, and no model survives into another trial.
            for engine_id in identifiers:
                engine = self.registry.get(engine_id)
                config = self.configs.get(engine_id)
                for style in selected_styles:
                    effective, requested, omitted = style_request(shared, style, engine.capabilities, strict=strict_presets)
                    for trial in range(1, repeats + 1):
                        started = time.perf_counter()
                        metadata = {}
                        failed = BenchmarkResult(engine_id, request.mode.value, request.streaming)
                        failed.text_sha256 = text_digest
                        failed.reference_sha256 = receipt["reference_sha256"]
                        if reference_error:
                            failed.errors.append(BenchmarkError("reference", "reference_unreadable", reference_error))
                            result = failed.to_dict()
                        else:
                            payload = {"version": 1, "engine_id": engine_id, "config": None if config is None else config.to_dict(), "request": request_to_dict(effective)}
                            try:
                                response = self.runner(payload, config, self.timeout)
                                result = validate_reply(response, engine_id, effective, text_digest, receipt["reference_sha256"])
                                metadata = response["runtime"]
                            except Exception as error:
                                code = "worker_timeout" if isinstance(error, subprocess.TimeoutExpired) else "worker_error"
                                failed.errors.append(BenchmarkError("worker", code, f"Isolated worker failed ({type(error).__name__})"))
                                result = failed.to_dict()
                        receipt["results"].append({
                            **result, "style": style.value, "trial": trial,
                            "process_wall_seconds": time.perf_counter() - started,
                            "requested_controls": {"seed": effective.seed, **requested},
                            "effective_controls": {"seed": effective.seed, **{key: getattr(effective, key) for key in CONTROLS}},
                            "omitted_controls": omitted, "runtime": metadata,
                        })
        return receipt
