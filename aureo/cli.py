"""Metadata-only engine discovery and benchmarking in a separate process."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .adapters import create_default_registry
from .benchmark import AureoBenchmark
from .profiles import AureoVoiceProfile
from .registry import AureoEngineRegistry
from .types import QualityMode, SynthesisRequest, VoiceReference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUREO VOICE ENGINE LAB: no automatic model downloads")
    parser.add_argument("--registry-factory", help="Explicit local developer hook, module:function; must return AureoEngineRegistry")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Describe unloaded adapters")
    catalog = commands.add_parser("catalog", help="Describe the three native benchmark adapters without SDK imports")
    catalog.add_argument("--runtime-config", type=Path)
    benchmark = commands.add_parser("benchmark", help="Compare engines; return metrics and errors, never audio")
    benchmark.add_argument("--text", required=True)
    benchmark.add_argument("--profile", type=Path)
    benchmark.add_argument("--reference", type=Path)
    benchmark.add_argument("--reference-text")
    benchmark.add_argument("--engines", nargs="+")
    benchmark.add_argument("--mode", choices=[mode.value for mode in QualityMode])
    benchmark.add_argument("--language")
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--variation", type=float)
    benchmark.add_argument("--energy", type=float)
    benchmark.add_argument("--expression")
    benchmark.add_argument("--fidelity", type=float)
    benchmark.add_argument("--speed", type=float)
    benchmark.add_argument("--streaming", action="store_true", help="Require actual streaming; unsupported engines return an error")
    experiment = commands.add_parser("experiment", help="Isolated native trials; save JSON and a Markdown comparison")
    experiment.add_argument("--runtime-config", type=Path)
    experiment.add_argument("--text", required=True)
    experiment.add_argument("--reference", type=Path, required=True)
    experiment.add_argument("--reference-text")
    experiment.add_argument("--language", default="es")
    experiment.add_argument("--mode", choices=[mode.value for mode in QualityMode], default="PRO")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--engines", nargs="+")
    experiment.add_argument("--styles", nargs="+", choices=["NATURAL", "ENERGÉTICO", "ENERGETICO", "FIDELIDAD"])
    experiment.add_argument("--repeats", type=int, default=1)
    experiment.add_argument("--timeout", type=float, default=300)
    experiment.add_argument("--strict-presets", action="store_true")
    experiment.add_argument("--variation", type=float)
    experiment.add_argument("--energy", type=float)
    experiment.add_argument("--expression")
    experiment.add_argument("--fidelity", type=float)
    experiment.add_argument("--speed", type=float)
    experiment.add_argument("--streaming", action="store_true")
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--report", type=Path)
    experiment.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in ("catalog", "experiment"):
            if args.registry_factory:
                raise ValueError("Use local runtime configuration for native experiments")
            from .runtimes import create_benchmark_registry, read_runtime_config
            configs = read_runtime_config(args.runtime_config) if args.runtime_config else {}
            if args.command == "catalog":
                output = {"engines": create_benchmark_registry(configs).describe()}
                status = 0
            else:
                from .experiment import BenchmarkExperiment
                from .report import save_receipt, validate_output_paths
                from .runtimes.config import PATH_FIELDS
                protected = [args.reference]
                if args.runtime_config:
                    protected.append(args.runtime_config)
                protected += [getattr(config, name) for config in configs.values() for name in PATH_FIELDS if getattr(config, name) is not None]
                validate_output_paths(args.output, args.report, overwrite=args.overwrite, protected_paths=protected)
                request = SynthesisRequest(
                    args.text, reference=VoiceReference(args.reference, args.reference_text),
                    mode=args.mode, language=args.language, seed=args.seed,
                    streaming=args.streaming,
                    **{name: getattr(args, name) for name in ("variation", "energy", "expression", "fidelity", "speed")},
                )
                output = BenchmarkExperiment(configs, timeout=args.timeout).run(
                    request, engine_ids=args.engines, styles=args.styles,
                    repeats=args.repeats, strict_presets=args.strict_presets,
                )
                save_receipt(output, args.output, args.report, overwrite=args.overwrite, protected_paths=protected)
                status = 0 if all(result["success"] for result in output["results"]) else 1
            print(json.dumps(output, ensure_ascii=False, allow_nan=False))
            return status
        with redirect_stdout(sys.stderr):  # SDK progress prints must not corrupt JSON
            registry = create_default_registry()
            if args.registry_factory:
                module, separator, attribute = args.registry_factory.partition(":")
                if not separator:
                    raise ValueError("A registry factory must use module:function")
                registry = getattr(importlib.import_module(module), attribute)()
                if not isinstance(registry, AureoEngineRegistry):
                    raise TypeError("Factory must return AureoEngineRegistry")
            if args.command == "list":
                output = {"engines": registry.describe()}
                status = 0
            else:
                profile = AureoVoiceProfile.read(args.profile) if args.profile else None
                request = profile.request(args.text, streaming=args.streaming) if profile else SynthesisRequest(args.text, streaming=args.streaming)
                overrides = {name: getattr(args, name) for name in ("mode", "language", "seed", "variation", "energy", "expression", "fidelity", "speed") if getattr(args, name) is not None}
                if args.reference is not None:
                    overrides["reference"] = VoiceReference(args.reference, args.reference_text)
                elif args.reference_text is not None:
                    if request.reference is None:
                        raise ValueError("A reference transcript requires reference audio")
                    overrides["reference"] = replace(request.reference, transcript=args.reference_text)
                request = replace(request, **overrides)
                engine_ids = args.engines or ([profile.select_engine(request.mode)] if profile else [item["id"] for item in registry.describe()])
                results = AureoBenchmark(registry).run(request, engine_ids)
                output = {"results": [result.to_dict() for result in results]}
                status = 0 if all(result.success for result in results) else 1
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        return status
    except Exception as error:
        print(json.dumps({"error": "lab_request_failed", "type": type(error).__name__}), file=sys.stderr)
        return 2
