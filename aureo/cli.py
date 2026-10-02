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
    benchmark.add_argument("--streaming", action="store_true", help="Require actual streaming; unsupported engines return an error")
    args = parser.parse_args(argv)
    try:
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
                overrides = {name: getattr(args, name) for name in ("mode", "language", "seed", "variation", "energy", "expression") if getattr(args, name) is not None}
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
