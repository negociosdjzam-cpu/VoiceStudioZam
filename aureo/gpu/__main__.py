"""Opt-in GPU worker and automatic three-style/three-seed benchmark."""
import argparse
import json
from pathlib import Path
import sys

from .client import PersistentGPUClient
from .config import GPUConfig, verify_gpu
from .reference import save_reference


def run(argv=None):
    parser = argparse.ArgumentParser(description="AUREO persistent NVIDIA GPU worker")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="Verify CUDA and the deployment VRAM floor; no downloads")
    commands.add_parser("warmup", help="Load once and print readiness, then exit")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("runpod")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--text", required=True)
    benchmark.add_argument("--reference", type=Path, required=True)
    benchmark.add_argument("--language", default="es")
    benchmark.add_argument("--seeds", nargs=3, type=int, default=[42, 43, 44])
    args = parser.parse_args(argv)
    config = GPUConfig.from_env()
    if args.command == "probe":
        print(json.dumps(verify_gpu(config)))
    elif args.command == "serve":
        from .http import create_app
        import uvicorn
        uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)
    elif args.command == "runpod":
        from .runpod import main as runpod_main
        runpod_main()
    else:
        client = PersistentGPUClient(config)
        try:
            if args.command == "warmup":
                result = client.start()
            else:
                voice_id = save_reference(config, args.reference.read_bytes())
                result = client.request({"operation": "benchmark", "text": args.text,
                                         "voice_id": voice_id, "language": args.language, "seeds": args.seeds})
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 1 if result.get("success") is False else 0
        finally:
            client.close()
    return 0


def main(argv=None):
    from dataclasses import asdict
    from ..benchmark import error_report
    try:
        return run(argv)
    except Exception as error:
        print(json.dumps({"error": asdict(error_report("gpu_cli", error))}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
