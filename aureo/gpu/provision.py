"""Explicit, revision-pinned model provisioning; refuse weights without CUDA."""
import argparse
import hashlib
import json
import os
import re

from .config import GPUConfig, verify_gpu

MODEL_FILES = ("ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json")


def provision(config, revision, *, gpu_check=verify_gpu, downloader=None):
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use an exact 40-character Hugging Face commit revision")
    gpu_check(config)  # BEFORE importing Hub or requesting any model artifact
    marker = config.model_dir / "aureo-model.json"
    if marker.exists():
        verify_model(config)
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["revision"] != revision:
            raise ValueError("Use a different model directory for a different revision")
        return saved
    config.model_dir.mkdir(parents=True, exist_ok=True)
    plan = config.model_dir / "aureo-provisioning.json"
    if plan.exists():
        if json.loads(plan.read_text(encoding="utf-8")) != {"revision": revision}:
            raise ValueError("Interrupted provisioning belongs to a different revision")
    else:
        if any((config.model_dir / name).exists() for name in MODEL_FILES):
            raise ValueError("Use an empty model directory or an existing verified manifest")
        with plan.open("x", encoding="utf-8") as stream:
            json.dump({"revision": revision}, stream)
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    downloader(repo_id="ResembleAI/chatterbox", revision=revision, allow_patterns=list(MODEL_FILES),
               local_dir=str(config.model_dir), token=os.environ.get("HF_TOKEN"))
    files = {}
    for name in MODEL_FILES:
        path = config.model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("The pinned snapshot is missing required multilingual assets")
        files[name] = _digest(path)
    receipt = {"version": 1, "repo_id": "ResembleAI/chatterbox", "revision": revision,
               "sdk": "chatterbox-tts==0.1.7", "sha256": files}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return receipt


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify_model(config):
    marker = config.model_dir / "aureo-model.json"
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if receipt.get("version") != 1 or receipt.get("repo_id") != "ResembleAI/chatterbox" or not re.fullmatch(r"[a-f0-9]{40}", receipt.get("revision", "")) or set(receipt.get("sha256", {})) != set(MODEL_FILES):
        raise ValueError("Invalid model provisioning manifest")
    for name, expected in receipt["sha256"].items():
        path = config.model_dir / name
        if not path.is_file() or _digest(path) != expected:
            raise ValueError("Provisioned model asset checksum mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Explicit Chatterbox provisioning on NVIDIA GPU only")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(provision(GPUConfig.from_env(), args.revision)))


if __name__ == "__main__":
    main()
