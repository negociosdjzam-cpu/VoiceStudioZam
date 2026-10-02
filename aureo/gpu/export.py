"""WAV exports pass through the existing synthetic-audio marking seam."""
import base64
import json
from pathlib import Path
import re
import struct
import sys
import wave

from ..types import AudioChunk


def mark_audio(chunk):
    # This is a checkout deployment. Keep backend unmodified and import only
    # its canonical marking seam inside the dedicated GPU child process.
    backend = Path(__file__).resolve().parents[2] / "backend"
    if not backend.is_dir():
        raise RuntimeError("GPU exports require the repository backend watermark seam")
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    import torch
    from services.watermark import mark_synthetic
    original = torch.tensor(chunk.samples, dtype=torch.float32).unsqueeze(0)
    marked = mark_synthetic(original, chunk.sample_rate, context="aureo.gpu.export", force=True)
    samples = tuple(marked.detach().cpu().reshape(-1).tolist())
    return AudioChunk(samples, chunk.sample_rate), {"native_perth": True, "audioseal_applied": marked is not original}


def write_wav(chunk, path):
    values = [max(-32768, min(32767, round(value * 32767))) for value in chunk.samples]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(chunk.sample_rate)
        audio.writeframes(struct.pack("<" + "h" * len(values), *values))


def artifact_path(config, run_id, filename):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run_id")
    if filename not in ("metrics.json", "report.md") and not re.fullmatch(r"(?:NATURAL|ENERGETICO|FIDELIDAD)-seed\d+\.wav", filename):
        raise ValueError("Invalid artifact name")
    root = config.results.resolve()
    directory = root / run_id
    path = directory / filename
    if directory.is_symlink() or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Unknown local artifact")
    if filename not in ("metrics.json", "report.md"):
        receipt = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if filename not in {row.get("wav") for row in receipt["results"] if row["success"]}:
            raise ValueError("Artifact is not part of a completed trial")
    return path


def with_inline_audio(config, receipt):
    total = 0
    audio = {}
    for row in receipt["results"]:
        if row["success"]:
            path = artifact_path(config, receipt["run_id"], row["wav"])
            total += 4 * ((path.stat().st_size + 2) // 3)
            if total > config.max_inline_bytes:
                raise ValueError("Inline audio exceeds the limit; retrieve persisted WAV artifacts separately")
            audio[row["wav"]] = base64.b64encode(path.read_bytes()).decode("ascii")
    return {**receipt, "audio_wav_base64": audio}
