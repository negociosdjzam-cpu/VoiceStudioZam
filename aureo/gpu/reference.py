"""Content-addressed WAV references; no URL fetches or pickle embeddings."""
import base64
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
import wave


def voice_path(config, voice_id):
    if not isinstance(voice_id, str) or not re.fullmatch(r"[a-f0-9]{64}", voice_id):
        raise ValueError("Invalid voice_id")
    path = config.references / (voice_id + ".wav")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Unknown local voice reference")
    data = path.read_bytes()
    if len(data) > config.max_reference_bytes or hashlib.sha256(data).hexdigest() != voice_id:
        raise ValueError("Reference integrity check failed")
    return path


def save_reference(config, data):
    if not isinstance(data, bytes) or not 0 < len(data) <= config.max_reference_bytes:
        raise ValueError("Reference exceeds the WAV size limit")
    try:
        with wave.open(io.BytesIO(data)) as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            if audio.getcomptype() != "NONE" or audio.getnchannels() not in (1, 2) or audio.getsampwidth() not in (1, 2, 3, 4):
                raise ValueError("Use an integer PCM mono/stereo WAV")
            if not 8000 <= rate <= 96000 or not 3 <= frames / rate <= 30:
                raise ValueError("Use a 3–30 second reference WAV")
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(frames)) != expected:
                raise ValueError("Truncated reference WAV")
    except (wave.Error, EOFError) as error:
        raise ValueError("Invalid reference WAV") from error
    voice_id = hashlib.sha256(data).hexdigest()
    config.references.mkdir(parents=True, exist_ok=True)
    destination = config.references / (voice_id + ".wav")
    with tempfile.NamedTemporaryFile(dir=config.references, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            voice_path(config, voice_id)
    finally:
        temporary.unlink(missing_ok=True)
    return voice_id


def resolve_reference(config, payload):
    voice_id, encoded = payload.get("voice_id"), payload.get("reference_wav_base64")
    if (voice_id is None) == (encoded is None):
        raise ValueError("Supply exactly one voice_id or reference_wav_base64")
    if encoded is not None:
        if not isinstance(encoded, str) or len(encoded) > 4 * ((config.max_reference_bytes + 2) // 3):
            raise ValueError("Reference exceeds the size limit")
        voice_id = save_reference(config, base64.b64decode(encoded, validate=True))
    return voice_id, voice_path(config, voice_id)
