"""Shared Pod/Serverless controller; no engine imports with model side effects."""
import base64

from .client import PersistentGPUClient
from .export import artifact_path, with_inline_audio
from .reference import resolve_reference
from .worker import validate_payload


class Gateway:
    def __init__(self, config, client=None):
        self.config = config
        self.client = PersistentGPUClient(config) if client is None else client

    def execute(self, payload):
        if isinstance(payload, dict) and payload.get("operation") == "fetch":
            if set(payload) != {"operation", "run_id", "filename"}:
                raise ValueError("Invalid artifact request")
            path = artifact_path(self.config, payload["run_id"], payload["filename"])
            if 4 * ((path.stat().st_size + 2) // 3) > self.config.max_inline_bytes:
                raise ValueError("Artifact exceeds the inline transfer limit")
            return {"filename": path.name, "content_base64": base64.b64encode(path.read_bytes()).decode("ascii")}
        from dataclasses import replace
        from ..adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
        from ..presets import style_request
        from ..types import VoiceReference
        base, _, styles = validate_payload(self.config, payload)
        voice_id, reference = resolve_reference(self.config, payload)
        validator = ChatterboxMultilingualAdapter()
        for style in styles:
            effective, _, _ = style_request(replace(base, reference=VoiceReference(reference)), style, validator.capabilities)
            validator.validate(effective)
        payload = {key: value for key, value in payload.items() if key != "reference_wav_base64"}
        payload["voice_id"] = voice_id
        result = self.client.request(payload)
        return with_inline_audio(self.config, result) if payload.get("include_audio", False) else result

    def close(self):
        self.client.close()
