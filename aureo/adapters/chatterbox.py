"""Bridge to Resemble's ChatterboxTTS (English base), not the legacy backend."""
from __future__ import annotations

from ..types import AudioChunk, EngineCapabilities, QualityMode, SynthesisRequest
from .audio import mono_chunk
from .base import RuntimeAdapter
from .seed import runtime_device, seed_scope


class ChatterboxAdapter(RuntimeAdapter):
    engine_id = "chatterbox"
    display_name = "Chatterbox"
    capabilities = EngineCapabilities(voice_reference=True, seed=True, variation=True, energy=True, languages=("en",))
    _PRESETS = {
        QualityMode.FAST: {"cfg_weight": 0.3, "temperature": 0.7},
        QualityMode.PRO: {"cfg_weight": 0.5, "temperature": 0.8},
        QualityMode.ULTRA: {"cfg_weight": 0.7, "temperature": 0.8},
    }

    def _generate(self, request: SynthesisRequest) -> AudioChunk:
        options = dict(self._PRESETS[request.mode])
        if request.variation is not None:
            options["temperature"] = 0.1 + 0.9 * request.variation
        options["exaggeration"] = 0.5 if request.energy is None else request.energy
        options["audio_prompt_path"] = None if request.reference is None else str(request.reference.path)
        # The native SDK retains the last reference in `conds`. Restore the
        # original conditioning so a later profile cannot inherit that voice.
        has_conditioning = hasattr(self._runtime, "conds")
        conditioning = getattr(self._runtime, "conds", None)
        try:
            with seed_scope(request.seed, device=runtime_device(self._runtime)):
                return mono_chunk(self._runtime.generate(request.text, **options), self._runtime.sr)
        finally:
            if has_conditioning:
                self._runtime.conds = conditioning
