"""Native ChatterboxMultilingualTTS bridge; separate from English Chatterbox."""
from ..types import AudioChunk, EngineCapabilities, SynthesisRequest
from .audio import mono_chunk
from .chatterbox import ChatterboxAdapter
from .seed import runtime_device, seed_scope


class ChatterboxMultilingualAdapter(ChatterboxAdapter):
    engine_id = "chatterbox-multilingual"
    display_name = "Chatterbox Multilingual"
    capabilities = EngineCapabilities(
        voice_reference=True, requires_voice_reference=True, seed=True,
        variation=True, energy=True, fidelity=True,
        languages=("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"),
    )

    def _generate(self, request: SynthesisRequest) -> AudioChunk:
        options = dict(self._PRESETS[request.mode])
        if request.variation is not None:
            options["temperature"] = 0.1 + 0.9 * request.variation
        if request.fidelity is not None:
            # Guidance strength proxy, NOT a measured speaker similarity score.
            options["cfg_weight"] = 0.1 + 0.8 * request.fidelity
        options["exaggeration"] = 0.5 if request.energy is None else request.energy
        options["audio_prompt_path"] = str(request.reference.path)
        options["language_id"] = request.language or "en"
        conditioning = getattr(self._runtime, "conds", None)
        try:
            with seed_scope(request.seed, device=runtime_device(self._runtime)):
                return mono_chunk(self._runtime.generate(request.text, **options), self._runtime.sr)
        finally:
            self._runtime.conds = conditioning
