"""OpenVoice V2 with an explicitly configured local Melo source synthesizer."""
from ..types import AureoEngineError, EngineCapabilities
from .openvoice_v2 import OpenVoiceV2Adapter


class OpenVoiceMeloAdapter(OpenVoiceV2Adapter):
    display_name = "OpenVoice V2 + MeloTTS"

    def __init__(self, runtime_factory=None, *, language="es", runtime_disposer=None):
        super().__init__(runtime_factory, runtime_disposer=runtime_disposer)
        self.capabilities = EngineCapabilities(
            voice_reference=True, requires_voice_reference=True, seed=True,
            variation=True, fidelity=True, speed=True, languages=(language,),
        )

    def _target_embedding(self, source, target, request):
        if request.fidelity is None or request.fidelity == 1:
            return target
        if getattr(source.speaker_embedding, "shape", None) != getattr(target, "shape", None):
            raise AureoEngineError("embedding_mismatch", "Source and target speaker embeddings have different shapes")
        return (1 - request.fidelity) * source.speaker_embedding + request.fidelity * target
