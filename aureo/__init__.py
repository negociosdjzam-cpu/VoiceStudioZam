"""AUREO Core foundation. Importing this package never imports OmniVoice/SDKs."""
from .engine import AureoEngine
from .lab import AureoEngineLab
from .profiles import AureoVoiceProfile
from .registry import AureoEngineRegistry
from .types import AudioChunk, AureoEngineError, EngineCapabilities, QualityMode, SynthesisRequest, VoiceReference

__all__ = [
    "AureoEngine", "AureoEngineLab", "AureoEngineRegistry", "AureoVoiceProfile",
    "AudioChunk", "AureoEngineError", "EngineCapabilities", "QualityMode",
    "SynthesisRequest", "VoiceReference",
]
