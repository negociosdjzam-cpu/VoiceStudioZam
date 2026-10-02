"""Explicit 1.7B Base identity; local loader rejects other Qwen variants."""
from .qwen3_tts import Qwen3TTSAdapter


class Qwen3TTS17BBaseAdapter(Qwen3TTSAdapter):
    engine_id = "qwen3-tts-1.7b-base"
    display_name = "Qwen3-TTS 1.7B Base"
