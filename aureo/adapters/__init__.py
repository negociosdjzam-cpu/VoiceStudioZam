"""Discovery creates unloaded adapters and never imports vendor SDKs."""
from typing import Callable, Mapping

from ..registry import AureoEngineRegistry
from .chatterbox import ChatterboxAdapter
from .openvoice_v2 import OpenVoiceRuntime, OpenVoiceSource, OpenVoiceV2Adapter
from .qwen3_tts import Qwen3TTSAdapter


def create_default_registry(runtime_factories: Mapping[str, Callable] | None = None) -> AureoEngineRegistry:
    factories = {} if runtime_factories is None else dict(runtime_factories)
    classes = (ChatterboxAdapter, Qwen3TTSAdapter, OpenVoiceV2Adapter)
    if set(factories) - {adapter.engine_id for adapter in classes}:
        raise ValueError("Unknown runtime factory identifier")
    registry = AureoEngineRegistry()
    for adapter in classes:
        registry.register(adapter(factories.get(adapter.engine_id)))
    return registry


__all__ = ["ChatterboxAdapter", "Qwen3TTSAdapter", "OpenVoiceV2Adapter", "OpenVoiceRuntime", "OpenVoiceSource", "create_default_registry"]
