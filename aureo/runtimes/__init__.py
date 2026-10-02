"""Native factories are lazy; discovery does not import any SDK."""
from ..adapters.chatterbox_multilingual import ChatterboxMultilingualAdapter
from ..adapters.openvoice_melo import OpenVoiceMeloAdapter
from ..adapters.qwen3_base import Qwen3TTS17BBaseAdapter
from ..registry import AureoEngineRegistry
from .chatterbox import load_chatterbox
from .config import ENGINE_IDS, LocalRuntimeConfig, read_runtime_config
from .openvoice import load_openvoice
from .qwen3 import load_qwen3

MODEL_IDS = {
    "chatterbox-multilingual": "ResembleAI/chatterbox",
    "qwen3-tts-1.7b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "openvoice-v2": "OpenVoice V2 + local MeloTTS + WavMark",
}


def _load_and_sync(loader, config):
    runtime = loader(config)
    from .common import torch_for_device
    torch = torch_for_device(config.device)
    if config.device.startswith("cuda"):
        torch.cuda.synchronize(config.device)
    elif config.device == "mps":
        torch.mps.synchronize()
    return runtime


def create_benchmark_registry(configs=None):
    configs = {} if configs is None else dict(configs)
    if set(configs) - set(ENGINE_IDS):
        raise ValueError("Unknown benchmark engine configuration")
    registry = AureoEngineRegistry()
    for cls, loader in (
        (ChatterboxMultilingualAdapter, load_chatterbox),
        (Qwen3TTS17BBaseAdapter, load_qwen3),
        (OpenVoiceMeloAdapter, load_openvoice),
    ):
        config = configs.get(cls.engine_id)
        factory = None if config is None else lambda loader=loader, config=config: _load_and_sync(loader, config)
        kwargs = {"language": "es" if config is None else config.language} if cls is OpenVoiceMeloAdapter else {}
        registry.register(cls(factory, **kwargs))
    return registry


__all__ = ["ENGINE_IDS", "MODEL_IDS", "LocalRuntimeConfig", "create_benchmark_registry", "read_runtime_config"]
