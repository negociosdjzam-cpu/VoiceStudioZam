"""Reject non-Base/non-1.7B assets before loading the Qwen runtime."""
from ..types import AureoEngineError
from .common import import_sdk, read_json, require_files, torch_for_device


def load_qwen3(config):
    root = config.checkpoint_dir
    if root is None:
        require_files(None)
    value = read_json(root / "config.json")
    if value.get("tts_model_type") != "base" or str(value.get("tts_model_size", "")).lower() != "1.7b":
        raise AureoEngineError("wrong_model", "Expected Qwen3-TTS 1.7B Base checkpoints")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        require_files(None)
    require_files(*weights)
    torch = torch_for_device(config.device)
    dtype = config.dtype
    if dtype == "auto":
        if config.device.startswith("cuda"):
            with torch.cuda.device(config.device):
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    sdk = import_sdk("qwen_tts")
    runtime = sdk.Qwen3TTSModel.from_pretrained(
        str(root), device_map=config.device, dtype=getattr(torch, dtype),
        attn_implementation="sdpa", local_files_only=True,
    )
    if runtime.model.tts_model_type != "base" or str(runtime.model.tts_model_size).lower() != "1.7b":
        raise AureoEngineError("wrong_model", "Loaded Qwen runtime is not 1.7B Base")
    return runtime
