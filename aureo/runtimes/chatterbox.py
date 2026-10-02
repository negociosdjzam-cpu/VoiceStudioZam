"""Only Chatterbox Multilingual's explicitly local loading API."""
from .common import import_sdk, require_files, torch_for_device


def load_chatterbox(config):
    root = config.checkpoint_dir
    if root is None:
        require_files(None)
    require_files(*(root / name for name in (
        "ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
    )))
    torch_for_device(config.device)
    sdk = import_sdk("chatterbox.mtl_tts")
    return sdk.ChatterboxMultilingualTTS.from_local(str(root), device=config.device)
