"""Local V2 converter + local Melo + matching speaker embedding + watermark."""
from unittest.mock import patch

from ..adapters.audio import mono_chunk
from ..adapters.openvoice_v2 import OpenVoiceRuntime, OpenVoiceSource
from ..types import AureoEngineError
from .common import import_sdk, read_json, require_files, torch_for_device
from .config import MELO_LANGUAGES


def load_openvoice(config):
    require_files(config.converter_config, config.converter_checkpoint, config.melo_config,
                  config.melo_checkpoint, config.source_embedding, config.watermark_checkpoint)
    if read_json(config.converter_config).get("_version_") != "v2":
        raise AureoEngineError("wrong_model", "Expected OpenVoice V2 converter configuration")
    source_config = read_json(config.melo_config)
    if config.speaker not in source_config.get("data", {}).get("spk2id", {}):
        raise AureoEngineError("speaker_missing", "The source speaker does not exist in the Melo configuration")
    torch = torch_for_device(config.device)
    watermark = import_sdk("wavmark")
    converter_sdk = import_sdk("openvoice.api")
    melo_sdk = import_sdk("melo.api")
    # Preserve native watermarking while preventing its default HF download.
    original_load = watermark.load_model
    with patch.object(watermark, "load_model", lambda: original_load(path=str(config.watermark_checkpoint))):
        converter = converter_sdk.ToneColorConverter(str(config.converter_config), device=config.device)
    converter.load_ckpt(str(config.converter_checkpoint))
    if converter.version != "v2":
        raise AureoEngineError("wrong_model", "Loaded converter is not OpenVoice V2")
    source_model = melo_sdk.TTS(
        language=MELO_LANGUAGES[config.language], device=config.device, use_hf=False,
        config_path=str(config.melo_config), ckpt_path=str(config.melo_checkpoint),
    )
    embedding = torch.load(str(config.source_embedding), map_location=config.device, weights_only=True).to(config.device)
    speaker_id = source_model.hps.data.spk2id[config.speaker]

    def synthesize_source(request):
        options = {"speed": 1.0 if request.speed is None else request.speed, "quiet": True}
        if request.variation is not None:
            options["noise_scale"] = 0.1 + 0.8 * request.variation
        audio = source_model.tts_to_file(request.text, speaker_id, output_path=None, **options)
        return OpenVoiceSource(mono_chunk(audio, source_model.hps.data.sampling_rate), embedding)

    return OpenVoiceRuntime(converter, synthesize_source)
