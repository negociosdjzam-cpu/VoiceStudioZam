import json
from dataclasses import replace

import pytest

from aureo import AureoVoiceProfile, QualityMode, VoiceReference


def test_profile_round_trip_preserves_selection_and_controls(tmp_path):
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"original-reference")
    profile = AureoVoiceProfile(
        "speaker-1", "chatterbox", mode=QualityMode.FAST,
        reference=VoiceReference(reference, "Hola referencia"),
        seed=23, variation=0.7, energy=0.3, language="en",
        mode_engines={"PRO": "qwen3-tts", "ULTRA": "openvoice-v2"},
    )
    destination = tmp_path / "profiles" / "speaker.json"
    profile.write(destination)
    restored = AureoVoiceProfile.read(destination)
    assert restored == profile
    assert restored.select_engine() == "chatterbox"
    assert restored.select_engine(QualityMode.PRO) == "qwen3-tts"
    assert restored.select_engine(QualityMode.ULTRA) == "openvoice-v2"
    request = restored.request("Nuevo texto", mode=QualityMode.PRO, streaming=True)
    assert request.text == "Nuevo texto" and request.reference == profile.reference
    assert request.seed == 23 and request.variation == 0.7 and request.energy == 0.3
    assert request.streaming and request.mode is QualityMode.PRO
    assert reference.read_bytes() == b"original-reference"
    with pytest.raises(TypeError):
        restored.mode_engines[QualityMode.FAST] = "other"


def test_manual_relative_reference_resolves_against_profile_file(tmp_path, monkeypatch):
    folder = tmp_path / "profiles"
    folder.mkdir()
    path = folder / "voice.json"
    value = AureoVoiceProfile("voice", "chatterbox", reference=VoiceReference("ref.wav")).to_dict()
    path.write_text(json.dumps(value))
    monkeypatch.chdir(tmp_path)
    assert AureoVoiceProfile.read(path).reference.path == folder / "ref.wav"
    # Programmatic writes preserve the reference as resolved by the caller.
    AureoVoiceProfile("voice", "chatterbox", reference=VoiceReference("caller.wav")).write(path)
    assert AureoVoiceProfile.read(path).reference.path == tmp_path / "caller.wav"


def test_atomic_write_failure_keeps_previous_profile(tmp_path, monkeypatch):
    destination = tmp_path / "voice.json"
    original = AureoVoiceProfile("voice", "chatterbox")
    original.write(destination)
    def fail_replace(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr("aureo.profiles.os.replace", fail_replace)
    with pytest.raises(OSError):
        replace(original, engine_id="qwen3-tts").write(destination)
    assert AureoVoiceProfile.read(destination) == original
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"legacy_id": "old"}, {"engine_id": "../engine"},
    {"mode_engines": {"INVALID": "engine"}}, {"seed": -1},
    {"reference": {"path": "ref.wav", "url": "https://example.com"}},
])
def test_reject_unknown_or_invalid_profile_configuration(change):
    value = AureoVoiceProfile("voice", "chatterbox").to_dict()
    with pytest.raises(ValueError):
        AureoVoiceProfile.from_dict({**value, **change})
