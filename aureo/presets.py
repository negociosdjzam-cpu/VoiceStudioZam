"""Explicit benchmark intent; unsupported preset controls are recorded."""
from dataclasses import replace
from enum import Enum

from .types import SynthesisRequest

CONTROLS = ("variation", "energy", "expression", "fidelity", "speed")


class BenchmarkStyle(str, Enum):
    NATURAL = "NATURAL"
    ENERGETICO = "ENERGÉTICO"
    FIDELIDAD = "FIDELIDAD"

    @classmethod
    def parse(cls, value):
        return cls("ENERGÉTICO" if value == "ENERGETICO" else value)


PRESETS = {
    BenchmarkStyle.NATURAL: dict(variation=0.35, energy=0.45, expression="natural", fidelity=0.7, speed=1.0),
    BenchmarkStyle.ENERGETICO: dict(variation=0.65, energy=0.85, expression="energetic", fidelity=0.55, speed=1.1),
    BenchmarkStyle.FIDELIDAD: dict(variation=0.15, energy=0.35, expression="neutral", fidelity=1.0, speed=1.0),
}


def style_request(base, style, capabilities, *, strict=False):
    requested = {**PRESETS[BenchmarkStyle.parse(style)]}
    # Explicit caller controls always remain strict, even in adaptive presets.
    requested.update({key: getattr(base, key) for key in CONTROLS if getattr(base, key) is not None})
    omitted = []
    effective = dict(requested)
    if not strict:
        for key in CONTROLS:
            if getattr(base, key) is None and not getattr(capabilities, key):
                effective[key] = None
                omitted.append(key)
    return replace(base, **effective), requested, omitted


def request_to_dict(request):
    return {
        "text": request.text, "mode": request.mode.value, "language": request.language,
        "reference": None if request.reference is None else {
            "path": str(request.reference.path), "transcript": request.reference.transcript,
        },
        "seed": request.seed, "streaming": request.streaming,
        **{name: getattr(request, name) for name in CONTROLS},
    }


def request_from_dict(value):
    from .types import VoiceReference
    if not isinstance(value, dict) or set(value) - SynthesisRequest.__dataclass_fields__.keys():
        raise ValueError("Invalid worker request")
    fields = dict(value)
    if fields.get("reference") is not None:
        reference = fields["reference"]
        if not isinstance(reference, dict) or set(reference) - {"path", "transcript"}:
            raise ValueError("Invalid worker reference")
        fields["reference"] = VoiceReference(**reference)
    return SynthesisRequest(**fields)
