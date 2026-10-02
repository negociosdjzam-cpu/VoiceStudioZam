"""Deterministic engines, with observable work rather than real model weights."""
import pytest

from aureo import AudioChunk, AureoEngine, EngineCapabilities


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_engine(clock):
    def make(engine_id="test", *, fail=None, hook=None):
        class TestEngine(AureoEngine):
            display_name = "Local test engine"
            capabilities = EngineCapabilities(
                voice_reference=True, seed=True, variation=True, energy=True,
                expression=True, streaming=True, languages=("en", "es"),
            )

            def __init__(self):
                super().__init__()
                self.engine_id = engine_id
                self.loads = self.unloads = 0
                self.has_runtime = False
                self.requests = []
                self.stream_closed = False

            def _load(self):
                self.loads += 1
                self.has_runtime = True
                clock.now += 2
                if fail == "load":
                    raise RuntimeError("private-token-in-sdk-error")

            def _generate(self, request):
                self.requests.append(request)
                clock.now += 3
                if hook:
                    hook(request)
                if fail == "generate":
                    raise RuntimeError("private-token-in-sdk-error")
                return AudioChunk((0.1,) * 10, 10)

            def _stream(self, request):
                self.requests.append(request)
                try:
                    if fail == "empty":
                        return
                    clock.now += 1
                    yield AudioChunk((0.1,) * 10, 10)
                    clock.now += 4
                    if fail == "partial":
                        raise RuntimeError("private-token-in-sdk-error")
                    yield AudioChunk((0.2,) * 10, 20 if fail == "rate" else 10)
                finally:
                    self.stream_closed = True

            def _unload(self):
                if self.has_runtime:
                    self.unloads += 1
                    self.has_runtime = False
                    clock.now += 1
                    if fail == "cleanup":
                        raise RuntimeError("private-token-in-sdk-error")

        return TestEngine()
    return make
