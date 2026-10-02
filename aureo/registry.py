"""An explicit instance registry, isolated from every legacy engine registry."""
from __future__ import annotations

from threading import RLock

from .engine import AureoEngine
from .types import AureoEngineError, valid_identifier


class AureoEngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, AureoEngine] = {}
        self._lock = RLock()

    def register(self, engine: AureoEngine) -> None:
        if not isinstance(engine, AureoEngine):
            raise TypeError("Registry entries must implement AureoEngine")
        engine_id = valid_identifier(engine.engine_id)
        with self._lock:
            if engine_id in self._engines:
                raise ValueError("Engine identifier is already registered")
            self._engines[engine_id] = engine

    def get(self, engine_id: str) -> AureoEngine:
        with self._lock:
            try:
                return self._engines[engine_id]
            except KeyError:
                raise AureoEngineError("unknown_engine", "The requested AUREO engine is not registered") from None

    def describe(self) -> list[dict]:
        with self._lock:
            engines = tuple(self._engines.values())
        return [engine.describe() for engine in engines]
