"""Explicit runtime injection: no automatic SDK import, install or download."""
from __future__ import annotations

from typing import Any, Callable

from ..engine import AureoEngine
from ..types import AureoEngineError


class RuntimeAdapter(AureoEngine):
    def __init__(self, runtime_factory: Callable[[], Any] | None = None,
                 *, runtime_disposer: Callable[[Any], None] | None = None):
        super().__init__()
        if runtime_factory is not None and not callable(runtime_factory):
            raise TypeError("The runtime factory must be callable")
        if runtime_disposer is not None and not callable(runtime_disposer):
            raise TypeError("The runtime disposer must be callable")
        self._factory = runtime_factory
        self._disposer = runtime_disposer
        self._runtime: Any = None

    def describe(self) -> dict:
        return {**super().describe(), "configured": self._factory is not None}

    def _load(self) -> None:
        if self._factory is None:
            raise AureoEngineError("runtime_not_configured", "Configure an explicit local runtime; no models are downloaded automatically")
        runtime = self._factory()
        if runtime is None:
            raise AureoEngineError("invalid_runtime", "The runtime factory returned no engine")
        self._runtime = runtime

    def _unload(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None and self._disposer is not None:
            self._disposer(runtime)
