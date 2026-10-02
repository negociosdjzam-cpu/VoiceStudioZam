"""Optional, sampled process RSS and PyTorch allocator observations."""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable


@dataclass(frozen=True)
class MemoryStats:
    ram_peak_bytes: int | None = None
    vram_peak_bytes: int | None = None
    sample_interval_seconds: float = 0.02


def read_memory() -> tuple[int | None, int | None]:
    ram = vram = None
    try:
        import psutil  # optional; no new project dependency
        ram = psutil.Process().memory_info().rss
    except (ImportError, OSError, RuntimeError):
        pass
    # Do not import torch, initialize CUDA, or reset global peak counters.
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            vram = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
    except (AttributeError, OSError, RuntimeError):
        pass
    return ram, vram


class MemorySampler:
    def __init__(self, *, reader: Callable[[], tuple[int | None, int | None]] = read_memory, interval: float = 0.02):
        if isinstance(interval, bool) or not math.isfinite(interval) or not 0.001 <= interval <= 1:
            raise ValueError("Memory sample interval must be between 0.001 and 1 second")
        self.reader = reader
        self.interval = interval
        self._stop = Event()
        self._lock = Lock()
        self._ram: int | None = None
        self._vram: int | None = None
        self._thread: Thread | None = None

    def _sample(self) -> None:
        try:
            ram, vram = self.reader()
        except Exception:
            return  # observability must not fail synthesis
        with self._lock:
            if ram is not None:
                self._ram = ram if self._ram is None else max(self._ram, ram)
            if vram is not None:
                self._vram = vram if self._vram is None else max(self._vram, vram)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Memory sampler is single-use")
        self._sample()
        self._thread = Thread(target=self._run, daemon=True, name="aureo-memory-sampler")
        self._thread.start()

    def stop(self) -> MemoryStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._sample()
        with self._lock:
            return MemoryStats(self._ram, self._vram, self.interval)
