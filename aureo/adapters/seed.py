"""Serialized seed scopes for explicit local SDK runtimes.

The lab CLI runs in its own process. In-process callers must not concurrently
use unrelated SDKs that mutate global RNGs; isolated runtime workers are the
next integration boundary. No framework is imported or initialized here.
"""
from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from threading import RLock
from typing import Iterator

_RNG_LOCK = RLock()


@contextmanager
def seed_scope(seed: int | None, *, device: object = None) -> Iterator[None]:
    with _RNG_LOCK:
        if seed is None:
            yield
            return
        python_state = random.getstate()
        numpy = sys.modules.get("numpy")
        torch = sys.modules.get("torch")
        numpy_state = torch_state = cuda_state = mps_state = None
        try:
            if numpy is not None:
                numpy_state = numpy.random.get_state()
                numpy.random.seed(seed)
            if torch is not None:
                torch_state = torch.random.get_rng_state()
                torch.random.default_generator.manual_seed(seed)
                if torch.cuda.is_initialized():
                    cuda_state = torch.cuda.get_rng_state_all()
                    torch.cuda.manual_seed_all(seed)
                if str(device).startswith("mps"):
                    mps_state = torch.mps.get_rng_state()
                    torch.mps.manual_seed(seed)
            random.seed(seed)
            yield
        finally:
            random.setstate(python_state)
            if numpy_state is not None:
                numpy.random.set_state(numpy_state)
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


def runtime_device(runtime: object) -> object:
    return getattr(runtime, "device", getattr(getattr(runtime, "model", None), "device", None))
