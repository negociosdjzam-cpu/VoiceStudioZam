"""Reuse native Voice DNA without replacing the existing AUREO adapter."""
from collections import OrderedDict
import copy
import hashlib
from pathlib import Path
import time

from ..adapters.seed import seed_scope


class CachedChatterboxRuntime:
    def __init__(self, native, *, capacity=3, synchronize=lambda: None):
        self.native = native
        self.capacity = capacity
        self.synchronize = synchronize
        self.cache = OrderedDict()
        self.hits = self.misses = 0
        self.last_hit = False
        self.last_prepare_seconds = 0.0

    @property
    def device(self):
        return self.native.device

    @property
    def sr(self):
        return self.native.sr

    @property
    def conds(self):
        return self.native.conds

    @conds.setter
    def conds(self, value):
        self.native.conds = value

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.cache), "capacity": self.capacity}

    def generate(self, text, *, audio_prompt_path, exaggeration=0.5, **options):
        digest = hashlib.sha256(Path(audio_prompt_path).read_bytes()).hexdigest()
        key = (digest, float(exaggeration))
        saved = self.native.conds
        self.last_hit = key in self.cache
        started = time.perf_counter()
        try:
            if self.last_hit:
                self.hits += 1
                conditioning = self.cache.pop(key)
            else:
                self.misses += 1
                # Preparation consumes its own stable RNG scope, so cache hits
                # do not alter the generator's requested-seed RNG sequence.
                with seed_scope(0, device=self.device):
                    self.native.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
                    self.synchronize()
                conditioning = self.native.conds
                if conditioning is None:
                    raise RuntimeError("Native reference preparation returned no conditioning")
            self.cache[key] = conditioning
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            # The pinned SDK replaces conds.t3 to change emotion. Clone the
            # containers, sharing immutable tensor storage, not the cache object.
            active = copy.copy(conditioning)
            active.t3 = copy.copy(conditioning.t3)
            active.gen = dict(conditioning.gen)
            self.native.conds = active
            self.last_prepare_seconds = time.perf_counter() - started
            return self.native.generate(text, audio_prompt_path=None, exaggeration=exaggeration, **options)
        finally:
            self.native.conds = saved
