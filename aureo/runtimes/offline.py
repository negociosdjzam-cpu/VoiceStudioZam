"""Offline SDK loading in a dedicated worker; never patch the legacy host."""
from contextlib import contextmanager
import os
import socket
from unittest.mock import patch

from ..types import AureoEngineError


@contextmanager
def offline_assets_only():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AureoEngineError("offline_asset_missing", "SDK attempted network access; provision all assets locally first")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AureoEngineError("offline_asset_missing", "SDK attempted network access; provision all assets locally first")
        return original_connect_ex(sock, address)

    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}), \
         patch.object(socket.socket, "connect", connect), patch.object(socket.socket, "connect_ex", connect_ex):
        yield
