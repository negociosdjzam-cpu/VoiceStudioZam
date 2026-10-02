"""A serialized, restartable persistent child; gateway threads never load TTS."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from ..types import AureoEngineError


class PersistentGPUClient:
    def __init__(self, config, *, command=None):
        self.config = config
        self.command = [str(config.worker_python or sys.executable), "-m", "aureo.gpu.worker"] if command is None else list(command)
        self.lock = threading.RLock()
        self.process = None
        self.responses = None
        self.ready = None

    @staticmethod
    def _read(process, responses):
        try:
            for line in process.stdout:
                try:
                    responses.put(json.loads(line))
                except (ValueError, UnicodeError):
                    responses.put(None)
                    break
        except (OSError, ValueError):
            pass  # closing a reaped process may close the reader's pipe
        finally:
            responses.put(None)

    def _receive(self, identifier=None):
        try:
            reply = self.responses.get(timeout=self.config.timeout_seconds)
        except queue.Empty:
            self.close()
            raise AureoEngineError("worker_timeout", "GPU worker timed out and was terminated") from None
        expected = ({"version", "ready"}, {"version", "error"}) if identifier is None else ({"version", "id", "result"}, {"version", "id", "error"})
        if not isinstance(reply, dict) or type(reply.get("version")) is not int or reply["version"] != 1 or set(reply) not in expected or (identifier is not None and reply.get("id") != identifier):
            self.close()
            raise AureoEngineError("worker_error", "GPU worker returned an invalid response")
        if "error" in reply:
            error = reply["error"]
            if not isinstance(error, dict) or not all(isinstance(error.get(key), str) for key in ("code", "message")):
                self.close()
                raise AureoEngineError("worker_error", "GPU worker returned an invalid error")
            raise AureoEngineError(error["code"], error["message"])
        return reply

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.ready
            self.close()
            environment = {**os.environ, **self.config.environment()}
            root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (root, environment.get("PYTHONPATH"))))
            # Authentication lives only in the gateway, not in the model child.
            environment.pop("AUREO_API_TOKEN", None)
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            text=True, encoding="utf-8", bufsize=1, env=environment)
            self.responses = queue.Queue()
            threading.Thread(target=self._read, args=(self.process, self.responses), daemon=True, name="aureo-gpu-ipc").start()
            try:
                reply = self._receive()
                if not isinstance(reply.get("ready"), dict) or reply["ready"].get("ready") is not True:
                    raise AureoEngineError("worker_error", "GPU worker is not ready")
                self.ready = reply["ready"]
                return self.ready
            except Exception:
                self.close()
                raise

    def request(self, payload):
        waiting = time.perf_counter()
        with self.lock:
            queue_seconds = time.perf_counter() - waiting
            started = time.perf_counter()
            self.start()
            identifier = uuid.uuid4().hex
            try:
                self.process.stdin.write(json.dumps({"id": identifier, "payload": payload}, ensure_ascii=False, allow_nan=False) + "\n")
                self.process.stdin.flush()
                reply = self._receive(identifier)
                if reply.get("id") != identifier or not isinstance(reply.get("result"), dict):
                    raise AureoEngineError("worker_error", "GPU worker response identity mismatch")
            except AureoEngineError as error:
                if error.code in ("worker_error", "worker_timeout"):
                    self.close()
                raise
            except (OSError, ValueError):
                self.close()
                raise AureoEngineError("worker_error", "GPU worker communication failed") from None
            return {**reply["result"], "gateway_seconds": time.perf_counter() - started, "queue_wait_seconds": queue_seconds}

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            self.ready = None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
