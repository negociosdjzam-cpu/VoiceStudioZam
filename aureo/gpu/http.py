"""Dedicated authenticated test API; no registration in the legacy backend."""
from contextlib import asynccontextmanager
import hmac
import json
import os

from ..types import AureoEngineError
from .config import GPUConfig
from .export import artifact_path
from .gateway import Gateway
from .reference import resolve_reference


def create_app(config=None, *, gateway=None, token=None, preload=True):
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import FileResponse
    from starlette.concurrency import run_in_threadpool
    config = GPUConfig.from_env() if config is None else config
    gateway = Gateway(config) if gateway is None else gateway
    token = os.environ.get("AUREO_API_TOKEN", "") if token is None else token
    if not isinstance(token, str) or len(token) < 16:
        raise ValueError("Set AUREO_API_TOKEN with at least 16 characters for the Pod API")

    @asynccontextmanager
    async def lifespan(app):
        try:
            if preload:
                await run_in_threadpool(gateway.client.start)
            yield
        finally:
            gateway.close()

    app = FastAPI(title="AUREO GPU test worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.gateway = gateway

    async def body(request):
        authorization = request.headers.get("authorization", "")
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Authentication required")
        content = bytearray()
        async for block in request.stream():
            content.extend(block)
            if len(content) > 20 * 1024 * 1024:
                raise HTTPException(413, "Request exceeds the size limit")
        try:
            payload = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON")))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            return payload
        except (ValueError, UnicodeError):
            raise HTTPException(422, "Invalid JSON request") from None

    @app.get("/healthz")
    def health():
        return {"alive": True, "service": "aureo-gpu"}

    @app.get("/readyz")
    def ready():
        process = gateway.client.process
        if process is None or process.poll() is not None or not gateway.client.ready:
            raise HTTPException(503, "GPU worker is not ready")
        return {"ready": True, "model_loaded": True}

    async def execute(request, operation):
        payload = await body(request)
        if "operation" in payload and payload["operation"] != operation:
            raise HTTPException(422, "Operation does not match endpoint")
        try:
            return await run_in_threadpool(gateway.execute, {**payload, "operation": operation})
        except AureoEngineError as error:
            status = 422 if error.code in ("unsupported_control", "streaming_unsupported", "language_unsupported") else 503
            raise HTTPException(status, {"code": error.code, "message": str(error)}) from None
        except (ValueError, OSError):
            raise HTTPException(422, "Invalid GPU request or unavailable artifact") from None

    @app.post("/v1/aureo/test")
    async def synthesize(request: Request):
        return await execute(request, "synthesize")

    @app.post("/v1/aureo/benchmark")
    async def benchmark(request: Request):
        return await execute(request, "benchmark")

    @app.post("/v1/aureo/reference")
    async def reference(request: Request):
        payload = await body(request)
        if set(payload) != {"reference_wav_base64"}:
            raise HTTPException(422, "Supply only reference_wav_base64")
        try:
            voice_id, _ = resolve_reference(config, payload)
            return {"voice_id": voice_id}
        except (ValueError, OSError):
            raise HTTPException(422, "Invalid reference WAV") from None

    @app.get("/v1/aureo/runs/{run_id}/{filename}")
    def artifact(run_id: str, filename: str, request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Authentication required")
        try:
            path = artifact_path(config, run_id, filename)
        except (ValueError, OSError):
            raise HTTPException(404, "Unknown artifact") from None
        return FileResponse(path, media_type="audio/wav" if filename.endswith(".wav") else "application/json" if filename.endswith(".json") else "text/markdown", filename=filename)
    return app
