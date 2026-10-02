"""Runpod polling stays in the parent; CUDA/offline SDKs stay in the child."""
from ..types import AureoEngineError
from .config import GPUConfig
from .gateway import Gateway


def create_handler(gateway):
    def handler(job):
        try:
            if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
                raise ValueError("A job must contain an input object")
            return gateway.execute(job["input"])
        except AureoEngineError as error:
            return {"error": {"code": error.code, "message": str(error)}}
        except (ValueError, OSError):
            return {"error": {"code": "invalid_request", "message": "Invalid GPU request or unavailable artifact"}}
    return handler


def main():
    import runpod
    gateway = Gateway(GPUConfig.from_env())
    try:
        gateway.client.start()  # preload ONCE before accepting Serverless jobs
        runpod.serverless.start({"handler": create_handler(gateway)})
    finally:
        gateway.close()
