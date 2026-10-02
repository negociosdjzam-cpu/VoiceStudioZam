# Build from repository root. No models are baked into this image.
FROM python:3.11.16-slim-bookworm
ENV PYTHONUNBUFFERED=1 AUREO_ENV_DIR=/opt/aureo-env AUREO_PYTHON=python3.11
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/VoiceStudioZam
COPY deploy/aureo/ deploy/aureo/
RUN bash deploy/aureo/install.sh --build
COPY aureo/ aureo/
COPY backend/core/ backend/core/
COPY backend/services/watermark.py backend/services/watermark.py
EXPOSE 8000
ENTRYPOINT ["bash", "deploy/aureo/start.sh"]
CMD ["serverless"]
