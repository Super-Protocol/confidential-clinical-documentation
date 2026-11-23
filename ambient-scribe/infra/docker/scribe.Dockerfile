# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026, replacing `api.Dockerfile` + `ui.Dockerfile` +
# `nginx.ubuntu.Dockerfile`.
#
# ONE image instead of three, and the three reasons are not tidiness:
#
# 1. **A listing pins digests, and every digest is a thing a verifier has to
#    trust.** The API, the built UI and the model weights are one artefact here,
#    so "which code and which model touched this consultation" has one answer.
# 2. **The weights are inside it.** Upstream's models live in two NIM containers
#    that need an NGC credential; these are fetched at build time against
#    `models/manifest.json`, sha256 by sha256, so a deployment reaches no model
#    host and the image digest covers the weights.
# 3. **One origin.** The UI is served by the same process as the API, so there is
#    no CORS to configure and the authenticating console in front of it has one
#    thing to proxy to.
#
# Everything in this image runs on the CPU. There is no CUDA base, no
# `NVIDIA_REQUIRE_CUDA`, and therefore none of the driver-floor trouble a
# recent CUDA base brings: the card in a deployment of this listing belongs to
# the generation engine, which is a separate component.

# ---------------------------------------------------------------------------
# 1. The UI
# ---------------------------------------------------------------------------
FROM node:22-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS ui

WORKDIR /ui
COPY ambient-scribe/apps/ui/package.json ambient-scribe/apps/ui/package-lock.json ./
# `npm ci` and not `npm install`: the lockfile is the record of what was built and
# tested, and upstream's Dockerfile deleted it before installing (it had to — four
# of its dependencies were `file:` tarballs). Those are gone, so the lock works.
RUN npm ci --no-audit --no-fund
COPY ambient-scribe/apps/ui ./
RUN npm run build

# ---------------------------------------------------------------------------
# 2. The Python environment and the model weights
# ---------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
COPY ambient-scribe/apps/api/pyproject.toml ambient-scribe/apps/api/uv.lock ./
COPY ambient-scribe/apps/api/ambient_scribe ./ambient_scribe
# `--locked` and not `--frozen`: a lockfile that no longer matches pyproject.toml
# should fail the build rather than be quietly re-resolved into something nobody
# measured.
RUN uv sync --locked --no-dev

# The weights. A file whose sha256 is not the manifest's fails the build.
COPY ambient-scribe/models /models-src
RUN python3 /models-src/fetch_models.py --manifest /models-src/manifest.json --root /

# ---------------------------------------------------------------------------
# 3. The image that runs
# ---------------------------------------------------------------------------
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# `libgomp1` is CTranslate2's OpenMP runtime, and the only thing this image needs
# that the base image does not already have. There is deliberately no apt ffmpeg
# and no libsndfile: PyAV, which faster-whisper depends on, ships a complete
# ffmpeg, and `services/audio.py` decodes with it. Dropping those two packages and
# librosa's numba/llvmlite/scipy/scikit-learn tree takes about 850 MB off this
# image — pull time a tenant would otherwise wait through.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r -g 1000 scribe && useradd -r -u 1000 -g scribe -d /app scribe

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UI_DIR=/app/ui \
    ASR_MODEL_DIR=/models/asr \
    DIARIZATION_SEGMENTATION_MODEL=/models/diarization/segmentation.onnx \
    DIARIZATION_EMBEDDING_MODEL=/models/diarization/embedding.onnx \
    TEMPLATES_DIR=/app/templates \
    UPLOAD_DIR=/data/uploads \
    STATE_DIR=/data/state

COPY --from=build /opt/venv /opt/venv
COPY --from=build /models /models
COPY --from=build /licenses/models /licenses/models

WORKDIR /app
COPY ambient-scribe/apps/api/ambient_scribe /app/ambient_scribe
COPY ambient-scribe/apps/api/templates /app/templates
COPY --from=ui /ui/dist /app/ui

# Apache-2.0 §4: the licence, the upstream notices and what we changed travel
# inside the image, because an image is redistribution in object form.
COPY LICENSE MODIFICATIONS.md THIRD-PARTY-NOTICES.md SECURITY.md /licenses/

RUN mkdir -p /data/uploads /data/state \
    && chown -R scribe:scribe /data

USER scribe
EXPOSE 8000

# Loopback by default. Nothing in this image authenticates anybody: the console
# in front of it does, and it reaches this process over the pod's loopback
# interface. A deployment that binds this to 0.0.0.0 publishes a consultation
# archive to its whole namespace.
ENV API_HOST=127.0.0.1 API_PORT=8000

CMD ["sh", "-c", "exec uvicorn ambient_scribe.main:app --host \"$API_HOST\" --port \"$API_PORT\""]
