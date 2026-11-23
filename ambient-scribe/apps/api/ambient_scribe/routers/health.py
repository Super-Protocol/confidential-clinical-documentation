# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health check router.

Modified by Super Protocol, 2026: the dependencies this reports are the ones
this deployment actually has — the ASR and diarization checkpoints on disk and
the generation endpoint — rather than whether an NVIDIA API key was set.

What it deliberately does NOT do is probe the generation endpoint. A health
check that calls the model turns an engine still loading its weights into an
unhealthy container, and Kubernetes then restarts the pod that was almost ready.
"""
import os

from fastapi import APIRouter, Depends

from ambient_scribe.deps import get_settings
from ambient_scribe.models import HealthResponse
from ambient_scribe.services.diarization import get_diarizer
from ambient_scribe.services.transcription import get_transcriber

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check(settings=Depends(get_settings)) -> HealthResponse:
    """Basic health check endpoint."""
    
    # Check service dependencies
    services = {}
    
    transcriber = get_transcriber(settings)
    services["asr"] = "loaded" if transcriber.loaded else (
        "ready" if os.path.isdir(transcriber.model_dir) else "missing")

    if not settings.enable_speaker_diarization:
        services["diarization"] = "disabled"
    else:
        diarizer = get_diarizer(settings)
        services["diarization"] = "loaded" if diarizer.loaded else (
            "ready" if diarizer.available else "missing")

    services["generation"] = "configured" if settings.llm_model else "not_configured"
    
    # Check templates directory
    services["templates"] = "ok" if os.path.exists(settings.templates_dir) else "missing"
    
    # Check upload directory
    services["storage"] = "ok" if os.path.exists(settings.upload_dir) else "missing"
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        services=services
    )
