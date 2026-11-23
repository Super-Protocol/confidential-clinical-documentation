# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: Riva and NVIDIA-hosted-inference settings
# replaced by local ASR and diarization settings, persistence and retention
# added, and the diarization floors made deployment configuration.

"""FastAPI dependencies and dependency injection."""
from functools import lru_cache
from typing import Generator
import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    # API Configuration
    debug: bool = False
    api_title: str = "Ambient Scribe API"
    api_version: str = "0.1.0"
    
    # ---------------------------------------------------------------------
    # Generation. An OpenAI-compatible endpoint inside this cluster space; the
    # NVIDIA-hosted branch and its API key are gone, because reaching
    # integrate.api.nvidia.com with a consultation is the one thing a tenant
    # deploys this to avoid.
    # ---------------------------------------------------------------------
    openai_base_url: str = "http://swarm-model-server:8000/v1"
    llm_model: str = ""
    # The endpoint is cluster-internal and on no public address, but an
    # unauthenticated inference endpoint is an unauthenticated inference
    # endpoint; the client carries a key regardless.
    llm_api_key: str = ""
    llm_timeout_seconds: float = 600.0
    llm_licence: str = ""
    # How many transcript segments the role-detection prompt sees. Upstream used
    # ten; a consultation can open with pleasantries that name nobody.
    role_detection_segments: int = 24

    # ---------------------------------------------------------------------
    # Speech recognition — faster-whisper on the CPU, from a checkpoint baked
    # into the image. `asr_model_dir` is a directory and never a model name:
    # a name would make the first transcription of a fresh deployment reach
    # huggingface.co, and this deployment reaches nothing.
    # ---------------------------------------------------------------------
    asr_model_dir: str = "/models/asr"
    asr_compute_type: str = "int8"
    asr_threads: int = 0  # 0 = every core the container was given
    asr_beam_size: int = 1
    asr_vad_filter: bool = True
    asr_language: str = "en"

    # ---------------------------------------------------------------------
    # Speaker diarization — sherpa-onnx over pyannote segmentation (MIT) and
    # NeMo TitaNet-Large embeddings (CC-BY-4.0), also baked in.
    # ---------------------------------------------------------------------
    enable_speaker_diarization: bool = True
    diarization_segmentation_model: str = "/models/diarization/segmentation.onnx"
    diarization_embedding_model: str = "/models/diarization/embedding.onnx"
    # An embedding-distance threshold, NOT a speaker count. Upstream asserts two
    # speakers; measured on a dictated monologue, asserting two splits the one
    # voice into two clusters and invents a conversation. See MODIFICATIONS.md.
    diarization_threshold: float = 0.5
    diarization_min_duration_on: float = 0.3
    diarization_min_duration_off: float = 0.5

    # The floors a diarization result has to clear before a speaker label is
    # shown to anybody or a role is inferred. Below any of them the labels are
    # dropped and the note is generated from an unattributed transcript.
    diarization_floor_assigned_word_fraction: float = 0.9
    diarization_floor_segment_purity: float = 0.85
    diarization_floor_speech_coverage: float = 0.3
    
    # Streaming Configuration
    enable_streaming: bool = True
    streaming_chunk_size: int = 1600
    
    # Storage
    storage_backend: str = "local"  # "local"
    upload_dir: str = "./uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    # Added by Super Protocol, 2026. Upstream keeps transcripts and notes in two
    # module-level dicts, so a restart loses every note a clinician drafted while
    # the recording — the most sensitive artefact in the system — is the only
    # thing that survives. That is the wrong way round. Both are written here.
    state_dir: str = "./state"
    # How long an uploaded recording is kept for playback, after which it is
    # deleted and the transcript keeps its text and its timecodes. 0 = delete as
    # soon as the transcript exists; a negative number = keep indefinitely.
    audio_retention_hours: float = 24.0
    retention_sweep_seconds: float = 900.0
    
    # Templates
    templates_dir: str = "./templates"
    
    # Processing
    max_concurrent_transcriptions: int = 3
    transcription_timeout: int = 300  # 5 minutes
    
    # Which origins may call this API with credentials. Upstream sends
    # `allow_origins=["*"]` with `allow_credentials=True`; the console in front of
    # this API carries a session cookie, so a wildcard there is a cross-site
    # request away from somebody else's page reading a consultation.
    cors_allow_origins: str = ""

    # NeMo Guardrails is gone, and so is the switch. Upstream wires
    # `nemoguardrails` to the `nvidia_ai_endpoints` engine and gates it on
    # `enable_guardrails`, which defaults to false — and ships NO rails
    # configuration at all, so the directory it would load from does not exist in
    # the repository. A switch that cannot be turned on, in front of a safety
    # claim nobody can test, is worse than its absence: see MODIFICATIONS.md.
    # This deployment serves no content-safety model and the listing says so.
    
    # System Prompt Configuration
    system_prompt: str = "You are a medical professional creating clinical notes. Be concise and accurate."
    
    # ---------------------------------------------------------------------
    # Added by Super Protocol, 2026.
    #
    # A number that reaches this process through a Helm chart or a marketplace
    # parameter arrives as a STRING, and a large one arrives in scientific
    # notation: `MAX_FILE_SIZE=5.24288e+08`, because a YAML emitter printed a
    # float64. `int("5.24288e+08")` raises, the model fails to validate, and the
    # process dies on import — which on a confidential cluster space is a pod
    # that never becomes ready with no pod-level visibility to read the reason
    # out of. It cost this port fifty minutes of a billed GPU before a local
    # cluster showed the traceback, and the same trap bit SUP-890.
    #
    # The chart renders integers with `int64` now, which is the real fix. This is
    # the other half: whatever writes the value, the value is understood.
    # ---------------------------------------------------------------------
    @field_validator("max_file_size", "transcription_timeout", "asr_threads",
                     "asr_beam_size", "max_concurrent_transcriptions",
                     "streaming_chunk_size", "role_detection_segments", mode="before")
    @classmethod
    def _integer_however_written(cls, value):
        if isinstance(value, str) and value.strip():
            return int(float(value))
        return value

    class Config:
        env_file = ".env"
        case_sensitive = False

def _load_settings() -> Settings:
    """Internal function to load settings."""
    # Always load .env file, but only override in development for hot-reloading
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    load_dotenv(override=debug_mode)  # Always load, but only override in debug mode
    
    settings = Settings()
    
    # Optional: Print what values were loaded (for debugging)
    print("Settings loaded:")
    print(f"  debug: {settings.debug}")
    print(f"  asr_model_dir: {settings.asr_model_dir} ({settings.asr_compute_type})")
    print(f"  diarization: {'on' if settings.enable_speaker_diarization else 'off'} "
          f"threshold={settings.diarization_threshold}")
    print(f"  openai_base_url: {settings.openai_base_url} model={settings.llm_model}")
    print(f"  enable_streaming: {settings.enable_streaming}")
    
    return settings

# Cached version for production
@lru_cache()
def _get_settings_cached() -> Settings:
    """Get cached application settings for production."""
    return _load_settings()

def get_settings() -> Settings:
    """Get application settings with smart caching based on environment."""
    # In development (DEBUG=true), don't cache settings to allow hot-reloading
    # In production (DEBUG=false), use caching for performance
    if os.getenv("DEBUG", "false").lower() == "true":
        return _load_settings()  # No caching in development
    else:
        return _get_settings_cached()  # Cached in production

def get_upload_dir() -> Path:
    """Get upload directory path."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

def get_state_dir() -> Path:
    """Where transcripts and notes are written. Added by Super Protocol, 2026."""
    state_dir = Path(get_settings().state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_templates_dir() -> Path:
    """Get templates directory path."""
    settings = get_settings()
    templates_dir = Path(settings.templates_dir)
    if not templates_dir.exists():
        # Create templates dir and look for templates in parent directories
        templates_dir.mkdir(parents=True, exist_ok=True)
        # Check if we're in the nested structure and find the root templates
        current = Path.cwd()
        for parent in [current, *current.parents]:
            potential_templates = parent / "templates"
            if potential_templates.exists() and potential_templates.is_dir():
                return potential_templates
    return templates_dir
