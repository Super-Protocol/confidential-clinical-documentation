# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: cross-origin credentials are no longer
# granted to every origin, the uploads mount follows the configured directory,
# the built UI is served from this process, and a retention sweep runs.

"""FastAPI main application module."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ambient_scribe.deps import get_settings, get_upload_dir
from ambient_scribe.routers import health, notes, templates, transcribe
from ambient_scribe.utils.store import sweep_audio

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ambient_scribe")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sweep expired recordings while the process runs.

    Added by Super Protocol, 2026. The retention a deployment declares has to be
    enforced by something; a recording is deleted when it ages out whether or not
    anybody opens the application again. One pass runs at startup, so a pod that
    was down past a recording's expiry does not resurrect it as a kept file.
    """
    upload_dir = get_upload_dir()

    async def sweeper():
        while True:
            try:
                await asyncio.to_thread(sweep_audio, upload_dir, settings.audio_retention_hours)
            except Exception as exc:  # noqa: BLE001 - a failed sweep must not stop the API
                logger.error("retention sweep failed: %s", exc)
            await asyncio.sleep(max(60.0, settings.retention_sweep_seconds))

    task = asyncio.create_task(sweeper()) if settings.audio_retention_hours >= 0 else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(
    title="Ambient Scribe API",
    description="Clinical transcription and note generation API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS. Upstream sends `allow_origins=["*"]` together with
# `allow_credentials=True` — a combination the browser refuses anyway, and one
# that would mean any page on the internet could read a consultation through the
# session cookie the console in front of this API sets. Empty by default: the UI
# is served from this same origin, so nothing legitimate is cross-origin.
_origins = [o.strip() for o in (settings.cors_allow_origins or "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

# Static files for uploads/storage. Upstream hard-codes `./uploads`, which is not
# the directory `UPLOAD_DIR` names once the two differ.
uploads_dir = get_upload_dir()
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
app.mount("/api/uploads", StaticFiles(directory=str(uploads_dir)), name="api_uploads")

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcription"])
app.include_router(notes.router, prefix="/api/notes", tags=["notes"])
app.include_router(templates.router, prefix="/api/templates", tags=["templates"])


# The built UI, from this same process. Upstream serves it from a second nginx
# container; one origin means no CORS to configure and one place for the console
# in front of this API to proxy to.
UI_DIR = Path(os.environ.get("UI_DIR", "/app/ui"))


@app.get("/api")
async def api_root():
    """Root of the API surface."""
    return {"message": "Ambient Scribe API", "version": settings.api_version}


if (UI_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIR / "assets")), name="ui_assets")

    @app.get("/{full_path:path}")
    async def ui(full_path: str):
        """Serve the single-page UI, and its own files where they exist."""
        candidate = (UI_DIR / full_path).resolve()
        if full_path and candidate.is_file() and UI_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(UI_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Ambient Scribe API", "version": settings.api_version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ambient_scribe.main:app", host="127.0.0.1", port=8000, reload=True)
