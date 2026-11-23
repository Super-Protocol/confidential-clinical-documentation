# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: transcripts are persisted, a transcript can
# be deleted with its recording, an already-attributed transcript can be imported
# from a gated corpus, and an upload is accepted on what it is rather than on the
# Content-Type the browser guessed.

"""Transcription router."""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional
import uuid
import json
import asyncio
from pathlib import Path

from datetime import datetime

from pydantic import BaseModel

from ambient_scribe.models import ErrorResponse, Transcript, TranscriptSegment
from ambient_scribe.services.quality import STATUS_ABSENT, STATUS_PROVIDED
from ambient_scribe.services.asr import transcribe_audio_file, stream_transcribe_audio_file
from ambient_scribe.deps import get_settings, get_state_dir, get_upload_dir
from ambient_scribe.utils.store import JsonStore, forget_audio

router = APIRouter()

# One JSON document per transcript on the deployment's own volume, not a dict that
# a restart empties. The mapping interface is unchanged, so every call site below
# and in `routers/notes.py` reads the same.
_transcripts = JsonStore(get_state_dir() / "transcripts", Transcript)

import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Added by Super Protocol, 2026. Upstream rejects anything whose Content-Type does
# not begin `audio/`, and a browser sends `application/octet-stream` for plenty of
# ordinary recordings — an .m4a from a dictaphone among them. The extension decides,
# and the decoder is the real gate: see services/audio.py, which either reads the
# container or says it cannot.
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".oga",
                  ".opus", ".webm", ".wma", ".aiff", ".aif", ".amr", ".3gp"}


def _require_audio(file: UploadFile, logger) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").lower()
    if content_type.startswith("audio/") or content_type.startswith("video/") \
            or suffix in AUDIO_SUFFIXES:
        return
    logger.warning("rejected upload: filename=%s content_type=%s", file.filename, content_type)
    raise HTTPException(
        status_code=400,
        detail=("That does not look like a recording. Accepted: "
                + ", ".join(sorted(AUDIO_SUFFIXES))))


@router.post("/file", response_model=Transcript)
@limiter.limit("5/minute")
async def transcribe_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings=Depends(get_settings),
    upload_dir: Path = Depends(get_upload_dir)
) -> Transcript:
    """Transcribe an uploaded audio file."""

    logger = logging.getLogger("ambient_scribe.transcribe")
    logger.debug(f"Received file upload: filename={file.filename}, content_type={file.content_type}")

    _require_audio(file, logger)

    # Check file size
    # Note: UploadFile does not always have .size attribute, so log what we can
    file_size = getattr(file, "size", None)
    logger.debug(f"File size: {file_size}")
    if file_size and file_size > settings.max_file_size:
        logger.warning(f"File too large: {file_size} > {settings.max_file_size}")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size} bytes"
        )

    # Generate unique ID and save file
    transcript_id = str(uuid.uuid4())
    file_path = upload_dir / f"{transcript_id}_{file.filename}"
    logger.info(f"Saving uploaded file to: {file_path}")

    try:
        # Save uploaded file
        content = await file.read()
        logger.debug(f"Read {len(content)} bytes from uploaded file")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"File saved successfully: {file_path}")

        # Transcribe audio
        logger.info(f"Starting transcription: file_path={file_path}, transcript_id={transcript_id}")
        transcript = await transcribe_audio_file(
            file_path=file_path,
            transcript_id=transcript_id,
            filename=file.filename,
            settings=settings
        )
        logger.info(f"Transcription completed: transcript_id={transcript_id}")

        # Add audio URL to transcript (via API proxy)
        transcript.audio_url = f"/api/uploads/{transcript_id}_{file.filename}"
        
        # Store transcript
        _transcripts[transcript_id] = transcript
        logger.debug(f"Transcript stored in memory: transcript_id={transcript_id}")

        # Keep the audio file (don't schedule cleanup for now)
        # This allows the audio player to access the file
        logger.debug(f"Audio file preserved for playback: {file_path}")

        return transcript

    except Exception as e:
        logger.error(f"Exception during transcription: {e}", exc_info=True)
        # Cleanup file if transcription failed
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up file after failure: {file_path}")
            except Exception as cleanup_exc:
                logger.error(f"Failed to cleanup file: {file_path}, error: {cleanup_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@router.post("/stream", response_model=None)
@limiter.limit("5/minute")
async def stream_transcribe_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings=Depends(get_settings),
    upload_dir: Path = Depends(get_upload_dir)
):
    """Stream transcribe an uploaded audio file with real-time updates."""

    logger = logging.getLogger("ambient_scribe.transcribe")
    logger.debug(f"Received streaming file upload: filename={file.filename}, content_type={file.content_type}")

    _require_audio(file, logger)

    # Check file size
    file_size = getattr(file, "size", None)
    logger.debug(f"File size: {file_size}")
    if file_size and file_size > settings.max_file_size:
        logger.warning(f"File too large: {file_size} > {settings.max_file_size}")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size} bytes"
        )

    # Generate unique ID and save file
    transcript_id = str(uuid.uuid4())
    file_path = upload_dir / f"{transcript_id}_{file.filename}"
    logger.info(f"Saving uploaded file to: {file_path}")

    try:
        # Save uploaded file
        content = await file.read()
        logger.debug(f"Read {len(content)} bytes from uploaded file")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"File saved successfully: {file_path}")

        # Create streaming generator
        async def stream_transcription():
            try:
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Starting transcription...', 'transcript_id': transcript_id})}\n\n"
                
                # Stream transcription results
                last_result = None
                async for result in stream_transcribe_audio_file(
                    file_path=file_path,
                    transcript_id=transcript_id,
                    filename=file.filename,
                    settings=settings
                ):
                    yield f"data: {json.dumps(result)}\n\n"
                    last_result = result
                
                # Store final transcript if completed and send audio_url
                if last_result and last_result.get('type') == 'complete' and 'transcript' in last_result:
                    transcript_data = last_result['transcript']
                    transcript_obj = Transcript(**transcript_data)
                    # Add audio URL
                    audio_url = f"/api/uploads/{transcript_id}_{file.filename}"
                    transcript_obj.audio_url = audio_url
                    _transcripts[transcript_id] = transcript_obj
                    
                    # Send audio_url as a separate event for frontend
                    yield f"data: {json.dumps({'type': 'audio_url', 'audio_url': audio_url, 'transcript_id': transcript_id})}\n\n"
                    logger.info(f"Streaming transcription completed: transcript_id={transcript_id}")

            except Exception as e:
                logger.error(f"Streaming transcription error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            stream_transcription(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

    except Exception as e:
        logger.error(f"Exception during streaming transcription setup: {e}", exc_info=True)
        # Cleanup file if setup failed
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up file after failure: {file_path}")
            except Exception as cleanup_exc:
                logger.error(f"Failed to cleanup file: {file_path}, error: {cleanup_exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Streaming transcription setup failed: {str(e)}"
        )

@router.get("/stream/{transcript_id}")
async def stream_transcription_status(transcript_id: str):
    """Stream transcription progress for existing transcript (placeholder for progress tracking)."""
    
    async def generate_progress():
        # Check if transcript exists
        if transcript_id in _transcripts:
            transcript_data = _transcripts[transcript_id].dict()
            yield f"data: {json.dumps({'type': 'complete', 'transcript': transcript_data})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Transcript not found'})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/{transcript_id}", response_model=Transcript)
async def get_transcript(transcript_id: str) -> Transcript:
    """Get a transcript by ID."""
    if transcript_id not in _transcripts:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found"
        )
    
    transcript = _transcripts[transcript_id]
    
    # Fix audio URL format if it doesn't have the /api prefix
    if transcript.audio_url and not transcript.audio_url.startswith('/api/uploads/'):
        if transcript.audio_url.startswith('/uploads/'):
            transcript.audio_url = '/api' + transcript.audio_url
            _transcripts[transcript_id] = transcript  # Update in memory
    
    return transcript

@router.get("/", response_model=list[Transcript])
async def list_transcripts() -> list[Transcript]:
    """List all transcripts."""
    transcripts = []
    for transcript_id, transcript in _transcripts.items():
        # Fix audio URL format if it doesn't have the /api prefix
        if transcript.audio_url and not transcript.audio_url.startswith('/api/uploads/'):
            if transcript.audio_url.startswith('/uploads/'):
                transcript.audio_url = '/api' + transcript.audio_url
                _transcripts[transcript_id] = transcript  # Update in memory
        transcripts.append(transcript)
    
    return transcripts

# ---------------------------------------------------------------------------
# Added by Super Protocol, 2026: a transcript that already exists elsewhere.
#
# The marketplace gatekeeper serves a data owner's ClickHouse table to a
# deployment whose published evidence the owner approved, and can withdraw it the
# moment that evidence changes. A consultation ALREADY transcribed — by another
# system, by a human typist — is therefore something a tenant can bring here
# under the owner's control, and generate a note from inside the enclave without
# the recording ever existing in this deployment.
#
# Its speaker labels are the owner's. Nothing here measured them, so nothing here
# claims to have: the transcript is `provided`, it records where it came from, and
# the note generation is told so in those words.
# ---------------------------------------------------------------------------
ROLE_WORDS = {
    "clinician": "provider", "doctor": "provider", "physician": "provider",
    "provider": "provider", "gp": "provider", "nurse": "provider",
    "patient": "patient", "client": "patient",
}


class ImportedSegment(BaseModel):
    start: float = 0.0
    end: float = 0.0
    text: str
    speaker: Optional[str] = None


class ImportedTranscript(BaseModel):
    source: str
    name: str
    segments: List[ImportedSegment]
    recorded_at: Optional[str] = None


@router.post("/import", response_model=Transcript)
async def import_transcript(payload: ImportedTranscript) -> Transcript:
    """Create a transcript from rows a data owner has approved this deployment to read."""
    if not payload.segments:
        raise HTTPException(status_code=400, detail="The transcript has no segments")

    # Speaker strings become stable numbers in order of first appearance, and a
    # string this deployment recognises as a role becomes that role. A string it
    # does not recognise stays a numbered speaker rather than being guessed at.
    numbers: dict[str, int] = {}
    roles: dict[int, str] = {}
    segments: List[TranscriptSegment] = []
    for item in payload.segments:
        tag = None
        if item.speaker:
            key = item.speaker.strip().lower()
            if key not in numbers:
                numbers[key] = len(numbers)
                role = ROLE_WORDS.get(key)
                if role:
                    roles[numbers[key]] = role
            tag = numbers[key]
        segments.append(TranscriptSegment(
            start=round(float(item.start), 3),
            end=round(max(float(item.end), float(item.start)), 3),
            text=item.text.strip(),
            speaker_tag=tag,
            confidence=None,
        ))

    transcript_id = str(uuid.uuid4())
    attributed = len(numbers) > 0
    transcript = Transcript(
        id=transcript_id,
        segments=segments,
        language="en",
        duration=max((s.end for s in segments), default=0.0),
        filename=payload.name,
        created_at=datetime.now(),
        speaker_roles=roles,
        diarization={
            "status": STATUS_PROVIDED if attributed else STATUS_ABSENT,
            "source": payload.source,
            "speakersFound": len(numbers),
            "reason": (f"the speaker labels arrived with the transcript from {payload.source}; "
                       "nothing in this deployment measured them")
            if attributed else
            (f"the rows imported from {payload.source} carry no speaker column, so nothing "
             "is attributed"),
            "recordedAt": payload.recorded_at or "",
        },
    )
    _transcripts[transcript_id] = transcript
    logger = logging.getLogger("ambient_scribe.transcribe")
    logger.info("imported transcript %s from %s: %d segments, %d speakers",
                transcript_id, payload.source, len(segments), len(numbers))
    return transcript


@router.delete("/{transcript_id}")
async def delete_transcript(
    transcript_id: str,
    settings=Depends(get_settings),
    upload_dir: Path = Depends(get_upload_dir),
):
    """Delete a transcript and the recording it was made from.

    Added by Super Protocol, 2026. "Delete" has to reach the audio or it means
    nothing: the recording is the protected health information, the transcript is
    a derivative of it. Notes generated from this transcript are left alone and
    are deleted individually — a signed note is a clinical record and deleting
    one as a side effect of tidying up a recording is not a decision this
    endpoint gets to make.
    """
    if transcript_id not in _transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found")
    transcript = _transcripts[transcript_id]
    audio_deleted = forget_audio(upload_dir, transcript)
    del _transcripts[transcript_id]
    return {
        "message": f"Transcript {transcript_id} deleted",
        "audio_deleted": audio_deleted,
    }


@router.delete("/{transcript_id}/audio")
async def delete_transcript_audio(
    transcript_id: str,
    upload_dir: Path = Depends(get_upload_dir),
):
    """Delete only the recording, keeping the transcript's text and timecodes."""
    if transcript_id not in _transcripts:
        raise HTTPException(status_code=404, detail="Transcript not found")
    transcript = _transcripts[transcript_id]
    deleted = forget_audio(upload_dir, transcript)
    if deleted:
        transcript.audio_url = None
        _transcripts[transcript_id] = transcript
    return {"message": f"Recording for {transcript_id} deleted" if deleted
            else "No recording was held for this transcript", "audio_deleted": deleted}


async def cleanup_file(file_path: Path):
    """Background task to cleanup uploaded files."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass  # Ignore cleanup errors
