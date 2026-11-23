# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: transcripts and notes survive a restart, and
# a recording does not outlive the retention the deployment declares.

"""A dict that is also a directory.

Upstream keeps transcripts and notes in two module-level dicts
(`routers/transcribe.py::_transcripts`, `routers/notes.py::_notes`). The
consequence on a cluster is the wrong way round: a pod restart — an eviction, a
node drain, a `reconfigure` — loses every note a clinician drafted, while the
uploaded recording, the most sensitive artefact in the system, is the one thing
that persists because it is a file on a volume.

So both stores become this: a `MutableMapping` with one JSON document per key on
the deployment's own volume. Every existing call site (`in`, `[]`, `pop`,
`len`, `.values()`) keeps working, which is why this is the shape rather than a
database.

**Writes are atomic.** A note is written to a temporary file in the same
directory and renamed over its target, so a pod killed mid-write leaves either
the old document or the new one and never half of either.

**Deleting a transcript deletes its recording.** That is the tenant-facing half
of the retention decision: "delete" has to mean the audio as well, or it means
nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("ambient_scribe.store")

# Keys are identifiers this application generated (uuid4) — but a key reaches
# this class from a URL path, so it is validated rather than trusted.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class JsonStore(MutableMapping):
    """One JSON file per entry, loaded once at startup and written on every change."""

    def __init__(self, directory: Path, model) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}
        self._load()

    def _path(self, key: str) -> Path:
        if not _SAFE_KEY.match(str(key)):
            raise KeyError(f"unsafe store key: {key!r}")
        return self._dir / f"{key}.json"

    def _load(self) -> None:
        for path in sorted(self._dir.glob("*.json")):
            try:
                self._cache[path.stem] = self._model(**json.loads(path.read_text("utf-8")))
            except Exception as exc:  # noqa: BLE001 - one bad file is not a dead deployment
                logger.error("ignoring unreadable %s: %s", path.name, exc)
        if self._cache:
            logger.info("loaded %d %s from %s", len(self._cache),
                        self._model.__name__, self._dir)

    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        path = self._path(key)
        with self._lock:
            payload = value.model_dump_json(indent=1) if hasattr(value, "model_dump_json") \
                else json.dumps(value.dict(), default=str, indent=1)
            fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            self._cache[key] = value

    def __delitem__(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            del self._cache[key]
            path.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self._cache))

    def __len__(self) -> int:
        return len(self._cache)


def audio_path_for(upload_dir: Path, transcript) -> Path | None:
    """The recording a transcript was made from, if it is still on disk."""
    if not transcript.filename:
        return None
    candidate = Path(upload_dir) / f"{transcript.id}_{transcript.filename}"
    return candidate if candidate.exists() else None


def forget_audio(upload_dir: Path, transcript) -> bool:
    """Delete a transcript's recording. The text and the timecodes stay."""
    path = audio_path_for(upload_dir, transcript)
    if path is None:
        return False
    try:
        path.unlink()
        logger.info("deleted the recording for transcript %s", transcript.id)
        return True
    except OSError as exc:
        logger.error("could not delete %s: %s", path, exc)
        return False


def sweep_audio(upload_dir: Path, retention_hours: float) -> dict:
    """Delete recordings older than the retention this deployment declares.

    `retention_hours < 0` keeps recordings indefinitely — the tenant's choice, and
    the listing says what it costs. `0` means delete as soon as the transcript
    exists, which the sweep achieves on its next pass; the transcript's text and
    timecodes are unaffected either way, so a note can still be generated and
    read. What is lost is audio playback.
    """
    upload_dir = Path(upload_dir)
    if retention_hours < 0 or not upload_dir.is_dir():
        return {"deleted": 0, "kept": 0, "retentionHours": retention_hours}
    cutoff = time.time() - retention_hours * 3600.0
    deleted = kept = 0
    for path in upload_dir.iterdir():
        if not path.is_file() or path.suffix == ".json":
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink()
                deleted += 1
            else:
                kept += 1
        except OSError as exc:
            logger.error("retention sweep could not handle %s: %s", path, exc)
    if deleted:
        logger.info("retention sweep deleted %d recording(s) older than %.1fh",
                    deleted, retention_hours)
    return {"deleted": deleted, "kept": kept, "retentionHours": retention_hours}
