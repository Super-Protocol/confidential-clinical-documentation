# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
"""Transcripts and notes that survive a restart, and a recording that does not outstay it."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ambient-scribe" / "apps" / "api"))

from ambient_scribe.models import Transcript, TranscriptSegment  # noqa: E402
from ambient_scribe.utils.store import (  # noqa: E402
    JsonStore, audio_path_for, forget_audio, sweep_audio,
)


def a_transcript(tid="t1", filename="consult.wav"):
    return Transcript(id=tid, segments=[TranscriptSegment(start=0, end=1, text="hello",
                                                          speaker_tag=0, confidence=0.9)],
                      language="en", duration=1.0, filename=filename,
                      created_at=datetime(2026, 11, 23))


def test_a_store_reloads_what_was_written(tmp_path):
    store = JsonStore(tmp_path / "transcripts", Transcript)
    store["t1"] = a_transcript("t1")
    assert len(store) == 1

    # a new process, the same directory
    reopened = JsonStore(tmp_path / "transcripts", Transcript)
    assert "t1" in reopened
    assert reopened["t1"].segments[0].text == "hello"


def test_delete_removes_the_document(tmp_path):
    store = JsonStore(tmp_path / "t", Transcript)
    store["t1"] = a_transcript("t1")
    del store["t1"]
    assert "t1" not in JsonStore(tmp_path / "t", Transcript)
    assert not list((tmp_path / "t").glob("*.json"))


def test_a_key_from_a_url_cannot_escape_the_directory(tmp_path):
    store = JsonStore(tmp_path / "t", Transcript)
    for key in ("../escape", "a/b", "", "." * 200):
        with pytest.raises(KeyError):
            store[key] = a_transcript("x")


def test_an_unreadable_document_does_not_stop_the_rest_loading(tmp_path):
    store = JsonStore(tmp_path / "t", Transcript)
    store["good"] = a_transcript("good")
    (tmp_path / "t" / "bad.json").write_text("{not json", encoding="utf-8")
    reopened = JsonStore(tmp_path / "t", Transcript)
    assert set(reopened) == {"good"}


def test_a_write_leaves_no_partial_file(tmp_path):
    store = JsonStore(tmp_path / "t", Transcript)
    store["t1"] = a_transcript("t1")
    assert not list((tmp_path / "t").glob("*.tmp"))


def test_forget_audio_deletes_the_recording_and_nothing_else(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    recording = uploads / "t1_consult.wav"
    recording.write_bytes(b"RIFF")
    other = uploads / "t2_other.wav"
    other.write_bytes(b"RIFF")

    t = a_transcript("t1")
    assert audio_path_for(uploads, t) == recording
    assert forget_audio(uploads, t) is True
    assert not recording.exists()
    assert other.exists()
    # idempotent: a second delete is not an error
    assert forget_audio(uploads, t) is False


def test_sweep_deletes_only_what_has_aged_out(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old = uploads / "old.wav"
    old.write_bytes(b"RIFF")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    fresh = uploads / "fresh.wav"
    fresh.write_bytes(b"RIFF")

    result = sweep_audio(uploads, retention_hours=1.0)
    assert result == {"deleted": 1, "kept": 1, "retentionHours": 1.0}
    assert not old.exists() and fresh.exists()


def test_a_negative_retention_keeps_everything(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old = uploads / "old.wav"
    old.write_bytes(b"RIFF")
    os.utime(old, (0, 0))
    assert sweep_audio(uploads, retention_hours=-1.0)["deleted"] == 0
    assert old.exists()


def test_zero_retention_deletes_on_the_next_pass(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "now.wav").write_bytes(b"RIFF")
    time.sleep(0.01)
    assert sweep_audio(uploads, retention_hours=0.0)["deleted"] == 1
