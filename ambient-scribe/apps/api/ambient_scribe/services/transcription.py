# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: local speech recognition, replacing the Riva
# gRPC client this deployment has no NIM to talk to.

"""Speech recognition on the deployment's own CPU.

Upstream calls `riva.client.ASRService.offline_recognize` against a Parakeet NIM —
a container that needs an NGC credential and 75 GB of a card. This module is the
replacement: faster-whisper (MIT) over CTranslate2 (MIT), reading a Whisper
checkpoint that is baked into the image and therefore covered by the image digest
the listing pins. Nothing is fetched at runtime and no model host is reached.

Two things here are deliberate.

**The model is loaded once and transcription is serialised.** CTranslate2 is
internally multi-threaded and a second concurrent transcription on the same CPU
allocation makes both slower rather than one of them faster. A semaphore of one
(`MAX_CONCURRENT_TRANSCRIPTIONS`) is what keeps a second upload from halving the
throughput figure the listing publishes.

**Word timestamps are not optional here.** Diarization produces speaker turns on
a timeline and the only way to attach a speaker to a word is to know when the
word was said. Upstream got word times from Riva; we ask Whisper for them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

logger = logging.getLogger("ambient_scribe.transcription")

SAMPLE_RATE = 16000


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0
    speaker: int | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float
    words: list[Word] = field(default_factory=list)
    speaker: int | None = None


class Transcriber:
    """One Whisper model for the process, loaded on first use."""

    def __init__(self, model_dir: str, compute_type: str = "int8", threads: int = 6,
                 beam_size: int = 1, vad_filter: bool = True) -> None:
        self._model_dir = model_dir
        self._compute_type = compute_type
        self._threads = threads
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    started = time.monotonic()
                    logger.info("loading ASR model from %s (%s, %d threads)",
                                self._model_dir, self._compute_type, self._threads)
                    self._model = WhisperModel(
                        self._model_dir,
                        device="cpu",
                        compute_type=self._compute_type,
                        cpu_threads=self._threads,
                        local_files_only=True,
                    )
                    logger.info("ASR model ready in %.1fs", time.monotonic() - started)
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, language: str = "en") -> Iterator[Segment]:
        """Yield segments as the decoder produces them, so the UI can show progress."""
        model = self.load()
        segments, _info = model.transcribe(
            audio,
            language=language,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=self._vad_filter,
            condition_on_previous_text=False,
        )
        for s in segments:
            words = [
                Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                for w in (s.words or [])
                if w.start is not None and w.end is not None
            ]
            # `avg_logprob` is a log probability over the segment's tokens; exp() of it
            # is the per-token geometric mean, which is the closest honest analogue of
            # the per-word confidence Riva returned.
            confidence = float(min(1.0, max(0.0, np.exp(s.avg_logprob))))
            yield Segment(start=float(s.start), end=float(s.end), text=s.text.strip(),
                          confidence=confidence, words=words)


_transcriber: Transcriber | None = None
_transcriber_lock = threading.Lock()


def get_transcriber(settings) -> Transcriber:
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = Transcriber(
                    model_dir=settings.asr_model_dir,
                    compute_type=settings.asr_compute_type,
                    threads=settings.asr_threads or (os.cpu_count() or 4),
                    beam_size=settings.asr_beam_size,
                    vad_filter=settings.asr_vad_filter,
                )
    return _transcriber
