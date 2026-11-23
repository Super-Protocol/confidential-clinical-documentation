# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: decoding an uploaded recording, replacing the
# librosa path and the separately installed ffmpeg binary.

"""Decode whatever container the recording arrived in, to 16 kHz mono float32.

Upstream decodes with `librosa.load`. That works, and it brings numba, llvmlite,
scipy and scikit-learn with it — about 400 MB of image for one function — on top
of an apt-installed `ffmpeg` and `libsndfile1`, another 450 MB. All of it is
redundant here: **faster-whisper already depends on PyAV, which is a complete
ffmpeg**, so the decoder this application needs is in the environment either way.
Dropping librosa and the apt packages takes about 850 MB off the image, which in
a confidential cluster space is pull time a tenant waits through.

Two notes for whoever syncs this with upstream.

**Do not call `faster_whisper.audio.decode_audio`.** It passes
`metadata_errors="ignore"` to `av.open`, an argument PyAV removed; with
faster-whisper 1.2.1 and PyAV 19 it raises `TypeError` on every file. The library's
own helper is broken against its own resolved dependency, which is exactly the
kind of thing a render test and a readiness probe both pass.

**The resampler has to be flushed.** `resample(None)` returns the frames still
held in its buffers, and without it the last fraction of a second of every
recording is silently missing — a missing-words bug that looks like a
transcription problem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger("ambient_scribe.audio")

SAMPLE_RATE = 16000


def decode(path: Path | str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return the recording as mono float32 at `sample_rate`.

    Raises `ValueError` for a file with no audio stream — an uploaded PDF, or a
    recording that failed to save — so the caller answers "that is not a
    recording" rather than transcribing silence.
    """
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("the file contains no audio stream")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        # Flush. See the module docstring.
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError("the file decoded to no audio")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    logger.info("decoded %s: %.1fs at %d Hz", Path(path).name, len(audio) / sample_rate,
                sample_rate)
    return audio
