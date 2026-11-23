# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: local speaker diarization, replacing the
# sortformer diarizer that only exists inside the Parakeet NIM.

"""Who spoke when — and an honest answer when we do not know.

This is the module the port of this blueprint turns on. Upstream calls
`add_speaker_diarization_to_config(config, True, 2)` unconditionally and
`detect_speaker_roles` then decides which speaker number is the clinician and
which is the patient. Whisper has no diarization of any kind, and a transcript
with no speaker separation still produces a fluent, well-structured SOAP note —
one that attributes the patient's symptoms to the clinician and the clinician's
plan to the patient. Nothing raises, every probe is green, and a clinician signs
it. So the capability is replaced rather than dropped, and the replacement
reports its own reliability.

**The stack**, all of it running on the CPU and all of it inside the image:

* segmentation — `pyannote/segmentation-3.0` exported to ONNX (MIT, CNRS);
* speaker embeddings — NVIDIA NeMo TitaNet-Large exported to ONNX (CC-BY-4.0);
* both driven by `sherpa-onnx` (Apache-2.0) over onnxruntime.

**The speaker count is discovered, not asserted.** Upstream passes 2 as a hint;
`sherpa-onnx` takes it as an instruction and splits one speaker into two clusters
on a dictated monologue, inventing a conversation. So this module clusters on an
embedding-distance threshold and reports however many speakers it found —
including one. A consultation that really has three people in the room gets
three, and a dictation gets one and is labelled as such.

**It refuses to guess.** `quality()` in `quality.py` scores the result
against floors the deployment declares, and a result below a floor is marked
unreliable: the speaker labels are dropped rather than shown, role detection is
not attempted, and note generation is told it is reading an unattributed
transcript. That is the whole point of this file.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("ambient_scribe.diarization")

SAMPLE_RATE = 16000


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


class Diarizer:
    """One diarization pipeline for the process, loaded on first use."""

    def __init__(self, segmentation_model: str, embedding_model: str, threads: int = 6,
                 threshold: float = 0.5, min_duration_on: float = 0.3,
                 min_duration_off: float = 0.5) -> None:
        self._segmentation_model = segmentation_model
        self._embedding_model = embedding_model
        self._threads = threads
        self._threshold = threshold
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        self._pipeline = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return os.path.exists(self._segmentation_model) and os.path.exists(self._embedding_model)

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    import sherpa_onnx

                    started = time.monotonic()
                    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                                model=self._segmentation_model),
                            num_threads=self._threads,
                        ),
                        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                            model=self._embedding_model, num_threads=self._threads),
                        # A threshold, never `num_clusters`: see the module docstring.
                        clustering=sherpa_onnx.FastClusteringConfig(threshold=self._threshold),
                        min_duration_on=self._min_duration_on,
                        min_duration_off=self._min_duration_off,
                    )
                    if not config.validate():
                        raise RuntimeError(
                            "speaker diarization configuration is invalid; "
                            f"segmentation={self._segmentation_model} embedding={self._embedding_model}")
                    self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
                    logger.info("diarization pipeline ready in %.1fs", time.monotonic() - started)
        return self._pipeline

    def diarize(self, audio: np.ndarray) -> list[Turn]:
        pipeline = self.load()
        if pipeline.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"diarization expects {pipeline.sample_rate} Hz, audio is {SAMPLE_RATE} Hz")
        result = pipeline.process(audio).sort_by_start_time()
        return [Turn(start=float(s.start), end=float(s.end), speaker=int(s.speaker))
                for s in result]


_diarizer: Diarizer | None = None
_diarizer_lock = threading.Lock()


def get_diarizer(settings) -> Diarizer:
    global _diarizer
    if _diarizer is None:
        with _diarizer_lock:
            if _diarizer is None:
                _diarizer = Diarizer(
                    segmentation_model=settings.diarization_segmentation_model,
                    embedding_model=settings.diarization_embedding_model,
                    threads=settings.asr_threads or (os.cpu_count() or 4),
                    threshold=settings.diarization_threshold,
                    min_duration_on=settings.diarization_min_duration_on,
                    min_duration_off=settings.diarization_min_duration_off,
                )
    return _diarizer


def assign_speakers(words, turns: list[Turn]) -> int:
    """Attach a speaker to each word by the turn it overlaps most. Returns the number assigned.

    A word that overlaps no turn keeps `speaker=None` — it is not given the nearest
    speaker, because `assignedWordFraction` is one of the floors and filling the
    gaps in would be filling in the evidence that the diarizer missed them.
    """
    assigned = 0
    for w in words:
        best_overlap, best_speaker = 0.0, None
        for t in turns:
            overlap = min(w.end, t.end) - max(w.start, t.start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, t.speaker
        w.speaker = best_speaker
        if best_speaker is not None:
            assigned += 1
    return assigned
