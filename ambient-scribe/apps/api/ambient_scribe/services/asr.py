# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: NVIDIA Riva replaced by local speech
# recognition and local speaker diarization, and role detection made conditional
# on a measured diarization result instead of unconditional.

"""Automatic speech recognition and speaker attribution, on the deployment's own CPU.

Upstream this module is a Riva gRPC client: `offline_recognize` against a Parakeet
NIM for the words, `add_speaker_diarization_to_config(config, True, 2)` for the
speakers, and `detect_speaker_roles` to decide which speaker number is the
clinician. The NIM needs an NGC credential and 75 GB of a card, and the whole
blueprint asks for five of them.

Here the words come from faster-whisper and the speakers from sherpa-onnx, both on
the CPU and both reading checkpoints baked into the image — see
`transcription.py` and `diarization.py`. The shape this module returns is
unchanged: a `Transcript` of `TranscriptSegment`s carrying `speaker_tag`, which
is what `format_transcript_with_speakers` and the UI already read.

**What is NOT unchanged, and is the point of the port**: a speaker label is shown
only when the diarization result passes the deployment's floors. `quality.py`
scores every transcript, the score travels with it, and a transcript that did not
pass arrives with no labels, no roles, and a stated reason. Upstream attributes
unconditionally; a clinical note that puts the patient's symptoms in the
clinician's mouth is fluent, well-formed and signed by a human who had no way to
know.

Riva's timestamp-repair helpers are gone with Riva. They existed because Riva
returned word times that disagreed with one another (`fix_inconsistent_timestamps`,
`regenerate_timestamps_from_text`, `add_estimated_timestamps`) and the repair had
no input to work on here; Whisper's word timestamps are monotonic by construction.
`mock_transcribe_audio_file` is gone too: a hard-coded doctor/patient exchange is
not a thing to leave in an image that produces clinical documents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import numpy as np

from ambient_scribe.deps import Settings
from ambient_scribe.models import Transcript, TranscriptSegment
from ambient_scribe.services import diarization as diar
from ambient_scribe.services.audio import decode
from ambient_scribe.services import quality as quality_mod
from ambient_scribe.services.transcription import SAMPLE_RATE, get_transcriber

logger = logging.getLogger("ambient_scribe.asr")

# One transcription at a time. CTranslate2 and onnxruntime are both internally
# threaded over the whole CPU allocation, so a second concurrent job makes both
# slower and makes the published throughput figure meaningless.
_engine_lock = asyncio.Lock()


def serialize_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime objects to ISO format strings for JSON serialization."""
    serialized = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def load_audio(file_path: Path) -> np.ndarray:
    """Decode an uploaded recording to the 16 kHz mono the engines want.

    Modified by Super Protocol, 2026. Upstream's `convert_to_wav` did this with
    librosa, because Riva wanted LINEAR_PCM at 16 kHz over the wire. Nothing goes
    over a wire now and `services/audio.py` decodes with PyAV, which
    faster-whisper already carries — see that module for the 850 MB of image this
    saves and the two traps in it.
    """
    return decode(file_path, SAMPLE_RATE)


def _to_transcript_segments(segments) -> List[TranscriptSegment]:
    out: List[TranscriptSegment] = []
    for s in segments:
        out.append(TranscriptSegment(
            start=round(s.start, 3),
            end=round(max(s.end, s.start), 3),
            text=s.text,
            speaker_tag=s.speaker,
            confidence=round(s.confidence, 4),
        ))
    return out


def _segment_speaker(segment) -> int | None:
    """The segment's speaker is its words' majority, or None if no word has one."""
    from collections import Counter

    labelled = [w.speaker for w in segment.words if w.speaker is not None]
    if not labelled:
        return None
    return Counter(labelled).most_common(1)[0][0]


async def detect_speaker_roles(transcript: Transcript, settings: Settings) -> dict:
    """Ask the deployment's own model which speaker is the patient and which the clinician.

    Modified by Super Protocol, 2026. Three changes, all of them about not guessing:

    * it is called only when `quality.py` returned `ok`, so there are at least two
      speakers and the attribution they rest on was measured;
    * it refuses a reply it cannot parse, and a reply that does not name two
      DISTINCT speakers that both appear in the transcript — upstream accepted
      `{"patient": 1, "provider": 1}` and produced a transcript in which one
      speaker was both people;
    * upstream's "The speaker that is not the patient is most likely the doctor,
      so be generous" is gone. Generosity is the failure mode.

    An empty dict means "roles unknown", and every caller renders that as unknown
    rather than filling it in.
    """
    try:
        from openai import AsyncOpenAI

        present = sorted({s.speaker_tag for s in transcript.segments if s.speaker_tag is not None})
        if len(present) < 2:
            return {}

        sample = [f"Speaker {s.speaker_tag}: {s.text}"
                  for s in transcript.segments[:settings.role_detection_segments]
                  if s.speaker_tag is not None]
        if not sample:
            return {}

        prompt = (
            "Below is an excerpt of a transcribed clinical consultation. Each line is "
            "labelled with the speaker number the diarizer assigned.\n\n"
            f"{chr(10).join(sample)}\n\n"
            "Decide which speaker number is the patient and which is the clinician. "
            "Answer with a single JSON object and nothing else: "
            '{\"patient\": <speaker number>, \"clinician\": <speaker number>}. '
            "If the excerpt does not make it clear which is which, answer "
            '{\"patient\": null, \"clinician\": null} instead of choosing.'
        )

        client = AsyncOpenAI(api_key=settings.llm_api_key or "not-used",
                            base_url=settings.openai_base_url,
                            timeout=settings.llm_timeout_seconds)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system",
                 "content": "You label speakers in clinical transcripts. Answer with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        raw = (response.choices[0].message.content or "").strip()
        # A model that wraps JSON in a fence is answering; a model that wrote prose is not.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            logger.warning("role detection: no JSON object in the reply; roles left unknown")
            return {}
        parsed = json.loads(raw[start:end + 1])

        patient = parsed.get("patient")
        clinician = parsed.get("clinician", parsed.get("provider"))
        if patient is None or clinician is None:
            logger.info("role detection: the model declined to choose; roles left unknown")
            return {}
        try:
            patient, clinician = int(patient), int(clinician)
        except (TypeError, ValueError):
            logger.warning("role detection: speaker numbers were not numbers; roles left unknown")
            return {}
        if patient == clinician or patient not in present or clinician not in present:
            logger.warning("role detection: reply named %s/%s against speakers %s; "
                           "roles left unknown", patient, clinician, present)
            return {}

        return {patient: "patient", clinician: "provider"}

    except Exception as exc:  # noqa: BLE001 - a failed role detection is not a failed transcript
        logger.warning("role detection failed, roles left unknown: %s", exc)
        return {}


async def _run_engines(audio: np.ndarray, settings: Settings) -> tuple[list, list, dict]:
    """Transcribe, diarize, attach speakers, and score. Blocking work off the event loop."""
    transcriber = get_transcriber(settings)
    diarizer = diar.get_diarizer(settings)
    audio_seconds = len(audio) / SAMPLE_RATE

    def work():
        timings = {}
        t0 = time.monotonic()
        segments = list(transcriber.transcribe(audio, language=settings.asr_language))
        timings["asrSeconds"] = round(time.monotonic() - t0, 2)

        turns: list = []
        if settings.enable_speaker_diarization and diarizer.available:
            t0 = time.monotonic()
            try:
                turns = diarizer.diarize(audio)
                timings["diarizationSeconds"] = round(time.monotonic() - t0, 2)
            except Exception as exc:  # noqa: BLE001
                logger.error("diarization failed, transcript will carry no speakers: %s", exc)
                turns = []
                timings["diarizationSeconds"] = round(time.monotonic() - t0, 2)
                timings["diarizationError"] = str(exc)
        words = [w for s in segments for w in s.words]
        if turns:
            diar.assign_speakers(words, turns)
            scores = quality_mod.score(segments, turns, audio_seconds)
        else:
            reason = ("speaker diarization is switched off for this deployment"
                      if not settings.enable_speaker_diarization
                      else "speaker diarization did not produce a result")
            scores = {**quality_mod.score(segments, [], audio_seconds),
                      "status": quality_mod.STATUS_ABSENT, "reason": reason}
        result = quality_mod.verdict(scores, quality_mod.floors_from(settings))
        quality_mod.apply(segments, result)
        for s in segments:
            s.speaker = _segment_speaker(s)
        result["audioSeconds"] = round(audio_seconds, 2)
        result.update(timings)
        if timings.get("asrSeconds"):
            result["secondsPerAudioMinute"] = round(
                60.0 * (timings["asrSeconds"] + timings.get("diarizationSeconds", 0.0))
                / max(audio_seconds, 0.001), 2)
        return segments, turns, result

    async with _engine_lock:
        return await asyncio.to_thread(work)


async def transcribe_audio_file(
    file_path: Path,
    transcript_id: str,
    filename: str,
    settings: Settings,
) -> Transcript:
    """Transcribe an uploaded audio file and attribute its speakers."""
    audio = load_audio(file_path)
    segments, _turns, result = await _run_engines(audio, settings)
    ts = _to_transcript_segments(segments)

    transcript = Transcript(
        id=transcript_id,
        segments=ts,
        language=settings.asr_language,
        duration=round(len(audio) / SAMPLE_RATE, 3),
        filename=filename,
        created_at=datetime.now(),
    )
    transcript.diarization = result
    logger.info("transcript %s: %s", transcript_id, json.dumps(result))

    if result.get("status") in quality_mod.ATTRIBUTABLE:
        transcript.speaker_roles = await detect_speaker_roles(transcript, settings)
    else:
        transcript.speaker_roles = {}
    return transcript


async def stream_transcribe_audio_file(
    file_path: Path,
    transcript_id: str,
    filename: str,
    settings: Settings,
) -> AsyncGenerator[dict, None]:
    """Yield each sentence as the decoder finishes it, then the attributed transcript.

    Modified by Super Protocol, 2026. Upstream chunked a *file* into a Riva
    streaming call to get interim results, and every interim result carried a
    speaker tag from a diarizer that had only heard the audio so far. Diarization
    is a whole-recording problem — the speaker of the first sentence is decided by
    how it compares with the last — so partials here carry NO speaker, and the
    speakers arrive once, in the `complete` event, with the score that says
    whether to trust them.

    Upstream also never called `detect_speaker_roles` on this path at all (the
    line is commented out in its source), while `ENABLE_STREAMING` defaults to
    true — so upstream's headline role attribution was off by default. It is on
    here, on both paths, subject to the same floors.
    """
    try:
        if not settings.enable_streaming:
            transcript = await transcribe_audio_file(file_path, transcript_id, filename, settings)
            yield {"type": "final", "transcript": serialize_for_json(transcript.dict())}
            return

        audio = load_audio(file_path)
        audio_seconds = len(audio) / SAMPLE_RATE
        yield {"type": "status", "transcript_id": transcript_id,
               "message": f"Transcribing {audio_seconds / 60:.1f} minutes of audio"}

        transcriber = get_transcriber(settings)
        diarizer = diar.get_diarizer(settings)

        # The decoder is a blocking generator. Drain it on a worker thread and hand
        # each segment to the event loop through a queue, so the sentences reach the
        # browser while the rest of the recording is still being decoded.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        timings: dict = {}

        def decode():
            try:
                t0 = time.monotonic()
                for seg in transcriber.transcribe(audio, language=settings.asr_language):
                    loop.call_soon_threadsafe(queue.put_nowait, ("segment", seg))
                timings["asrSeconds"] = round(time.monotonic() - t0, 2)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        async with _engine_lock:
            worker = asyncio.create_task(asyncio.to_thread(decode))
            collected = []
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    raise payload
                if kind == "done":
                    break
                collected.append(payload)
                yield {
                    "type": "final_segment",
                    "segment": {"start": round(payload.start, 3),
                                "end": round(max(payload.end, payload.start), 3),
                                "text": payload.text,
                                # No speaker yet, and that is deliberate — see the docstring.
                                "speaker_tag": None,
                                "confidence": round(payload.confidence, 4)},
                    "speaker": "Speaker",
                    "speaker_tag": None,
                }
            await worker

            yield {"type": "status", "transcript_id": transcript_id,
                   "message": "Identifying who spoke"}

            def attribute():
                turns: list = []
                if settings.enable_speaker_diarization and diarizer.available:
                    t0 = time.monotonic()
                    try:
                        turns = diarizer.diarize(audio)
                        timings["diarizationSeconds"] = round(time.monotonic() - t0, 2)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("diarization failed: %s", exc)
                        timings["diarizationSeconds"] = round(time.monotonic() - t0, 2)
                        timings["diarizationError"] = str(exc)
                words = [w for s in collected for w in s.words]
                if turns:
                    diar.assign_speakers(words, turns)
                    scores = quality_mod.score(collected, turns, audio_seconds)
                else:
                    reason = ("speaker diarization is switched off for this deployment"
                              if not settings.enable_speaker_diarization
                              else "speaker diarization did not produce a result")
                    scores = {**quality_mod.score(collected, [], audio_seconds),
                              "status": quality_mod.STATUS_ABSENT, "reason": reason}
                res = quality_mod.verdict(scores, quality_mod.floors_from(settings))
                quality_mod.apply(collected, res)
                for s in collected:
                    s.speaker = _segment_speaker(s)
                res["audioSeconds"] = round(audio_seconds, 2)
                res.update(timings)
                res["secondsPerAudioMinute"] = round(
                    60.0 * (timings.get("asrSeconds", 0.0) + timings.get("diarizationSeconds", 0.0))
                    / max(audio_seconds, 0.001), 2)
                return res

            result = await asyncio.to_thread(attribute)

        transcript = Transcript(
            id=transcript_id,
            segments=_to_transcript_segments(collected),
            language=settings.asr_language,
            duration=round(audio_seconds, 3),
            filename=filename,
            created_at=datetime.now(),
        )
        transcript.diarization = result
        logger.info("transcript %s: %s", transcript_id, json.dumps(result))

        if result.get("status") in quality_mod.ATTRIBUTABLE:
            transcript.speaker_roles = await detect_speaker_roles(transcript, settings)
        else:
            transcript.speaker_roles = {}

        yield {"type": "complete", "transcript": serialize_for_json(transcript.dict())}

    except Exception as exc:  # noqa: BLE001
        logger.error("transcription failed: %s", exc, exc_info=True)
        yield {"type": "error", "error": f"Transcription failed: {exc}"}
