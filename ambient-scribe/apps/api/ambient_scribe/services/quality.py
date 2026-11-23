# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: the diarization reliability check, and the
# decision to degrade visibly rather than attribute silently.

"""Measure the thing that can be wrong, print it, and refuse below a floor.

A clinical note generated from a transcript with no speaker separation is the
worst failure this application can produce, because it does not look like a
failure. The text is fluent, the structure is right, the headings are the ones a
clinician expects — and the patient's symptoms are in the clinician's mouth. No
exception is raised, the health endpoint is green, the note arrives in seconds.

So every transcript carries a score, the score has floors, and a transcript below
a floor **loses its speaker labels** rather than showing uncertain ones. The
scores are computed from the result itself; none of them needs a reference
transcript, which is what makes them usable on a tenant's own recording rather
than only on a test fixture.

| score | what it means | why it catches a lie |
|---|---|---|
| `speakersFound` | clusters the diarizer settled on | 1 on a two-person consultation is the headline failure; it is reported, not patched over |
| `assignedWordFraction` | words that fell inside some speaker turn | low means the segmenter called most of the audio non-speech and the words it did place are a minority report |
| `meanSegmentPurity` | per-sentence agreement with that sentence's majority speaker | a Whisper segment is a sentence, and a sentence has one speaker; a diarizer flipping mid-sentence scores low here even when the overall split looks plausible |
| `speechCoverage` | share of the recording called speech | near 0 means silence or a format problem; near 1 on a consultation with pauses means the segmenter is admitting noise |

The floors are deployment configuration rather than constants, because a floor is
a judgement about how much attribution error a clinical reader will tolerate, and
that is the tenant's judgement. The defaults are the measured ones — see
`MODIFICATIONS.md` for the figures the fixtures produced.

`status` is what the rest of the application reads:

* `ok` — labels and roles are shown and the note generation is told who spoke;
* `unreliable` — a floor was missed. Labels are dropped, roles are not attempted,
  the note is generated from an unattributed transcript and says so;
* `single-speaker` — one speaker found. Not an error: a dictation. Roles are not
  attempted, and note generation is told the transcript is one voice;
* `absent` — diarization did not run at all (no model, or it raised). The
  transcript is still produced; nothing is attributed.
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger("ambient_scribe.quality")

STATUS_OK = "ok"
STATUS_UNRELIABLE = "unreliable"
STATUS_SINGLE = "single-speaker"
STATUS_ABSENT = "absent"

# Statuses in which a speaker label may be shown to a reader and a role may be inferred.
ATTRIBUTABLE = {STATUS_OK}


def score(segments, turns, audio_seconds: float) -> dict:
    """Score a diarization result. No reference transcript involved."""
    words = [w for s in segments for w in s.words]
    speakers = sorted({t.speaker for t in turns})
    if not words:
        return {"status": STATUS_ABSENT, "reason": "the recording produced no words",
                "speakersFound": len(speakers), "assignedWordFraction": 0.0,
                "meanSegmentPurity": 0.0, "speechCoverage": 0.0,
                "wordsScored": 0, "segmentsScored": 0, "turns": len(turns)}

    assigned = [w for w in words if w.speaker is not None]
    purities = []
    for s in segments:
        labelled = [w for w in s.words if w.speaker is not None]
        # Fewer than three words is a filler ("Right.", "Of course.") and its purity
        # is 1.0 by construction, which would flatter the mean.
        if len(labelled) < 3:
            continue
        majority = Counter(w.speaker for w in labelled).most_common(1)[0][1]
        purities.append(majority / len(labelled))

    covered = sum(t.end - t.start for t in turns)
    return {
        "speakersFound": len(speakers),
        "assignedWordFraction": round(len(assigned) / len(words), 4),
        "meanSegmentPurity": round(sum(purities) / len(purities), 4) if purities else 0.0,
        "speechCoverage": round(covered / audio_seconds, 4) if audio_seconds > 0 else 0.0,
        "wordsScored": len(words),
        "segmentsScored": len(purities),
        "turns": len(turns),
    }


def verdict(scores: dict, floors: dict) -> dict:
    """Turn scores into a status plus the reason, and never into a silent pass."""
    out = dict(scores)
    out["floors"] = dict(floors)

    if scores.get("status") == STATUS_ABSENT:
        return out

    missed = []
    if scores["assignedWordFraction"] < floors["assignedWordFraction"]:
        missed.append(
            f"only {scores['assignedWordFraction']:.0%} of the words fell inside a speaker turn "
            f"(floor {floors['assignedWordFraction']:.0%})")
    # One scored sentence is enough to fail on. A segment is scored only when it
    # has three or more labelled words, so a filler ("Right.", "Of course.") is
    # already excluded; what is left is a real sentence, and a real sentence with a
    # speaker change through the middle of it is the signal, not noise.
    if scores["segmentsScored"] >= 1 and scores["meanSegmentPurity"] < floors["meanSegmentPurity"]:
        missed.append(
            f"speaker changes cut across sentences: mean purity {scores['meanSegmentPurity']:.0%} "
            f"(floor {floors['meanSegmentPurity']:.0%})")
    if scores["speechCoverage"] < floors["speechCoverage"]:
        missed.append(
            f"only {scores['speechCoverage']:.0%} of the recording was identified as speech "
            f"(floor {floors['speechCoverage']:.0%})")

    if missed:
        out["status"] = STATUS_UNRELIABLE
        out["reason"] = "; ".join(missed)
    elif scores["speakersFound"] <= 1:
        out["status"] = STATUS_SINGLE
        out["reason"] = ("one voice was found in this recording, so nothing is attributed to a "
                         "speaker and no clinician/patient roles are inferred")
    else:
        out["status"] = STATUS_OK
        out["reason"] = ""
    return out


def apply(segments, result: dict) -> None:
    """Drop every speaker label the verdict does not permit us to show."""
    if result.get("status") in ATTRIBUTABLE:
        return
    for s in segments:
        s.speaker = None
        for w in s.words:
            w.speaker = None


def floors_from(settings) -> dict:
    return {
        "assignedWordFraction": settings.diarization_floor_assigned_word_fraction,
        "meanSegmentPurity": settings.diarization_floor_segment_purity,
        "speechCoverage": settings.diarization_floor_speech_coverage,
    }
