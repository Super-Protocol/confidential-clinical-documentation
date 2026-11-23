#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
"""Measure the thing that can be wrong, print it, and fail below a floor.

Runs the deployment's own recognition and diarization over the fixtures in
`fixtures/`, scores speaker attribution against their ground truth, and reports
transcription throughput per minute of audio — the two numbers a buyer of this
application asks for and the two a render test cannot produce.

It fails, rather than reporting, when:

* attribution accuracy on a two-speaker fixture falls below `--min-accuracy`;
* the quality probe's own verdict is not the expected one for that fixture —
  including on the dictation, where `single-speaker` is the PASS and finding two
  speakers is the failure the probe exists to catch.

    ASR_MODEL_DIR=/models/asr \
    DIARIZATION_SEGMENTATION_MODEL=/models/diarization/segmentation.onnx \
    DIARIZATION_EMBEDDING_MODEL=/models/diarization/embedding.onnx \
    python3 tests/measure.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "ambient-scribe" / "apps" / "api"))

# Two-speaker fixtures expect `ok`; the dictation expects `single-speaker`.
EXPECTED = {
    "consult_mixed": "ok",
    "consult_same_sex": "ok",
    "consult_noisy": "ok",
    "dictation": "single-speaker",
}


def ground_truth_speaker(turns, midpoint: float) -> str:
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
    return min(turns, key=lambda t: min(abs(t["start"] - midpoint),
                                        abs(t["end"] - midpoint)))["speaker"]


def attribution_accuracy(segments, truth) -> dict:
    """Word-weighted accuracy under the best mapping of cluster index to role.

    The best mapping, not a fixed one: a cluster index is arbitrary, so scoring
    against a fixed mapping would measure the diarizer's choice of label rather
    than its separation. What it cannot hide is a cluster that holds both people.
    """
    pairs = []
    for segment in segments:
        for word in segment.words:
            if word.speaker is None:
                pairs.append((ground_truth_speaker(truth["turns"],
                                                   (word.start + word.end) / 2), None))
                continue
            pairs.append((ground_truth_speaker(truth["turns"], (word.start + word.end) / 2),
                          word.speaker))
    labels = sorted({p[1] for p in pairs if p[1] is not None})
    best = (0, {})
    for permutation in itertools.permutations(["clinician", "patient"]):
        mapping = {label: permutation[i] for i, label in enumerate(labels[:2])}
        correct = sum(1 for truth_role, label in pairs
                      if label is not None and mapping.get(label) == truth_role)
        if correct > best[0]:
            best = (correct, mapping)
    return {"words": len(pairs), "correct": best[0],
            "accuracy": round(best[0] / len(pairs), 4) if pairs else 0.0,
            "unassignedWords": sum(1 for _, label in pairs if label is None),
            "mapping": {str(k): v for k, v in best[1].items()}}


def run_one(wav: Path, settings) -> dict:
    from ambient_scribe.services import diarization as diar
    from ambient_scribe.services import quality as quality_mod
    from ambient_scribe.services.asr import load_audio
    from ambient_scribe.services.transcription import SAMPLE_RATE, get_transcriber

    truth = json.loads(wav.with_suffix(".truth.json").read_text("utf-8"))
    audio = load_audio(wav)
    audio_seconds = len(audio) / SAMPLE_RATE

    transcriber = get_transcriber(settings)
    started = time.monotonic()
    segments = list(transcriber.transcribe(audio, language=settings.asr_language))
    asr_seconds = time.monotonic() - started

    diarizer = diar.get_diarizer(settings)
    started = time.monotonic()
    turns = diarizer.diarize(audio)
    diar_seconds = time.monotonic() - started

    words = [w for s in segments for w in s.words]
    diar.assign_speakers(words, turns)
    scores = quality_mod.score(segments, turns, audio_seconds)
    verdict = quality_mod.verdict(scores, quality_mod.floors_from(settings))

    # Score the attribution BEFORE the verdict is applied: when the verdict drops
    # the labels, what we want on the record is how good the labels it dropped were.
    accuracy = attribution_accuracy(segments, truth)
    quality_mod.apply(segments, verdict)

    return {
        "fixture": wav.stem,
        "audioSeconds": round(audio_seconds, 1),
        "asrSeconds": round(asr_seconds, 1),
        "diarizationSeconds": round(diar_seconds, 1),
        "secondsPerAudioMinute": round(60 * (asr_seconds + diar_seconds) / audio_seconds, 1),
        "asrRealtimeFactor": round(audio_seconds / asr_seconds, 2),
        "status": verdict["status"],
        "expectedStatus": EXPECTED.get(wav.stem, "ok"),
        "speakersFound": verdict["speakersFound"],
        "assignedWordFraction": verdict["assignedWordFraction"],
        "meanSegmentPurity": verdict["meanSegmentPurity"],
        "speechCoverage": verdict["speechCoverage"],
        "attribution": accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    from ambient_scribe.deps import get_settings
    settings = get_settings()

    wavs = [HERE / "fixtures" / f"{name}.wav" for name in (args.fixture or list(EXPECTED))]
    missing = [w.name for w in wavs if not w.exists()]
    if missing:
        print(f"missing fixtures: {', '.join(missing)}\n"
              f"generate them with: python3 {HERE / 'fixtures' / 'synthesize.py'}",
              file=sys.stderr)
        return 2

    results, failures = [], []
    for wav in wavs:
        result = run_one(wav, settings)
        results.append(result)
        print(json.dumps(result, indent=1), flush=True)

        if result["status"] != result["expectedStatus"]:
            failures.append(f"{result['fixture']}: verdict {result['status']}, "
                            f"expected {result['expectedStatus']}")
        if result["expectedStatus"] == "ok" and \
                result["attribution"]["accuracy"] < args.min_accuracy:
            failures.append(f"{result['fixture']}: attribution "
                            f"{result['attribution']['accuracy']:.1%} below "
                            f"{args.min_accuracy:.0%}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")

    print("\n--- summary")
    for result in results:
        print(f"  {result['fixture']:<18} {result['status']:<15} "
              f"attribution {result['attribution']['accuracy']:>7.1%}  "
              f"{result['secondsPerAudioMinute']:>5.1f}s per audio minute")
    for failure in failures:
        print(f"  FAIL  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
