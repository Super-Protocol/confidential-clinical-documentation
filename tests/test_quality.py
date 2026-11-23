# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
"""The diarization quality probe: does it fire when it should, and only then.

These run with no model and no audio. The probe is pure arithmetic over a
diarization result, which is deliberate — it is the piece that has to be right
when everything else is behaving, and a test needing a GPU is a test nobody runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ambient-scribe" / "apps" / "api"))

from ambient_scribe.services import quality as q  # noqa: E402
from ambient_scribe.services.diarization import Turn, assign_speakers  # noqa: E402
from ambient_scribe.services.transcription import Segment, Word  # noqa: E402

FLOORS = {"assignedWordFraction": 0.9, "meanSegmentPurity": 0.85, "speechCoverage": 0.3}


def seg(start, words, speaker_of=None):
    """A segment of one-second words beginning at `start`."""
    ws = [Word(start=start + i, end=start + i + 0.9, word=w) for i, w in enumerate(words)]
    if speaker_of is not None:
        for w, s in zip(ws, speaker_of):
            w.speaker = s
    return Segment(start=start, end=start + len(words), text=" ".join(words),
                   confidence=0.9, words=ws)


def test_two_clean_speakers_pass():
    segments = [seg(0, ["a", "b", "c", "d"], [0, 0, 0, 0]),
                seg(10, ["e", "f", "g", "h"], [1, 1, 1, 1])]
    turns = [Turn(0, 4, 0), Turn(10, 14, 1)]
    result = q.verdict(q.score(segments, turns, 14.0), FLOORS)
    assert result["status"] == q.STATUS_OK
    assert result["speakersFound"] == 2
    assert result["meanSegmentPurity"] == 1.0


def test_one_speaker_is_single_not_failure():
    segments = [seg(0, ["a", "b", "c", "d"], [0, 0, 0, 0])]
    turns = [Turn(0, 4, 0)]
    result = q.verdict(q.score(segments, turns, 4.0), FLOORS)
    assert result["status"] == q.STATUS_SINGLE
    assert "one voice" in result["reason"]


def test_speaker_flipping_inside_a_sentence_is_unreliable():
    """The case a plausible-looking overall split hides: purity, not speaker count."""
    segments = [seg(0, ["a", "b", "c", "d"], [0, 1, 0, 1]),
                seg(10, ["e", "f", "g", "h"], [1, 0, 1, 0])]
    turns = [Turn(0, 14, 0), Turn(0, 14, 1)]
    result = q.verdict(q.score(segments, turns, 14.0), FLOORS)
    assert result["status"] == q.STATUS_UNRELIABLE
    assert "cut across sentences" in result["reason"]


def test_mostly_unassigned_words_are_unreliable():
    segments = [seg(0, ["a", "b", "c", "d"], [0, None, None, None])]
    turns = [Turn(0, 1, 0)]
    result = q.verdict(q.score(segments, turns, 100.0), FLOORS)
    assert result["status"] == q.STATUS_UNRELIABLE
    assert "fell inside a speaker turn" in result["reason"]


def test_a_failed_verdict_drops_every_label():
    segments = [seg(0, ["a", "b", "c", "d"], [0, 1, 0, 1])]
    turns = [Turn(0, 4, 0), Turn(0, 4, 1)]
    result = q.verdict(q.score(segments, turns, 4.0), FLOORS)
    assert result["status"] == q.STATUS_UNRELIABLE
    q.apply(segments, result)
    assert all(w.speaker is None for s in segments for w in s.words)
    assert all(s.speaker is None for s in segments)


def test_a_passing_verdict_keeps_the_labels():
    segments = [seg(0, ["a", "b", "c", "d"], [0, 0, 0, 0]),
                seg(10, ["e", "f", "g", "h"], [1, 1, 1, 1])]
    turns = [Turn(0, 4, 0), Turn(10, 14, 1)]
    result = q.verdict(q.score(segments, turns, 14.0), FLOORS)
    q.apply(segments, result)
    assert [w.speaker for w in segments[0].words] == [0, 0, 0, 0]


def test_no_words_is_absent_and_not_a_pass():
    result = q.verdict(q.score([], [], 10.0), FLOORS)
    assert result["status"] == q.STATUS_ABSENT
    assert result["status"] not in q.ATTRIBUTABLE


def test_absent_is_never_attributable():
    """Nothing but `ok` may show a label. This is the whole contract."""
    for status in (q.STATUS_ABSENT, q.STATUS_UNRELIABLE, q.STATUS_SINGLE):
        assert status not in q.ATTRIBUTABLE
    assert q.STATUS_OK in q.ATTRIBUTABLE


def test_short_segments_do_not_flatter_the_purity_mean():
    """"Right." is one word and trivially pure; it must not outvote a real sentence."""
    segments = [seg(0, ["yes"], [0]),
                seg(5, ["no"], [1]),
                seg(10, ["a", "b", "c", "d"], [0, 1, 0, 1])]
    turns = [Turn(0, 14, 0), Turn(0, 14, 1)]
    scores = q.score(segments, turns, 14.0)
    assert scores["segmentsScored"] == 1
    assert scores["meanSegmentPurity"] == 0.5


def test_assign_speakers_leaves_a_word_outside_every_turn_unassigned():
    """A word not covered by a turn is not given the nearest speaker."""
    segment = seg(0, ["a", "b", "c", "d"])
    turns = [Turn(0, 2, 0)]
    assigned = assign_speakers(segment.words, turns)
    assert assigned == 2
    assert [w.speaker for w in segment.words] == [0, 0, None, None]


def test_assign_speakers_takes_the_turn_with_the_most_overlap():
    segment = seg(0, ["a"])
    # the word spans 0.0-0.9; the second turn covers 0.3-0.9 of it
    turns = [Turn(0.0, 0.3, 0), Turn(0.3, 2.0, 1)]
    assign_speakers(segment.words, turns)
    assert segment.words[0].speaker == 1


@pytest.mark.parametrize("coverage_seconds,expect", [(1.0, q.STATUS_UNRELIABLE), (12.0, q.STATUS_OK)])
def test_speech_coverage_floor(coverage_seconds, expect):
    segments = [seg(0, ["a", "b", "c", "d"], [0, 0, 0, 0]),
                seg(10, ["e", "f", "g", "h"], [1, 1, 1, 1])]
    turns = [Turn(0, coverage_seconds / 2, 0), Turn(10, 10 + coverage_seconds / 2, 1)]
    # every word is still assigned, so only the coverage floor can fail
    for s in segments:
        for w in s.words:
            pass
    scores = q.score(segments, turns, 20.0)
    scores["assignedWordFraction"] = 1.0
    scores["meanSegmentPurity"] = 1.0
    assert q.verdict(scores, FLOORS)["status"] == expect
