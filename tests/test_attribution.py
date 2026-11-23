# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
"""What reaches the model about who spoke — and that it is never a guess.

Upstream's three silent guesses are the subject of these tests: the keyword
fallback in the prompt builder, the cluster-index fallback in the UI (covered in
`MODIFICATIONS.md` §3 and by the viewer's own code), and a role-detection reply
that names one speaker as both people.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ambient-scribe" / "apps" / "api"))

from ambient_scribe.models import Transcript, TranscriptSegment  # noqa: E402
from ambient_scribe.services.llm import format_transcript_with_speakers  # noqa: E402


def transcript(segments, roles=None, diarization=None):
    return Transcript(id="t", segments=segments, language="en", duration=60.0,
                      filename="consult.wav", created_at=datetime(2026, 11, 23),
                      speaker_roles=roles or {}, diarization=diarization)


def s(text, tag=None, start=0.0):
    return TranscriptSegment(start=start, end=start + 5, text=text, speaker_tag=tag,
                             confidence=0.9)


def test_attributed_transcript_names_the_roles():
    out = format_transcript_with_speakers(transcript(
        [s("Where is the pain?", 0), s("In my chest.", 1, 10)],
        roles={0: "provider", 1: "patient"},
        diarization={"status": "ok", "speakersFound": 2}))
    assert "Speaker 0 is the provider" in out
    assert "Speaker 1 is the patient" in out
    assert "Provider (Speaker 0): Where is the pain?" in out
    assert "Patient (Speaker 1): In my chest." in out


def test_unattributed_transcript_says_so_and_forbids_attribution():
    """The keyword guess upstream used is the thing this replaces."""
    out = format_transcript_with_speakers(transcript(
        # "my symptoms" and "recommend" are exactly the phrases upstream matched on
        [s("I have been experiencing my symptoms for weeks."),
         s("I recommend a statin and an ECG.", start=10)],
        diarization={"status": "unreliable", "speakersFound": 2,
                     "reason": "speaker changes cut across sentences"}))
    assert "NOT attributed" in out
    assert "Do NOT state or imply who said anything" in out
    assert "speaker changes cut across sentences" in out
    # and no role word is put in front of either line
    assert "Patient (" not in out and "Provider (" not in out
    assert "Unidentified speaker: I have been experiencing" in out


def test_unreliable_with_several_voices_reports_the_count_without_using_it():
    out = format_transcript_with_speakers(transcript(
        [s("one"), s("two", start=10)],
        diarization={"status": "unreliable", "speakersFound": 3, "reason": "r"}))
    assert "appears to contain 3 voices" in out
    assert "not reliable enough to use" in out


def test_single_speaker_does_not_claim_a_dictation_and_attributes_nothing():
    """One cluster is a dictation OR a conversation the clusterer merged. We cannot tell.

    The first version of this port said "Treat it as dictation by the clinician", which put the
    patient's symptoms in the clinician's mouth whenever a two-person consultation came back as
    one cluster — and in that case every floor passes, because `meanSegmentPurity` is 1.0 by
    construction with one speaker. It was a fourth guess in a port whose point was removing
    three.
    """
    out = format_transcript_with_speakers(transcript(
        [s("Patient attended for review of chest pain.", 0)],
        diarization={"status": "single-speaker", "speakersFound": 1,
                     "reason": "one voice was detected: either a dictation, or a conversation "
                               "whose speakers were not separated"}))
    assert "ONE voice was detected" in out
    assert "cannot tell which" in out
    assert "UNATTRIBUTED" in out
    assert "Do NOT state or imply who said anything" in out
    # the words the old version used, and the claim this test exists to forbid
    assert "dictation by the clinician" not in out
    assert "Treat it as dictation" not in out


# Every status but `ok` and `provided`. A header that names a role in ANY of them is the defect
# this whole module is about, so it is asked as one parametrised question rather than per case.
ROLE_WORDS = ("clinician", "doctor", "physician", "patient", "provider", "nurse")


@pytest.mark.parametrize("status,speakers", [
    ("single-speaker", 1),
    ("unreliable", 2),
    ("unreliable", 1),
    ("absent", 0),
    ("something-a-later-version-added", 3),
])
@pytest.mark.parametrize("tags", [(0, 1), (None, None)],
                         ids=["cluster-numbers-kept", "labels-dropped"])
def test_an_unattributed_header_never_names_a_role(status, speakers, tags):
    """No status but `ok` and `provided` may put a role in front of a line, or claim one.

    Both label shapes are asked, because the pipeline drops the tags itself
    (`quality.apply`) and a transcript that arrives with them anyway — written by an older
    version, or by an operator — must still not be attributed. A NUMBERED speaker is fine: a
    cluster index carries no role information and says so.
    """
    out = format_transcript_with_speakers(transcript(
        [s("I have chest pain and shortness of breath.", tags[0]),
         s("I will arrange an ECG and start you on aspirin.", tags[1], 10)],
        diarization={"status": status, "speakersFound": speakers, "reason": "a stated reason"}))

    lowered = out.lower()
    for role in ROLE_WORDS:
        # A role word may appear in the INSTRUCTION ("do not infer roles", "who said anything")
        # and in the transcript's own words. What it may never be is a LABEL in front of a line,
        # or a claim about whose voice this recording holds.
        assert f"{role} (speaker" not in lowered, f"{role} labelled a line under {status}"
        assert f"] {role}:" not in lowered, f"{role} labelled a line under {status}"
    assert "dictation by the" not in lowered
    assert "treat it as dictation" not in lowered

    # What it says instead: a cluster number, or nothing at all.
    if tags == (0, 1):
        assert "Speaker 0:" in out and "Speaker 1:" in out
    else:
        assert "Unidentified speaker:" in out

    # And the instruction is in every one of them, including a status no version has yet.
    assert "Do NOT state or imply who said anything" in out


def test_absent_diarization_is_treated_as_unattributed():
    out = format_transcript_with_speakers(transcript(
        [s("I feel short of breath.")],
        diarization={"status": "absent", "speakersFound": 0,
                     "reason": "speaker diarization is switched off for this deployment"}))
    assert "NOT attributed" in out
    assert "switched off for this deployment" in out


def test_no_diarization_field_at_all_still_refuses_to_attribute():
    """An older transcript on the volume, from before the score existed."""
    out = format_transcript_with_speakers(transcript([s("I feel unwell.")]))
    assert "NOT attributed" in out


def test_labels_without_roles_are_numbered_not_named():
    out = format_transcript_with_speakers(transcript(
        [s("a", 0), s("b", 1, 10)], diarization={"status": "ok", "speakersFound": 2}))
    assert "Speaker 0: a" in out
    assert "Speaker 1: b" in out


class _Reply:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,expected", [
    ('{"patient": 1, "clinician": 0}', {1: "patient", 0: "provider"}),
    ('```json\n{"patient": 1, "clinician": 0}\n```', {1: "patient", 0: "provider"}),
    ('{"patient": 1, "provider": 0}', {1: "patient", 0: "provider"}),
    # upstream accepted this: one speaker as both people
    ('{"patient": 1, "clinician": 1}', {}),
    # the model is given a way to decline, and declining is honoured
    ('{"patient": null, "clinician": null}', {}),
    # a speaker number that is not in the transcript
    ('{"patient": 7, "clinician": 0}', {}),
    ("I think speaker 1 is the patient.", {}),
    ("", {}),
])
async def test_role_detection_refuses_everything_but_a_clean_answer(monkeypatch, reply, expected):
    from ambient_scribe.services import asr

    class _Completions:
        async def create(self, **_kwargs):
            return _Reply(reply)

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)

    from ambient_scribe.deps import Settings
    settings = Settings(llm_model="m", openai_base_url="http://x/v1")
    t = transcript([s("a", 0), s("b", 1, 10)], diarization={"status": "ok", "speakersFound": 2})
    assert await asr.detect_speaker_roles(t, settings) == expected


@pytest.mark.asyncio
async def test_role_detection_is_not_attempted_with_one_speaker():
    from ambient_scribe.deps import Settings
    from ambient_scribe.services import asr
    settings = Settings(llm_model="m", openai_base_url="http://x/v1")
    t = transcript([s("a", 0)], diarization={"status": "single-speaker", "speakersFound": 1})
    # no HTTP client is patched here: reaching one would be the failure
    assert await asr.detect_speaker_roles(t, settings) == {}


def test_a_provided_attribution_is_shown_and_credited():
    """An imported transcript's labels are the data owner's, and are named as theirs."""
    out = format_transcript_with_speakers(transcript(
        [s("Where is the pain?", 0), s("In my chest.", 1, 10)],
        roles={0: "provider", 1: "patient"},
        diarization={"status": "provided", "speakersFound": 2,
                     "source": "acme-consultations"}))
    assert "already speaker-separated from acme-consultations" in out
    assert "Provider (Speaker 0): Where is the pain?" in out


def test_an_imported_transcript_maps_role_words_and_leaves_the_rest_numbered():
    import asyncio

    from ambient_scribe.routers.transcribe import ImportedSegment, ImportedTranscript, import_transcript

    result = asyncio.get_event_loop().run_until_complete(import_transcript(ImportedTranscript(
        source="acme-consultations", name="2026-11-20-harper",
        segments=[
            ImportedSegment(start=0, end=5, text="Good morning.", speaker="Clinician"),
            ImportedSegment(start=5, end=9, text="Morning, doctor.", speaker="patient"),
            ImportedSegment(start=9, end=12, text="I will take the bloods.", speaker="phlebotomist"),
        ])))
    assert result.diarization["status"] == "provided"
    assert result.diarization["source"] == "acme-consultations"
    assert result.diarization["speakersFound"] == 3
    # "Clinician" and "patient" are role words; "phlebotomist" is not, and is not guessed at
    assert result.speaker_roles == {0: "provider", 1: "patient"}
    assert [seg.speaker_tag for seg in result.segments] == [0, 1, 2]


def test_an_imported_transcript_with_no_speaker_column_is_not_attributed():
    import asyncio

    from ambient_scribe.routers.transcribe import ImportedSegment, ImportedTranscript, import_transcript

    result = asyncio.get_event_loop().run_until_complete(import_transcript(ImportedTranscript(
        source="acme-consultations", name="unlabelled",
        segments=[ImportedSegment(start=0, end=5, text="Good morning.")])))
    assert result.diarization["status"] == "absent"
    assert result.speaker_roles == {}
    assert "carry no speaker column" in result.diarization["reason"]
