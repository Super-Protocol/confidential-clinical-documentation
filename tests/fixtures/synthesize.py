#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
# SPDX-License-Identifier: Apache-2.0
"""Synthesise consultation audio with per-turn ground truth.

The fixtures are generated rather than committed: four WAV files are forty
megabytes, and a generator is reproducible where a binary blob is only trusted.
Two different piper voices read the two sides of `consultation.json`, so every
word has a known speaker and `measure.py` can score attribution without a human
labelling anything.

**The honest limitation, which belongs next to any figure this produces:**
synthesised speech is cleaner than a consulting room. No crosstalk, no overlap,
no chair scraping, no accent the model has not heard, no speakerphone. These
fixtures measure whether the pipeline is wired correctly and whether the quality
probe fires when it should. They are an upper bound on accuracy, not a clinical
validation.

    pip install piper-tts soundfile librosa numpy
    python3 -m piper.download_voices --download-dir voices \
        en_GB-alan-medium en_US-amy-medium en_US-lessac-medium
    python3 synthesize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
SR = 16000


def _voice(name: str):
    from piper import PiperVoice
    for base in (HERE / "voices", HERE.parent / "voices", Path("voices")):
        path = base / f"{name}.onnx"
        if path.exists():
            return PiperVoice.load(str(path))
    raise SystemExit(f"voice {name} not found; see this file's docstring")


def build(script: Path, out_wav: Path, clinician: str, patient: str,
          gap: float = 0.25, noise: float = 0.0, seed: int = 7) -> None:
    from piper import SynthesisConfig

    rng = np.random.default_rng(seed)
    turns = json.loads(script.read_text("utf-8"))
    voices = {"clinician": _voice(clinician), "patient": _voice(patient)}
    configs = {
        "clinician": SynthesisConfig(length_scale=1.0, volume=1.0),
        "patient": SynthesisConfig(length_scale=1.06, volume=0.92),
    }

    pieces, truth, clock = [], [], 0.0
    for role, text in turns:
        voice = voices[role]
        audio = np.concatenate(
            [c.audio_float_array for c in voice.synthesize(text, syn_config=configs[role])]
        ).astype(np.float32)
        if voice.config.sample_rate != SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=voice.config.sample_rate, target_sr=SR)
        duration = len(audio) / SR
        truth.append({"speaker": role, "start": round(clock, 3),
                      "end": round(clock + duration, 3), "text": text})
        pieces.append(audio)
        clock += duration
        silence = max(0.05, gap + rng.normal(0, 0.08))
        pieces.append(np.zeros(int(silence * SR), dtype=np.float32))
        clock += silence

    mix = np.concatenate(pieces)
    if noise > 0:
        mix = mix + rng.normal(0, noise, len(mix)).astype(np.float32)
    mix = (mix / (np.max(np.abs(mix)) or 1.0) * 0.9).astype(np.float32)
    sf.write(out_wav, mix, SR)
    out_wav.with_suffix(".truth.json").write_text(
        json.dumps({"sampleRate": SR, "duration": round(len(mix) / SR, 3), "turns": truth},
                   indent=1), encoding="utf-8")
    print(f"{out_wav.name}  {len(mix) / SR:.1f}s  {len(truth)} turns")


def main() -> int:
    consult = HERE / "consultation.json"
    dictation = HERE / "dictation.json"
    build(consult, HERE / "consult_mixed.wav", "en_GB-alan-medium", "en_US-amy-medium")
    build(consult, HERE / "consult_same_sex.wav", "en_GB-alan-medium", "en_US-lessac-medium")
    build(consult, HERE / "consult_noisy.wav", "en_GB-alan-medium", "en_US-amy-medium", noise=0.004)
    build(dictation, HERE / "dictation.wav", "en_GB-alan-medium", "en_US-amy-medium")
    return 0


if __name__ == "__main__":
    sys.exit(main())
