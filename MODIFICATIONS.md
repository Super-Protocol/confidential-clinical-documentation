# What this fork changes, and why

Forked from [NVIDIA-AI-Blueprints/ambient-provider](https://github.com/NVIDIA-AI-Blueprints/ambient-provider)
at release **v1.0.0**, commit **`eb4c0be5`** (`main` HEAD at fork time, 2026-11-23), Apache-2.0.
`LICENSE`, `THIRD-PARTY-NOTICES.md` and `SECURITY.md` are upstream's and are unmodified; upstream
ships no `NOTICE` file, checked at fork time. Every file we changed keeps its SPDX header and
carries a `Modified by Super Protocol, 2026:` line beneath it.

Upstream is the **provider agent** of NVIDIA's ambient-healthcare-agents blueprint — the one that
turns a recorded consultation into a clinical note. The sibling patient agent
(`ambient-patient`, Pipecat / WebRTC / real-time voice) is a different product and is not in this
fork at all.

This fork exists to make that application deployable and verifiable inside a confidential cluster
space: no NVIDIA NIM container, no NGC credential, no NVIDIA AI Enterprise entitlement, no network
destination outside the deployment, and a stated reliability for the one thing that can be
confidently wrong.

NVIDIA is a trademark of NVIDIA Corporation. This fork is not produced, endorsed or supported by
NVIDIA.

---

## 1. Speech recognition: Riva/Parakeet NIM → faster-whisper on the CPU

`apps/api/ambient_scribe/services/asr.py` was a Riva gRPC client
(`riva.client.ASRService.offline_recognize`) against `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us`.
That container needs an NGC API key and, by upstream's own README, 75 GB of a card — part of a
footprint of **five A100s / ≈ 400 GB** across the blueprint.

Replaced by **faster-whisper 1.2.1** (MIT) over **CTranslate2** (MIT), serving
`Systran/faster-distil-whisper-large-v3` (MIT) from a checkpoint baked into the image, int8 on the
CPU. New module: `services/transcription.py`. `nvidia-riva-client` is gone from the dependencies.

Riva's timestamp-repair helpers went with it — `extract_time`, `fix_inconsistent_timestamps`,
`regenerate_timestamps_from_text`, `add_estimated_timestamps` and `process_riva_response` existed
because Riva returned word times that disagreed with one another, and Whisper's word timestamps are
monotonic by construction. `mock_transcribe_audio_file`, a hard-coded doctor/patient exchange, is
also gone: an image that produces clinical documents should not be able to invent one.

## 2. Speaker diarization: sortformer inside the NIM → sherpa-onnx on the CPU

This is the change the port turns on, and it has three parts.

**The replacement.** Upstream's diarizer is `sortformer`, reachable only inside the Parakeet NIM.
faster-whisper has no diarization at all. New module `services/diarization.py` runs
**sherpa-onnx 2.0.3** (Apache-2.0) over **pyannote/segmentation-3.0 exported to ONNX** (MIT, CNRS)
and **NVIDIA NeMo TitaNet-Large exported to ONNX** (CC-BY-4.0, attribution required and given),
both on the CPU and both inside the image.

**The speaker count is discovered, not asserted.** Upstream calls
`add_speaker_diarization_to_config(config, True, 2)` with the comment "2 is just a hint". It is not
a hint to this clustering implementation, and asserting it is measurably wrong: on a **dictated
monologue** fixture, `num_clusters=2` returns **two** speakers and 10 turns — one voice split into a
conversation that did not happen. Clustering on an embedding-distance threshold instead returns
**one** speaker on the same audio, and still exactly **two** on both two-speaker fixtures at every
threshold from 0.4 to 0.7. The default is `0.5`.

**It reports its own reliability.** New module `services/quality.py` scores every transcript —
speakers found, the fraction of words that fell inside a speaker turn, per-sentence speaker purity,
and how much of the recording was called speech — and compares them against floors the deployment
declares. A result below a floor is `unreliable`: the speaker labels are **dropped**, role detection
is not attempted, and note generation is told it is reading an unattributed transcript. One voice is
`single-speaker`, which is a dictation and not an error. The score travels on the transcript
(`Transcript.diarization`), is logged as one JSON line, and is shown in the UI.

Why this rather than a best guess: a transcript with no speaker separation still produces a fluent,
correctly structured SOAP note. It just puts the patient's symptoms in the clinician's mouth and the
clinician's plan in the patient's. Nothing raises, the health endpoint is green, and a human signs
it.

Measured end to end by `tests/measure.py` over four synthesised fixtures (`tests/fixtures/`,
3.1–5.5 minutes each, two piper voices with per-turn ground truth), on 6 CPU threads:

| fixture | speakers found | verdict | word-weighted attribution | mean segment purity |
|---|---|---|---|---|
| different-sex voices | 2 | `ok` | **98.7 %** | 99.4 % |
| same-sex voices | 2 | `ok` | **97.2 %** | 98.0 % |
| different-sex + noise | 2 | `ok` | **98.7 %** | 99.1 % |
| dictation, one voice | **1** | `single-speaker` | 98.6 % | 100 % |

The dictation row is the one worth reading twice: it is the fixture upstream's configuration gets
wrong, and `single-speaker` is the PASS. `tests/measure.py` fails if it ever reports two.

On a real deployment of the listing that carries this fork (one H100 NVL cloud, the scribe on six
CPU cores beside a generation engine holding the card), the same pipeline ran at **16–19 s of wall
clock per minute of audio** for recognition and diarization together, and a SOAP note was generated
from a five-minute consultation in **10–20 s**. The first request after a pod start is much slower —
six to eight minutes for a five-minute recording — because the models load on first use.

Synthesised speech is cleaner than a consulting room — no crosstalk, no overlap, no speakerphone,
no accent the model has not heard. These figures are an upper bound and a wiring check, not a
clinical validation; what a deployment relies on is the floors, not the figures.

## 3. Three places where upstream guessed who was speaking, silently

Each of these produced a confident attribution from no evidence. All three are removed.

1. **`services/llm.py::format_transcript_with_speakers`** fell back, whenever `speaker_roles` was
   empty, to keyword matching: a line containing "my symptoms" made that speaker the **Patient**, a
   line containing "recommend" made them the **Provider**. The guess was then printed into the
   prompt as fact. It is now either a measured attribution or an explicit statement to the model
   that the transcript is unattributed and must not be attributed.
2. **`apps/ui/src/components/TranscriptViewer.tsx::getSpeakerLabel`** fell back to
   `speakerTag === 0 → 'Provider'`, `1 → 'Patient'`. A cluster index carries no role information;
   that is a coin toss displayed as a fact. Unattributed speakers now read "Speaker 1", "Speaker 2"
   or "Unidentified speaker", and a banner says what the diarization found.
3. **`services/asr.py::detect_speaker_roles`** prompted the model with "The speaker that is not the
   patient is most likely the doctor, so be generous." Generosity is the failure mode. The prompt
   now offers the model an explicit "cannot tell" answer, the reply is parsed strictly, and a reply
   naming one speaker as both people — which upstream accepted — is refused.

Upstream also **never called `detect_speaker_roles` on the streaming path at all** (the line is
commented out in its source) while `ENABLE_STREAMING` defaults to true, so its headline role
attribution was off by default and the keyword guess above was the ordinary path. Roles are detected
on both paths here, subject to the same floors.

## 4. Generation: NVIDIA-hosted or a 49B NIM → the deployment's own endpoint

`OPENAI_BASE_URL` and `LLM_MODEL` already made the generation endpoint configurable, which is why
`nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1` (2 × H100 by upstream's README) is a
drop-in replacement rather than a rewrite. What changed:

* `NVIDIA_API_KEY` and `RIVA_FUNCTION_ID` are gone, with the hosted branch that used them. Reaching
  `integrate.api.nvidia.com` with a consultation is the thing a tenant deploys this to avoid.
* the client carries `LLM_API_KEY` and a timeout, because an unauthenticated inference endpoint is
  an unauthenticated inference endpoint even on a cluster-internal Service.

## 5. NeMo Guardrails: removed

`services/guardrails.py` is deleted and `nemoguardrails` / `langchain-nvidia-ai-endpoints` are out
of the dependencies. Upstream gates the whole path on `ENABLE_GUARDRAILS`, which defaults to
**false**, and ships **no rails configuration at all** — the directory `guardrails_config_path`
points at does not exist in the repository — while the engine it wires is `nvidia_ai_endpoints`,
i.e. NVIDIA-hosted inference. A safety switch that cannot be turned on, in front of a claim nobody
can test, is worse than its absence. This fork serves no content-safety model and says so.

## 6. NVIDIA proprietary material in the UI: removed

Not a preference — a licence blocker, and the reason is worth reading before any future upstream
sync. `apps/ui` vendored four NVIDIA tarballs and the repository's Apache-2.0 root licence does not
cover them:

| vendored file | licence |
|---|---|
| `nv-brand-assets-icons-3.8.0.tgz` | `LicenseRef-NvidiaProprietary` — "any use, reproduction, disclosure or distribution … without an express license agreement from NVIDIA … is strictly prohibited" |
| `nv-brand-assets-react-icons-inline-3.8.0.tgz` | the same |
| `kui-react-0.402.1.tgz` | NVIDIA Kaizen UI License; distribution only "as incorporated into a software application", accessed only by that application |
| `kui-foundations-0.403.0.tgz` | the same |

A published container image is redistribution, and the brand assets are NVIDIA trade dress besides.
All four are deleted. Their replacements:

* the five icons (`Health`, `Microphone`, `FolderOpen`, `Soundwaves`) come from `lucide-react` (ISC),
  already a dependency;
* the four `@kui/react` components actually used — `ThemeProvider`, `AppBar`, `Button`, `Text` —
  are rebuilt in `apps/ui/src/components/kit.tsx` with the same props, so `App.tsx` changes by one
  import line;
* the `@kui/foundations` Tailwind plugin, which supplied about forty design tokens
  (`text-brand`, `bg-surface-raised`, `bg-accent-blue-subtle` …) and the `nv-dark` theme, is
  replaced by an explicit palette in `tailwind.config.js`. The token **names** are kept; the
  **values** are ours, a neutral slate-and-teal dark theme.

The UI builds and behaves as before.

## 7. Transcripts and notes survive a restart; a recording does not outlive its retention

Upstream keeps both in module-level dicts (`routers/transcribe.py::_transcripts`,
`routers/notes.py::_notes`). On a cluster the consequence is the wrong way round: an eviction, a
node drain or a reconfigure loses every note a clinician drafted, while the uploaded recording — the
protected health information — persists because it is a file on a volume.

`utils/store.py` adds `JsonStore`, a `MutableMapping` with one atomically-written JSON document per
entry, so every existing call site is unchanged. Beside it:

* `DELETE /api/transcribe/{id}` deletes the transcript **and its recording** — "delete" has to reach
  the audio or it means nothing. Notes are deleted individually; a signed note is a clinical record.
* `DELETE /api/transcribe/{id}/audio` deletes only the recording, keeping the text and timecodes.
* `AUDIO_RETENTION_HOURS` (default 24) is swept by a background task, including once at startup so
  a pod that was down past an expiry does not resurrect a kept file. Negative keeps indefinitely.

## 8. A transcript that already exists elsewhere

`POST /api/transcribe/import` creates a transcript from rows rather than from a recording. It exists
because of what the marketplace gatekeeper is: a data owner approves a deployment against its
published evidence, serves it a ClickHouse table, and withdraws it the moment that evidence changes.
A consultation already transcribed — by another system, by a human typist — is therefore something a
tenant can bring into the enclave under the owner's control and generate a note from, with no
recording in this deployment at all.

Its speaker labels are the owner's, and the code says so rather than claiming them: the transcript's
status is `provided`, it records the source it came from, and the note generation is told
"this transcript arrived already speaker-separated from <source>". A speaker string this deployment
recognises as a role (`clinician`, `doctor`, `patient`, …) becomes that role; one it does not stays a
numbered speaker rather than being guessed at. Rows with no speaker column produce an unattributed
transcript, reported as such.

## 9. Smaller corrections

* **`main.py`: CORS.** Upstream sends `allow_origins=["*"]` with `allow_credentials=True`. The
  console in front of this API carries a session cookie, so that is a cross-site request away from
  somebody else's page reading a consultation. Empty by default; the UI is same-origin.
* **`main.py`: the uploads mount followed `./uploads`** rather than `UPLOAD_DIR`, so the two
  diverged as soon as a deployment set the variable.
* **`main.py`: the built UI is served from this process**, replacing the second nginx container.
* **`main.py`: uvicorn's `__main__` binds loopback**, not `0.0.0.0`.
* **`routers/transcribe.py`: an upload is accepted on what it is.** Upstream rejects anything whose
  `Content-Type` does not begin `audio/`, and browsers send `application/octet-stream` for plenty of
  ordinary recordings — an `.m4a` from a dictaphone among them. The extension is accepted too and
  the decoder is the real gate.
* **`routers/health.py`** reports the dependencies this deployment has — the checkpoints on disk and
  whether a generation endpoint is configured — instead of whether an NVIDIA API key was set. It
  deliberately does not call the model: a health check that does turns an engine still loading its
  weights into a restarted pod.
* **One transcription at a time.** CTranslate2 and onnxruntime are both threaded across the whole
  CPU allocation; a second concurrent job makes both slower and makes a published throughput figure
  meaningless.
* **A number is a number however it is written.** `MAX_FILE_SIZE=5.24288e+08` is what a Helm
  chart emits for `524288000`: a YAML emitter prints a large float64 in scientific notation,
  `int()` raises on it, the settings model fails to validate, and the process dies on import.
  On a confidential cluster space that is a pod which never becomes ready and no pod-level
  visibility to read the reason out of — it cost this port fifty minutes of a billed GPU before
  a local cluster showed the traceback. The integer settings now parse through `float`.
* **Streaming partials carry no speaker.** Diarization is a whole-recording problem — the speaker of
  the first sentence is decided by how it compares with the last — so the speakers arrive once, with
  the score, in the `complete` event. Upstream attached a speaker to every interim result from a
  diarizer that had heard only the audio so far.

## 10. Packaging

One image (`infra/docker/scribe.Dockerfile`) instead of three: the API, the built UI and the model
weights. `models/manifest.json` pins every model file by sha256 and records its licence;
`models/fetch_models.py` fetches them **at build time** and fails the build on a mismatch. So a
deployment reaches no model host — not huggingface.co, not NGC — and the image digest a listing pins
covers the weights as well as the code.

Everything in the image runs on the CPU: no CUDA base, no `NVIDIA_REQUIRE_CUDA`, and none of the
driver-floor trouble a recent CUDA base brings. The card in a deployment belongs to the generation
engine, which is a separate component.

`apps/api/uv.lock` is regenerated (93 packages, down from 468). The repository-root
`pyproject.toml` / `uv.lock` are upstream's notebook environment and are untouched; no image is
built from them.

## 11. What is NOT in this fork

* `ambient-patient` and `ace-controller-voice-interface` — the patient agent, its Pipecat/WebRTC
  stack and its TTS. A separate product.
* The `riva-nim` Compose profile's two NIM services, and `NGC_API_KEY`.
* `Nemotron-Labs-Audex-{2B,30B-A3B}` — `nvidia-oneway-noncommercial-license`. Recorded because the
  name invites the mistake.
* NemoGuard content-safety and topic-control weights (`license: other`).
