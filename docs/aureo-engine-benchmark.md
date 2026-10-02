# AUREO VOICE ENGINE BENCHMARK — Phase 3

For persistent Chatterbox GPU loading, reference caching, WAV exports and
Runpod deployment, see [Phase 4](aureo-runpod.md).

The native benchmark runs real SDK loading and synthesis once checkpoints and
dependencies are provisioned locally. This phase builds and tests that
infrastructure; it does not download models, publish audio, or assign engines
to FAST/PRO/ULTRA without measurements. Existing OmniVoice, routes, SQLite,
user-data paths, UI, credits and administration remain unchanged.

## Isolation and measurements

`python -m aureo experiment` starts a fresh subprocess for **each engine/style/
repeat**, sequentially. Every engine can use its own Python executable and SDK
environment. This matters because the inspected Chatterbox source declares
Transformers 5.2.0 and PyTorch 2.6.0 for Python 3.11, whereas Qwen3-TTS declares
Transformers 4.57.3. Do not install those SDKs together in VoiceStudio's main
environment or change its frozen dependency lockfiles.

| Component | Files | Role |
| --- | --- | --- |
| Native bridges | `aureo/adapters/chatterbox_multilingual.py`, `qwen3_base.py`, `openvoice_melo.py` | Separate multilingual, 1.7B Base and V2+Melo adapters |
| Local provisioning contract | `aureo/runtimes/` | Strict local configuration, asset checks and lazy native SDK loading |
| Presets | `aureo/presets.py` | Style intent, explicit capability negotiation and private request serialization |
| Worker | `aureo/worker.py` | Offline native loading, existing AUREO benchmark and worker metadata |
| Experiment | `aureo/experiment.py` | One reference snapshot, isolated trials, timeouts and validated JSON replies |
| Outputs | `aureo/report.py` | Versioned JSON and readable Markdown with errors and capability notes |

The parent sends text/reference/configuration through stdin, never a shell
command. Only metadata returns; waveform samples stay in the worker and are
discarded. The worker exits after each trial, releasing its model and CUDA
context. A timed-out worker is killed and reaped; its error does not stop later
trials. A worker fingerprint mismatch or malformed measurement becomes an
error, never a successful benchmark row. No engine is loaded in the parent.

One reference snapshot serves **all engines, styles and repeats**, even if the
original file changes during the experiment. Text, audio and transcript SHA-256
hashes identify inputs; raw text, transcripts and input/model paths are absent
from saved reports. Requested/effective controls, language, mode, package
versions, Python/PyTorch/CUDA versions and detected GPU names provide context.

The worker forces Hugging Face/Transformers offline mode and blocks Python
IPv4/IPv6 socket connections throughout loading and generation. Unix driver
sockets remain available. This guard is confined to the worker, not applied to
the existing application. Missing tokenizer, BERT, dictionary or watermark
assets fail instead of downloading silently. It is not a sandbox for arbitrary
untrusted native code; use the documented SDKs and pre-provision all their assets.

| Measurement | Meaning |
| --- | --- |
| Load | SDK import + local model construction/loading + device synchronization |
| First audio | First actual CPU PCM chunk, including load; buffered SDKs return first audio at completion |
| Generation | Consumption of the native synthesis output, including partial failures |
| Total | Engine run inside the worker, including cleanup and memory sampler stop |
| Process wall | Interpreter startup, private IPC, the engine run and worker exit |
| Audio duration | Number of PCM samples / sample rate |
| RTF | Generation seconds / audio duration; smaller than 1 is faster than real time |
| End-to-end RTF | Worker engine total / audio duration |
| RAM | Sampled **worker** RSS, when psutil is installed |
| VRAM | Sampled PyTorch-allocated CUDA bytes; unavailable measurements are null |

CUDA/MPS are synchronized after loading. Native buffered outputs are copied to
CPU before their first-audio time is observed. Peak memory sampling can miss
brief peaks and does not include full physical GPU usage, reserved allocator
memory or MPS allocations. Each engine is cold and each process is fresh;
OS filesystem/driver/kernel caches are not flushed. Reference preparation is
outside engine timings. Interpreter/IPC overhead is separately visible.

## Native APIs and controls

| Engine ID | Local API | Required assets |
| --- | --- | --- |
| `chatterbox-multilingual` | `ChatterboxMultilingualTTS.from_local` | `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json`; any SDK conditioning/tokenizer/watermark assets |
| `qwen3-tts-1.7b-base` | `Qwen3TTSModel.from_pretrained(local_directory, local_files_only=True)` | Full `Qwen/Qwen3-TTS-12Hz-1.7B-Base` snapshot, including processor/speech tokenizer assets |
| `openvoice-v2` | `ToneColorConverter` + `MeloTTS.TTS` with explicit local config/checkpoint | V2 converter, Melo language model, matching source speaker embedding, local WavMark checkpoint and text/BERT resources |

Qwen configurations and loaded runtimes must both declare **Base / 1.7B**.
CustomVoice, VoiceDesign and 0.6B are rejected. Attention defaults to SDPA, so
FlashAttention installation is not required. `dtype=auto` chooses float32 on
CPU/MPS and bf16 on supporting CUDA devices, otherwise fp16. OpenVoice requires
V2 configuration and a matching Melo speaker ID; its supported output language
is the configured source model's language. The converter's native watermark is
retained and loaded from the explicit local WavMark path.

| Control | Chatterbox Multilingual | Qwen3-TTS 1.7B Base | OpenVoice V2 + Melo |
| --- | --- | --- | --- |
| Reference / seed | Required reference; serialized RNG scope | Required reference; transcript-based ICL when available | Required reference, explicit source embedding; seed covers source/converter |
| Variation `[0,1]` | Temperature `0.1 + 0.9*v` | Temperature `0.1 + 0.9*v` | Source noise scale `0.1 + 0.8*v`, conversion tau `0.1 + 0.8*v` |
| Energy `[0,1]` | Native exaggeration | Unsupported | Unsupported |
| Free-form expression | Unsupported by this API | Unsupported by the Base clone API | Unsupported by these APIs |
| Fidelity `[0,1]` | CFG strength `0.1 + 0.8*f`: a guidance proxy | Unsupported as a continuous native control; ICL still uses the transcript | Speaker embedding blend: `(1-f)*source + f*reference` |
| Speed `[0.25,4]` | Unsupported | Unsupported | Native Melo speech-rate multiplier |
| Real audio streaming | Unsupported | Unsupported; native simulated text-input streaming is not chunked audio | Unsupported |

Fidelity controls **do not measure speaker similarity**. Speed is speech rate,
not a promise of faster inference; it changes audio duration and therefore RTF.
Unsupported explicit CLI controls return errors before loading, rather than
being silently ignored. Existing English Chatterbox and the Phase 2 discovery/
benchmark interface remain available with their original engine IDs. AUREO
profiles now round-trip optional fidelity/speed while reading existing profiles.

## NATURAL, ENERGÉTICO and FIDELIDAD

| Style | Variation | Energy | Expression intent | Fidelity | Speech speed |
| --- | --- | --- | --- | --- | --- |
| NATURAL | 0.35 | 0.45 | natural | 0.70 | 1.00 |
| ENERGÉTICO | 0.65 | 0.85 | energetic | 0.55 | 1.10 |
| FIDELIDAD | 0.15 | 0.35 | neutral | 1.00 | 1.00 |

Default preset adaptation applies only supported controls and records every
omission in JSON and Markdown. The style name is **intent**, not a guarantee
that different engines implement equivalent emotions. No initial bridge
supports free-form expression. `--strict-presets` instead rejects any unsupported
preset control. Explicit `--energy`, `--expression`, `--fidelity`, `--speed` or
`--variation` always remain strict and override the preset across all styles.
`ENERGETICO` is accepted as a CLI alias for `ENERGÉTICO`.

Styles are independent from FAST/PRO/ULTRA modes. Reports summarize successful
trials with medians and show all failed trials in JSON. They leave final mode
bindings unassigned: latency alone cannot establish cloning quality. No MOS,
speaker similarity or subjective quality scores are invented. This phase exports
no synthetic audio; future listening/export integration must preserve the
application's `mark_synthetic` policy.

## Commands and configuration

```bash
uv run --no-sync python -m aureo catalog
uv run --no-sync python -m aureo experiment \
  --text "Esta es una prueba de voz para AUREO." \
  --reference /workspace/references/speaker.wav \
  --reference-text "Exact transcript of the reference recording" \
  --language es --seed 42 \
  --runtime-config /workspace/aureo-runtimes.json \
  --engines chatterbox-multilingual \
  --styles NATURAL ENERGETICO FIDELIDAD --repeats 3 \
  --output /workspace/aureo-results/first-chatterbox.json
```

See [the one-engine configuration template](aureo-runtime.example.json). It
intentionally configures only the first engine. Without `--engines`, only engines
present in the runtime configuration are tested. With no configuration, the
catalog's three adapters remain unconfigured and return errors without loading
models. These errors can still be saved as infrastructure diagnostic reports.

Relative paths in the runtime JSON resolve against that file's directory. Each
`python` field is an executable path, not a command or shell fragment. Workers
import AUREO from this checkout/installed package and native SDKs from that
interpreter's environment. No main-app SDK/dependency upgrade is needed.

Qwen runtime example (add only after its separate environment/assets are ready):

```json
{"python": "/workspace/aureo-runtime-envs/qwen/bin/python", "device": "cuda:0",
 "checkpoint_dir": "/workspace/aureo-models/qwen3-1.7b-base", "dtype": "auto"}
```

OpenVoice runtime example (all component checkpoints must be present):

```json
{"python": "/workspace/aureo-runtime-envs/openvoice/bin/python", "device": "cuda:0",
 "language": "es", "speaker": "ES",
 "converter_config": "/models/openvoice-v2/converter/config.json",
 "converter_checkpoint": "/models/openvoice-v2/converter/checkpoint.pth",
 "melo_config": "/models/melo-es/config.json",
 "melo_checkpoint": "/models/melo-es/checkpoint.pth",
 "source_embedding": "/models/openvoice-v2/base_speakers/ses/es.pth",
 "watermark_checkpoint": "/models/wavmark/model.pkl"}
```

Results default to an adjacent `.md` comparison; `--report` selects another
Markdown path. Both files are staged completely before publication, with
atomic replacement per file, not a two-file transaction. The shared run ID
identifies a pair after a filesystem failure. Existing outputs require
`--overwrite`. Inputs, runtime configuration, executables and model directories
are protected from output overwrite. Preflight runs before expensive trials.

Exit 0 means all trials succeeded, 1 means trial errors recorded in the result,
and 2 means invalid configuration or an output write failure. `--timeout`
defaults to 300 seconds per worker. `assets_missing`, `sdk_missing`,
`gpu_unavailable`, `wrong_model`, `offline_asset_missing`, `unsupported_control`,
`worker_timeout` and `worker_error` distinguish prerequisites and failures.

## Install first: Chatterbox Multilingual

Start with **Chatterbox Multilingual**, then Qwen3-TTS 1.7B Base, then OpenVoice
V2+Melo. Chatterbox directly supports Spanish and energy/exaggeration, has one
native model entry point, and avoids OpenVoice's multi-component setup. Qwen
adds a larger Base model and tokenizer assets; OpenVoice adds a base synthesizer,
matching embedding, BERT resources and watermark checkpoint.

Use an NVIDIA L4/A10 with **24 GB VRAM** as a conservative first test target,
not a claimed minimum requirement or measured memory usage. The current
development Cloud has **no GPU** and no installed SDKs; GPU inference has not
been run. Use a clean 6–10 second mono WAV, a matching transcript and one short
Spanish text for the first trial. Begin with one NATURAL trial, then all three
styles and three repeats after that succeeds.

The following preparation is for a separately provisioned GPU Cloud, **not
executed in this phase**:

```bash
uv venv /workspace/aureo-runtime-envs/chatterbox --python 3.11
uv pip install --python /workspace/aureo-runtime-envs/chatterbox/bin/python \
  "chatterbox-tts==0.1.7" "psutil>=7.2.2"
```

Verify `torch.cuda.is_available()` in that interpreter and record the installed
SDK/PyTorch/CUDA versions. Provision **only this engine's** model assets through
the official [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
repository, choosing and recording a fixed revision. Include the multilingual
V2 T3 file, voice encoder, S3 generator, tokenizer and any additional assets
required by the chosen SDK. Local factory checks and the worker's offline guard
identify anything missing. Do not run all engines until each separate environment
has independently passed its first NATURAL trial.

Inspected native API references:
[Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/mtl_tts.py),
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py),
[OpenVoice](https://github.com/myshell-ai/OpenVoice/blob/main/openvoice/api.py),
[MeloTTS](https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py),
[WavMark](https://github.com/wavmark/wavmark/blob/main/src/wavmark/__init__.py).

Infrastructure tests use deliberate SDK fakes and real subprocesses, validating
native argument mappings, identity checks, offline enforcement, hashes, controls,
timeouts/reaping, profile compatibility, JSON persistence and reports. The CPU
worker test validates actual process RSS and pipeline execution, not speech
quality or real model performance. No model download or physical GPU inference
is claimed by passing infrastructure tests.

Phase 3 validation in the current CPU Cloud: `tests/aureo` **133 passed**;
the complete `tests/` suite **9,609 passed, 31 skipped, 8 xfailed, 1 xpassed**;
`backend/tests` **472 passed**. Wheel and source distribution builds passed,
including all 31 AUREO Python modules and the unchanged OmniVoice package.
Isolated wheel discovery required no native SDK or PyTorch import. A CLI
diagnostic saved nine expected `runtime_not_configured` trials (three engines
times three styles) in matching JSON/Markdown, with one shared reference hash.
These diagnostic errors validate reporting; they are not real engine timings.
