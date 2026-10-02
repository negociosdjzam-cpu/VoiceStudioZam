# AUREO VOICE ENGINE LAB — Phase 2

For native Multilingual/1.7B Base/V2+Melo loading, isolated trials, fidelity/speed
controls and saved comparisons, see [the Phase 3 benchmark guide](aureo-engine-benchmark.md).

This is a parallel, independent Python foundation for future AUREO Core work.
It is usable through Python and `python -m aureo`. Existing OmniVoice engines,
`TTSBackend`, HTTP routes, SQLite profiles, user-data paths, UI, credits,
administration and update mechanisms continue to use their existing code.
There is no new application route or engine selector in the UI in this phase.

## Boundaries

| Layer | Files | Responsibility |
| --- | --- | --- |
| Contracts | `aureo/types.py`, `engine.py` | Immutable requests, mono PCM chunks, capabilities, load/generate/stream/unload and per-instance serialization |
| Selection | `registry.py`, `profiles.py`, `lab.py` | Explicit engine instances, JSON voice profiles and FAST/PRO/ULTRA dispatch |
| Runtime bridges | `adapters/` | Translate AUREO requests into native SDK calls on explicitly supplied local runtimes |
| Measurements | `benchmark.py`, `metrics.py` | Sequential comparisons, timings, sampled memory and per-engine errors |
| Developer interface | `cli.py`, `__main__.py` | JSON discovery and benchmark reports; no audio exports |

The contracts use the standard library. They import neither the legacy project
nor vendor SDKs, NumPy or PyTorch. Both wheels and source distributions include
`aureo` alongside the existing `omnivoice` package. Vendor dependencies and
checkpoints are deliberately not installed or downloaded by this layer.

`AureoEngine` implementations declare their capabilities and implement `_load`,
`_generate`, `_unload` and optionally `_stream`. An actual streaming plugin yields
`AudioChunk` values before completion. Its iterator must be closed on cancellation.
Chunks must contain finite mono PCM and keep the same sample rate within a stream.
The engine's session lock covers rendering and cleanup, so another lab operation
cannot unload the same instance halfway through generation.

## Initial native bridges

| Adapter / runtime | Reference | Seed / variation | Energy / expression | Real streaming |
| --- | --- | --- | --- | --- |
| `ChatterboxAdapter` / English `ChatterboxTTS` | Optional local audio | Yes / temperature | Energy maps to exaggeration; expression unsupported | Unsupported |
| `Qwen3TTSAdapter` / **Base** `Qwen3TTSModel` | Required; transcript optional | Yes / temperature | Unsupported in this clone API | Unsupported |
| `OpenVoiceV2Adapter` / `OpenVoiceRuntime` | Required | Yes / converter tau | Unsupported | Unsupported |

Chatterbox's reference conditioning is restored after each request, including
SDK failures, to prevent the previous profile's voice from becoming the next
profile's default. This first bridge supports the English base model only.
Qwen maps language codes `zh/en/ja/ko/de/fr/ru/pt/es/it` to its native language
names; absent language uses Auto. A transcript enables its transcript-based
clone mode; otherwise it uses the speaker-vector mode. Its `non_streaming_mode`
flag controls the native batch call and does **not** provide real audio streaming.
OpenVoice V2 requires both a caller-configured base TTS and the matching source
speaker embedding, in addition to its tone-color converter. No implicit
OmniVoice or MeloTTS model is created. Its intermediate source WAV is temporary
and removed on success or failure; the converter's own watermark default remains.

`variation` and `energy` are finite values in `[0, 1]`, and seed is an unsigned
32-bit integer. Variation maps to temperature `0.1 + 0.9*v` for Chatterbox/Qwen
and conversion tau `0.1 + 0.8*v` for OpenVoice. Energy maps directly to
Chatterbox exaggeration, an approximation specific to that model. Unsupported
controls produce explicit errors before loading. Expression is present in the
common contract for future plugins; none of these initial clone bridges claims it.

FAST/PRO/ULTRA can select different engines through a profile. Within Chatterbox
they choose CFG weights `0.3/0.5/0.7` and default temperatures `0.7/0.8/0.8`;
within Qwen they choose top-k `20/50/80` and top-p `0.8/0.95/0.97`. OpenVoice's
base synthesizer receives the mode and owns its policy. These are initial
sampling policies, not measured speed or quality guarantees. A supplied
variation overrides the relevant temperature policy.

SDK references used for these bridges:
[Chatterbox](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/tts.py),
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py),
[OpenVoice](https://github.com/myshell-ai/OpenVoice/blob/main/openvoice/api.py).
Native execution against real checkpoints remains to be validated when models
are explicitly approved and available locally.

## Profiles and host audio policy

A profile is an explicit, versioned JSON file, separate from existing SQLite:

```json
{
  "version": 1,
  "profile_id": "speaker-1",
  "engine_id": "chatterbox",
  "mode": "PRO",
  "mode_engines": {
    "FAST": "chatterbox",
    "PRO": "qwen3-tts",
    "ULTRA": "openvoice-v2"
  },
  "reference": {"path": "reference.wav", "transcript": "Reference transcript"},
  "seed": 42,
  "variation": 0.5
}
```

Modes without a binding use `engine_id`. Manual relative reference paths resolve
against the profile file's directory. Programmatic `profile.write(path)` saves
the caller's resolved absolute reference and atomically replaces that one file.
There is no migration, default user-data location or shared profile database.
Controls are shared across modes: choose controls supported by every selected
engine, or create separate profiles; no adapter silently discards a control.

`AureoEngineLab(registry, audio_processor=host_policy)` dispatches a profile to
its selected engine and runs every returned chunk through the supplied host
policy. The host must provide an explicit processor. A future VoiceStudio
integration must route exported synthesis through
`services.watermark.mark_synthetic`, preserving the application's provenance
policy. Streaming requires an appropriate buffering/watermark policy from the
host before export. The independent core does not import that legacy service.
Direct adapter PCM is an internal value, not an exported take. This phase adds
no audio export route; benchmarks discard PCM after measuring it.

## Offline discovery and local runtime configuration

```bash
uv run --no-sync python -m aureo list
uv run --no-sync python -m aureo benchmark --text "Hello" \
  --reference /absolute/path/reference.wav --seed 42 --variation 0.5 \
  --engines chatterbox qwen3-tts openvoice-v2
```

Discovery reports unloaded, unconfigured adapters. Without factories, the
benchmark returns `runtime_not_configured` for each eligible adapter, with
null time-to-first-audio and zero generated duration. This is expected; no
weights, SDKs or estimated generation times are fabricated. CLI exit codes:
0 for success, 1 for engine errors, 2 for invalid configuration. `--streaming`
requires an engine that advertises actual streaming; these first bridges reject it.

Provide local runtime factories explicitly when SDKs and all necessary assets
are already present. For example, in a caller-owned Python module:

```python
from aureo.adapters import create_default_registry

def build_registry():
    def chatterbox_runtime():
        # Import here only after explicitly provisioning the SDK and local
        # assets. Set HF_HUB_OFFLINE=1 in the process environment beforehand.
        from chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_local("/models/chatterbox", device="cpu")

    return create_default_registry({"chatterbox": chatterbox_runtime})
```

Then use `python -m aureo --registry-factory local_module:build_registry benchmark
--text "Hello" --engines chatterbox`. The hook is an explicit developer code
import, not discovery. For Qwen, return a locally loaded **Base** model from
`Qwen3TTSModel.from_pretrained(local_directory, local_files_only=True)` with all
tokenizer/codec assets already installed. For OpenVoice, return
`OpenVoiceRuntime(converter, synthesize_source)`, where `synthesize_source(request)`
returns `OpenVoiceSource(AudioChunk(...), matching_source_embedding)` from the
caller's already configured local TTS. Factories own offline provisioning:
the lab does not make arbitrary caller code incapable of accessing a network.

Adapters drop their runtime reference on unload; optional `runtime_disposer`
callbacks can release resources owned by the runtime. A factory that returns
an externally retained object does not make a genuinely cold load. Do not
reuse externally retained runtimes when measuring loading cost.

Seed scopes serialize these adapters and restore Python, already-imported
NumPy and PyTorch CPU/CUDA RNGs, plus MPS RNG when an explicit MPS device is
used. They do not import frameworks or initialize CUDA. Repeatability still
depends on SDK kernels and hardware; identical seeds across different models
do not imply identical voices. Unrelated legacy SDKs do not share this RNG
lock: run the lab CLI separately, or use isolated workers before combining
concurrent application synthesis with seeded lab execution.

## Benchmark definitions

Each comparison uses one immutable request, identical text and controls, and
one temporary snapshot of reference bytes for every engine. Results include
text/reference SHA-256 hashes, not raw text, transcripts, paths or waveform data.
Engines run sequentially. Errors record the operation stage and stable code;
unexpected SDK errors expose the exception class, not potentially private messages.

| Field | Measurement |
| --- | --- |
| `load_seconds` | Engine load call; also measured when load fails |
| `time_to_first_audio_seconds` | Entry into the engine run until first PCM chunk, including reset/load |
| `generation_time_to_first_audio_seconds` | Start of synthesis until first PCM chunk |
| `generation_seconds` | Synthesis/iterator consumption, including time spent before a generation failure |
| `total_seconds` | Complete engine run including validation, reset/load, synthesis, cleanup and memory sampler stop |
| `generated_duration_seconds` | PCM sample count divided by sample rate, summed across chunks |
| `real_time_factor` | Generation seconds / audio duration; smaller than 1 is faster than real time |
| `end_to_end_real_time_factor` | Total seconds / audio duration |

Reference snapshot preparation and hashing are outside individual engine times.
Buffered APIs have first-audio time equal to the first completed waveform;
no fake intermediate chunk is created. Empty or changing-rate streams fail.
Partial audio duration and first-audio time remain visible after an error,
but real-time factors are null and `partial_audio` is true. One engine's error
does not prevent testing the next. Cold comparisons unload before and after
each engine, including failed loads. Python `cold=False` retains healthy
runtimes for warm comparisons; callers must unload them afterwards.

RAM is sampled current-process RSS when `psutil` is available. VRAM is sampled
PyTorch-allocated CUDA bytes across already initialized devices; no global
peak counters are reset. These are sampled process peaks, not exact per-engine
allocation or full physical GPU usage. Sampling can miss brief peaks and adds
some overhead. External workers, MPS memory and other allocators are not covered.
Unavailable values are null; observability failures never fail synthesis.
The default sample interval is 20 ms and is included in the report.

## Validation scope

`tests/aureo/` exercises contracts, independent discovery, profile dispatch and
atomic persistence, concurrent lifecycle protection, cancellation, deterministic
timing, reference snapshots, per-engine failures, native SDK argument mappings,
RNG restoration, optional memory reporting and CLI behavior. Test runtimes are
deliberate fakes: these tests validate the integration contracts, not speech
quality, real checkpoint performance or physical GPU measurements. No AI model
is downloaded by the tests or this implementation.

Cloud verification for this phase (Linux, CPU):

| Check | Result |
| --- | --- |
| Main Python suite, including 82 new AUREO cases | 9,558 passed; 31 skipped, 8 xfailed, 1 existing xpassed |
| Isolated backend suite | 472 passed |
| Electron suite | 1,254 passed |
| Shared frontend suite | 2,946 passed |
| Wheel + source distribution | Built; all 17 AUREO modules present; isolated wheel import passed |
| Existing runtime source layout without AUREO | Wheel build passed |
| `uv sync --frozen --all-extras` | Passed without lockfile changes |

Electron's existing runtime fixtures exceeded their default five-second timeout
in this Cloud filesystem. The passing runs used `--maxWorkers=2
--testTimeout=15000` for both Vitest configurations, with all assertions retained
and no Electron source changes. Locale checks also passed. Actual Python,
NumPy and PyTorch CPU seed reproduction/restoration and process RSS were
verified separately without initializing CUDA. Real SDK checkpoints, physical
GPU inference, and execution on macOS/Windows are outside this Cloud validation.
