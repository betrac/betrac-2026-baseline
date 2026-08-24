# Changelog

All notable changes to the BeTraC 2026 Baseline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Audio was silently truncated before reaching the model.** Both Omni processor
  families frame audio with a `WhisperFeatureExtractor` whose `__call__` defaults
  to `truncation=True` with `max_length = chunk_length * sampling_rate`.
  `chunk_length` is read from each checkpoint's `preprocessor_config.json`, so the
  baseline was only ever hearing the first:
  - **300 s** for `Qwen2.5-Omni-3B` / `-7B` (`chunk_length: 300`)
  - **30 s** for `Qwen3-Omni-30B-A3B-Instruct` / `-Thinking` (`chunk_length` absent →
    Whisper's default of 30)

  On the BeTraC validation split (400 dialogs, mean 530 s, max 1474 s) that meant
  **55.9 %** of the audio reached the Qwen2.5 models and only **5.7 %** reached the
  Qwen3 models. `run_omni.py` now passes `truncation=False` to the processor.
  Verify on any checkpoint with `python scripts/check_audio_truncation.py`.

### Fixed (also)
- **Qwen2.5 output was silently capped at 1024 tokens.** The model's `generate()`
  wrapper declares its own `thinker_max_new_tokens=1024` and discards a plain
  `max_new_tokens=`; the repo's 4096 never arrived. Now passed as
  `thinker_max_new_tokens`. Measured metric impact of this fix alone: none
  (see experiments/RESULTS.md) — the score gains are the audio fix.
- **Thinking-budget overflow emitted raw chain-of-thought as the SOAP note.**
  Reasoning grows with audio heard; 8192 tokens overflowed mid-`<think>` on
  full recordings with `success: true`. Budget raised to 16384 and an unclosed
  `<think>` now fails the sample instead of poisoning the results.
- **`write_audio_tempfile` could silently truncate audio on a full filesystem**
  (bare `os.write` short-writes instead of raising). Now buffered and
  size-verified.
- **Completed generations of near-empty noise are no longer `success: true`**
  (observed from a misconfigured sharded run); summaries under 25 words are
  recorded as failures.

### Changed
- `--resume` no longer treats records from before the audio fix (no
  `audio_sec` field) or failed records as complete — they are re-processed.
  Previously, pulling this fix and re-running with `--resume` was a silent
  no-op that left truncated notes in place.
- OOM handling: a CUDA out-of-memory on a sample now retries in a fresh
  subprocess with progressively shorter audio (60 %, then 35 %, floored at
  300 s) instead of scoring zero.

### Added
- `--num-gpus N` (env `NUM_GPUS`): shard the model across N GPUs with explicit
  per-device memory caps. **Experimental** — 2×40 GB for the 30B is marginal
  (observed hangs and one corrupted output); a single ≥80 GB card is the
  reliable configuration. See README "Audio length".
- `scripts/paired_bootstrap.py` — the significance test behind
  experiments/RESULTS.md.
- `--max-audio-seconds` (env `MAX_AUDIO_SECONDS` in every runner script) bounds the
  audio length explicitly. The default is derived from the thinker context window
  rather than hidden in a preprocessor config: ~1126 s for Qwen2.5-Omni (32 k ctx,
  25 audio tokens/s) and ~4372 s for Qwen3-Omni (64 k ctx, 13 audio tokens/s).
  Pass a negative value for no cap. Audio is cut at load time via
  `qwen_omni_utils`' `audio_end`, and anything dropped is logged as a warning.
- `context_length` and `audio_tokens_per_second` in `MODEL_CONFIGS`, used to
  compute the cap above.
- `scripts/check_audio_truncation.py` — reproduces the bug and verifies the fix
  against the real processors. Downloads processor files only, no model weights.
- Output records now carry `audio_sec` and `audio_used_sec`, so a truncation
  regression is visible in the results themselves rather than only in logs.

## [0.1.0] - 2026-04-04

### Added
- End-to-end audio-to-SOAP-note baseline using Qwen Omni models
- Support for four verified models:
  - Qwen2.5-Omni-3B (Lightweight track)
  - Qwen2.5-Omni-7B (Heavyweight track)
  - Qwen3-Omni-30B-A3B-Instruct (Heavyweight track)
  - Qwen3-Omni-30B-A3B-Thinking (Heavyweight track, chain-of-thought)
- Model config registry (`MODEL_CONFIGS`) with per-model defaults for device and generation params
  - Auto-detection of model family (Qwen3-Omni-MoE vs Qwen2.5-Omni)
  - Qwen2.5-Omni-7B auto-routes to CPU on MPS (dense 7B exceeds MPS memory)
  - CUDA users unaffected — `device_map="auto"` handles GPU placement
- MPS high watermark safety limit (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, configurable)
- HuggingFace dataset loading as primary data source (`--dataset`/`--split` CLI)
  - CSV manifest retained as fallback for custom data (`--manifest`)
- Subprocess isolation for GPU memory management
- Makefile targets: `run`, `run-manifest`, `test`, `test-5`, `test-hf`
- Scripts: `create_manifest.py`, `validate_manifest.py`, `merge_results.py`, `show_results.py`
- Test suite: `test_run_omni.py`, `test_data_loading.py` (unit + integration)
- Pre-commit hooks (ruff linting/formatting, file quality checks)
- Comprehensive documentation: README, ARCHITECTURE, CONTRIBUTING, TROUBLESHOOTING, CHALLENGE
- Apache 2.0 license with attribution to Alibaba Cloud (Qwen3-Omni cookbooks)

### Verified
- Qwen2.5-Omni-3B: 5/5 validation samples on MPS (~70–120s/sample)
- Qwen2.5-Omni-7B: 1/1 validation sample on CPU (~160s/sample, auto-routed from MPS)
- Qwen3-Omni-30B-A3B-Instruct: 5/5 validation samples on MPS (~260–360s/sample)
- Qwen3-Omni-30B-A3B-Thinking: 5/5 validation samples on MPS (~450–570s/sample)
- All models tested with torch 2.11.0 + transformers 4.57.6 on Apple Silicon

[Unreleased]: https://github.com/betrac/betrac-2026-baseline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/betrac/betrac-2026-baseline/releases/tag/v0.1.0
