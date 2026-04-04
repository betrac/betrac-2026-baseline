# Changelog

All notable changes to the BeTraC 2026 Baseline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
