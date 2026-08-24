#!/usr/bin/env python3
"""Standalone omni step: process audio files directly to summaries using Qwen3-Omni.

Unlike the cascade pipeline (ASR -> LLM), this uses a single omni model that
processes audio end-to-end without a separate transcription step.

Primary usage (HuggingFace dataset):
    python run_omni.py --output results/omni_output.jsonl --split validation
    python run_omni.py --output results/omni_output.jsonl --split validation --limit 5

Fallback usage (CSV manifest for custom data):
    python run_omni.py --manifest ../../data/manifests/example.csv --output ../../results/omni_output.jsonl
"""
# Copyright 2026 BeTraC contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a derivative work based on example cookbooks provided by
# Alibaba Cloud for the Qwen3-Omni model, originally licensed under the
# Apache License, Version 2.0 (Copyright 2025 Alibaba Cloud).

import argparse
import ast
import csv
import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import psutil
import torch
from loguru import logger

# Environment setup for vLLM (must be set before import)
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

DEFAULT_PROMPT = """\
Role: You are a professional medical scribe and clinical documentation specialist.
Task: Analyze the provided audio of a doctor-patient meeting and generate a comprehensive summary.
Format: Use the SOAP note structure (Subjective, Objective, Assessment, Plan).
Details to Include:
Chief Complaint: The primary reason for the visit.
Key Findings: Symptoms discussed and any visual observations from the video (if applicable).
Diagnosis/Assessment: The doctor's professional opinion or tentative findings.
Treatment Plan: Medications, lifestyle changes, or procedures.
Action Items: Specific follow-up tasks, tests to schedule, or future appointments.
Constraints: Maintain a professional and objective tone. Use bullet points for readability. Ensure all drug names are spelled correctly.
"""


# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

# Audio context budget
# ---------------------
# The audio tower emits a fixed number of tokens per second of audio, so the
# longest audio that fits is bounded by the thinker LLM context window:
#
#     max_audio_seconds = (context_length - max_new_tokens - PROMPT_TOKEN_RESERVE)
#                         / audio_tokens_per_second
#
# `audio_tokens_per_second` is measured, not guessed (see
# scripts/check_audio_truncation.py, which recomputes it for any model):
#   - Qwen2.5-Omni  : 25.0 tok/s  (40 ms per audio token)
#   - Qwen3-Omni-MoE: 13.0 tok/s  (the MoE audio tower downsamples 2x)
#
# NOTE: this is a *context* limit, not the feature-extractor limit that caused
# the truncation bug. See `run_model` for the `truncation=False` fix.
PROMPT_TOKEN_RESERVE = 512

MODEL_CONFIGS: dict[str, dict] = {
    "qwen2.5-omni-3b": {
        "family": "qwen2.5-omni",
        "device_map": "auto",
        "max_new_tokens": 4096,
        "context_length": 32768,
        "audio_tokens_per_second": 25.0,
    },
    "qwen2.5-omni-7b": {
        "family": "qwen2.5-omni",
        "device_map": "auto",
        "mps_device_map": "cpu",       # MPS causes >128GB memory blowup on 7B dense
        "max_new_tokens": 4096,
        "context_length": 32768,
        "audio_tokens_per_second": 25.0,
    },
    "qwen3-omni-30b-a3b-instruct": {
        "family": "qwen3-omni-moe",
        "device_map": "auto",          # MoE, ~3B active params — MPS fine
        "max_new_tokens": 8192,
        "thinker_max_new_tokens": 8192,
        "context_length": 65536,
        "audio_tokens_per_second": 13.0,
    },
    "qwen3-omni-30b-a3b-thinking": {
        "family": "qwen3-omni-moe",
        "device_map": "auto",
        "max_new_tokens": 16384,
        # Reasoning length grows with how much consultation the model hears, so
        # the 8192 that sufficed when audio was cut at 30s overflows on full
        # recordings — leaving raw chain-of-thought where the note should be.
        "thinker_max_new_tokens": 16384,
        "context_length": 65536,
        "audio_tokens_per_second": 13.0,
    },
}


def get_model_config(model_name: str) -> dict:
    """Look up model config by matching the model name against known configs.

    Returns matching config dict, or a default config if no match found.
    Matches are checked longest-key-first so "qwen3-omni-30b-a3b-thinking"
    matches before "qwen3-omni-30b".
    """
    name_lower = model_name.lower()
    # Sort by key length descending so more specific keys match first
    for key in sorted(MODEL_CONFIGS, key=len, reverse=True):
        if key in name_lower:
            return MODEL_CONFIGS[key]
    # Fallback: detect family from name, conservative defaults
    is_qwen3 = "qwen3" in name_lower
    return {
        "family": "qwen3-omni-moe" if is_qwen3 else "qwen2.5-omni",
        "device_map": "auto",
        "max_new_tokens": 8192,
        "context_length": 65536 if is_qwen3 else 32768,
        "audio_tokens_per_second": 13.0 if is_qwen3 else 25.0,
    }


def compute_max_audio_seconds(
    model_config: dict, override: float | None = None
) -> float | None:
    """Return the longest audio (in seconds) that fits the model context.

    Args:
        model_config: Entry from MODEL_CONFIGS (or the fallback).
        override: CLI value for --max-audio-seconds. A negative value means
            "no limit" and returns None; None means "derive from the config".

    Returns:
        Cap in seconds, or None for no cap.
    """
    if override is not None:
        return None if override < 0 else float(override)

    ctx = model_config.get("context_length")
    rate = model_config.get("audio_tokens_per_second")
    if not ctx or not rate:
        return None
    gen = model_config.get(
        "thinker_max_new_tokens", model_config.get("max_new_tokens", 8192)
    )
    budget = ctx - gen - PROMPT_TOKEN_RESERVE
    return max(60.0, budget / float(rate))


# Audio caps to retry with after a CUDA/MPS out-of-memory error, as fractions
# of the audio length that failed. Anything below OOM_RETRY_FLOOR_SEC is not
# worth attempting — it would throw away most of the consultation.
OOM_RETRY_FACTORS: tuple[float, ...] = (0.6, 0.35)
OOM_RETRY_FLOOR_SEC = 300.0

# A completed generation with fewer words than this is treated as corrupted
# output, not success. The shortest legitimate SOAP note across all four
# validation runs is 163 words; degenerate output observed from a misconfigured
# sharded run was ~15 "words" of noise.
DEGENERATE_SUMMARY_MIN_WORDS = 25


def _is_oom_error(error: str) -> bool:
    """True if a subprocess error string looks like an out-of-memory failure."""
    lowered = (error or "").lower()
    return "out of memory" in lowered or "outofmemory" in lowered


# device_map="auto" packs each visible GPU to capacity, which leaves no room for
# activations and the KV cache — the model loads and then OOMs allocating 2 MiB.
# When sharding across GPUs we cap each one and let the remainder offload to CPU.
GPU_HEADROOM_GIB = 10


def build_max_memory(num_gpus: int) -> dict[int | str, str] | None:
    """Per-device memory budget for multi-GPU sharding, or None for the default.

    Reserves GPU_HEADROOM_GIB on every card for activations and the KV cache;
    whatever does not fit spills to CPU, as in the single-GPU path.
    """
    if not num_gpus or num_gpus < 2 or not torch.cuda.is_available():
        return None
    max_memory: dict[int | str, str] = {}
    for i in range(min(num_gpus, torch.cuda.device_count())):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable = max(int(total_gib) - GPU_HEADROOM_GIB, 8)
        max_memory[i] = f"{usable}GiB"
    max_memory["cpu"] = "200GiB"
    return max_memory


def get_audio_duration(path: str) -> float | None:
    """Return audio duration in seconds, or None if it cannot be determined."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:  # pragma: no cover - best-effort diagnostic only
        return None


# ---------------------------------------------------------------------------
# Thinking model helpers
# ---------------------------------------------------------------------------

def is_thinking_model(model_name: str, flag: bool = False) -> bool:
    """Return True if the model is a thinking variant.

    Auto-detected from 'thinking' in the model name, or forced via flag.
    """
    return flag or "thinking" in model_name.lower()


def extract_thinking(text: str) -> str:
    """Extract content from the first <think>…</think> block, or empty string."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def has_unclosed_thinking(text: str) -> bool:
    """True if the model opened a <think> block and never closed it.

    That means generation hit its token budget mid-reasoning, so there is no
    SOAP note at all — `strip_thinking` finds no complete block and returns the
    raw chain-of-thought. Left undetected, such a record is written with
    success=true and poisons the metrics with reasoning text.
    """
    return "<think>" in text and "</think>" not in text


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def read_manifest(manifest_path: str, base_dir: str | None = None) -> list[dict[str, str]]:
    """Read a CSV manifest and return list of sample dicts.

    Args:
        manifest_path: Path to CSV manifest with at least 'id' and 'audio_path' columns.
        base_dir: Base directory for resolving relative audio paths.
                  Defaults to the manifest's parent directory.

    Returns:
        List of dicts with 'id' and 'audio_path' keys.
    """
    manifest_path_obj = Path(manifest_path)
    if base_dir is None:
        base_dir = str(manifest_path_obj.parent)

    samples = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"].strip()
            audio_path = row["audio_path"].strip()
            # Resolve relative paths against base_dir
            if not Path(audio_path).is_absolute():
                audio_path = str(Path(base_dir) / audio_path)
            samples.append({"id": sample_id, "audio_path": audio_path})
    return samples


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(output_path: str) -> set[str]:
    """Load IDs already processed from an existing JSONL output file.

    Records written before the audio-truncation fix have no `audio_sec` field.
    Those are NOT treated as complete: otherwise pulling the fix and re-running
    with --resume finishes instantly and silently leaves 300s/30s-truncated
    notes in place. Failed records are re-tried for the same reason.
    """
    completed: set[str] = set()
    stale = 0
    if not Path(output_path).exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("audio_sec") is None or not entry.get("success", True):
                        stale += 1
                        continue
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if stale:
        logger.warning(
            "{}: {} record(s) predate the audio-truncation fix or failed — "
            "they will be re-processed.", output_path, stale,
        )
    return completed


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------


def load_hf_samples(
    dataset_name: str, split: str, offset: int = 0, limit: int | None = None
) -> list[dict]:
    """Load samples from a HuggingFace WebDataset.

    Returns a list of dicts with keys: id, audio_bytes, reference_soap.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i < offset:
            continue
        if limit is not None and (i - offset) >= limit:
            break

        # Extract sample ID from the JSON metadata field.
        # The field may be a Python repr string (single quotes) or
        # standard JSON or already a dict.
        sample_id = f"sample_{i:04d}"
        meta_raw = item.get("json", "")
        if isinstance(meta_raw, dict):
            sample_id = str(meta_raw.get("id", sample_id))
        elif isinstance(meta_raw, (str, bytes, bytearray)):
            if isinstance(meta_raw, (bytes, bytearray)):
                meta_raw = meta_raw.decode("utf-8")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    sample_id = str(meta.get("id", sample_id))
                except json.JSONDecodeError:
                    try:
                        meta = ast.literal_eval(meta_raw)
                        sample_id = str(meta.get("id", sample_id))
                    except (ValueError, SyntaxError):
                        pass

        # Get audio bytes
        audio_bytes = item.get("opus", b"")
        if isinstance(audio_bytes, dict):
            audio_bytes = audio_bytes.get("bytes", b"")

        # Get reference SOAP note
        ref_soap = item.get("soap.txt", "")
        if isinstance(ref_soap, (bytes, bytearray)):
            ref_soap = ref_soap.decode("utf-8")

        samples.append({
            "id": sample_id,
            "audio_bytes": audio_bytes,
            "reference_soap": ref_soap,
        })

    logger.info(f"Loaded {len(samples)} samples from {dataset_name} [{split}]")
    return samples


def write_audio_tempfile(audio_bytes: bytes) -> str:
    """Write audio bytes to a temporary file and return the path.

    The caller is responsible for deleting the file after use.
    """
    fd, path = tempfile.mkstemp(suffix=".opus")
    # os.write() is a single write(2): on a full filesystem it returns a short
    # count instead of raising, which would hand the model a truncated
    # recording with no error anywhere. Wrap the fd in a buffered writer, which
    # loops until everything is written and raises on failure, then verify.
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    written = Path(path).stat().st_size
    if written != len(audio_bytes):
        Path(path).unlink(missing_ok=True)
        raise OSError(
            f"Short write to {path}: {written} of {len(audio_bytes)} bytes "
            "(is the temp filesystem full?)"
        )
    return path


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt(prompt_path: str | None) -> str:
    """Load prompt text from a YAML file or return the default prompt.

    Args:
        prompt_path: Path to a YAML file with a 'text' key, or None for default.

    Returns:
        The prompt text string.
    """
    if prompt_path is None:
        return DEFAULT_PROMPT

    import yaml  # type: ignore[import-untyped]

    with open(prompt_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "text" in data:
        return data["text"]
    raise ValueError(f"Prompt YAML must have a 'text' key, got: {list(data.keys()) if isinstance(data, dict) else type(data)}")


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def log_memory_usage(stage: str) -> None:
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(
        "Memory usage {}: RSS={:.2f} GB, VMS={:.2f} GB (PID={})",
        stage, mem_info.rss / (1024**3), mem_info.vms / (1024**3), os.getpid(),
    )


def log_device_info(stage: str) -> None:
    """Log available compute devices and per-device memory."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info("Device info ({}): {} CUDA device(s) visible", stage, n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            logger.info(
                "  cuda:{} — {} | {:.1f} GB total | {:.1f} GB free",
                i, props.name, total / 1024**3, free / 1024**3,
            )
    else:
        logger.info("Device info ({}): no CUDA available — using CPU/MPS", stage)


def _load_model_processor(
    model_name: str,
    device: str = "auto",
    use_vllm: bool = False,
    use_flash_attn2: bool = False,
    num_gpus: int = 1,
) -> tuple[Any, Any]:
    """Load model and processor.

    Auto-detects the model family (Qwen3-Omni-MoE vs Qwen2.5-Omni) and
    loads the appropriate classes.

    Args:
        model_name: HuggingFace model name or local path.
        device: Device to load the model on ("auto", "cpu", "mps", or CUDA ID).
        use_vllm: Use vLLM backend instead of transformers.
        use_flash_attn2: Enable Flash Attention 2.

    Returns:
        Tuple of (model, processor).
    """
    from qwen_omni_utils import process_mm_info  # noqa: F401

    # Detect model family from the name
    name_lower = model_name.lower()
    is_qwen3 = "qwen3" in name_lower

    processor_cls: Any
    if is_qwen3:
        from transformers import Qwen3OmniMoeProcessor
        processor_cls = Qwen3OmniMoeProcessor
    else:
        from transformers import Qwen2_5OmniProcessor
        processor_cls = Qwen2_5OmniProcessor

    # Resolve device_map and dtype from the device flag
    model_dtype: Any
    if device == "cpu":
        device_map = "cpu"
        model_dtype = torch.float32
    elif device == "mps":
        device_map = "auto"
        model_dtype = "auto"
    else:
        # "auto" or CUDA device ID — let accelerate decide
        device_map = "auto"
        model_dtype = "auto"

    if not use_vllm:
        model_cls: Any
        if is_qwen3:
            from transformers import Qwen3OmniMoeForConditionalGeneration
            model_cls = Qwen3OmniMoeForConditionalGeneration
        else:
            from transformers import Qwen2_5OmniForConditionalGeneration
            model_cls = Qwen2_5OmniForConditionalGeneration

        logger.info(
            "Loading {} (family={}, device={}, device_map={})",
            model_name, "qwen3-omni-moe" if is_qwen3 else "qwen2.5-omni",
            device, device_map,
        )

        max_memory = build_max_memory(num_gpus)
        load_kwargs: dict[str, Any] = {
            "dtype": model_dtype,
            "device_map": device_map,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
            logger.info("Sharding across {} GPUs, max_memory={}", num_gpus, max_memory)
        if use_flash_attn2:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        model = model_cls.from_pretrained(model_name, **load_kwargs)
    else:
        from vllm import LLM

        model = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 1, "video": 3, "audio": 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

    processor = processor_cls.from_pretrained(model_name)
    return model, processor


def _is_qwen3_model(model: Any) -> bool:
    """Check if the model is a Qwen3-Omni-MoE model (vs Qwen2.5-Omni)."""
    return "qwen3" in type(model).__module__.lower()


def run_model(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 8192,
    thinker_max_new_tokens: int = 8192,
    use_vllm: bool = False,
    use_audio_in_video: bool = True,
    return_audio: bool = False,
    thinking: bool = False,
    context_length: int | None = None,
) -> tuple[str, npt.NDArray[np.int16] | None, str]:
    """Run inference on the model with given messages.

    Handles both Qwen3-Omni-MoE (returns token IDs needing batch_decode)
    and Qwen2.5-Omni (returns text strings directly).

    Args:
        model: The loaded model (transformers or vLLM).
        processor: The model processor.
        messages: List of message dictionaries for the model.
        max_new_tokens: Maximum tokens to generate (from model config).
        thinker_max_new_tokens: Maximum tokens for Qwen3 thinker (from model config).
        use_vllm: Whether model is a vLLM instance.
        use_audio_in_video: Route audio through video pipeline.
        return_audio: Whether to return generated audio (ignored for thinking models).
        thinking: Use thinking-model generation parameters and strip <think> tags.
        context_length: Thinker context window, used only to warn when the
            prompt plus generation budget would not fit.

    Returns:
        Tuple of (generated_text, audio_output, thinking_text).
    """
    from qwen_omni_utils import process_mm_info

    log_memory_usage("before inference")

    if not use_vllm:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            # --- Audio truncation fix -------------------------------------
            # Both Omni processors delegate audio framing to a
            # WhisperFeatureExtractor whose __call__ defaults to
            # truncation=True with max_length=n_samples, where
            # n_samples = chunk_length * sampling_rate. `chunk_length` comes
            # from each checkpoint's preprocessor_config.json, so audio was
            # being silently cut at:
            #     Qwen2.5-Omni   chunk_length=300  ->  300 s
            #     Qwen3-Omni-MoE chunk_length unset -> default 30 -> 30 s
            # (The `n_samples: 4800000` key in those JSON files is overwritten
            # by WhisperFeatureExtractor.__init__, so it does not help.)
            # Passing truncation=False disables that cut and lets the whole
            # recording through; the audio towers are windowed and handle
            # arbitrary lengths. Length is instead bounded up-front by
            # --max-audio-seconds, derived from the thinker context window.
            truncation=False,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        n_prompt_tokens = int(inputs["input_ids"].shape[1])
        if "feature_attention_mask" in inputs:
            fe = processor.feature_extractor
            audio_sec = (
                float(inputs["feature_attention_mask"].sum())
                * fe.hop_length
                / fe.sampling_rate
            )
            logger.info(
                "Audio reaching the model: {:.1f}s | prompt tokens: {}",
                audio_sec, n_prompt_tokens,
            )
        else:
            logger.info("Prompt tokens: {}", n_prompt_tokens)

        # --max-audio-seconds is derived assuming the built-in prompt
        # (~170 tokens). A much longer custom prompt can eat the remaining
        # headroom, so say so rather than let the context silently overflow.
        if context_length:
            gen_budget = thinker_max_new_tokens if _is_qwen3_model(model) else max_new_tokens
            if n_prompt_tokens + gen_budget > context_length:
                logger.warning(
                    "Prompt ({} tokens) + generation budget ({}) exceeds the "
                    "{}-token context by {}. Lower --max-audio-seconds or "
                    "shorten the prompt.",
                    n_prompt_tokens, gen_budget, context_length,
                    n_prompt_tokens + gen_budget - context_length,
                )

        is_qwen3 = _is_qwen3_model(model)

        if is_qwen3:
            # Qwen3-Omni-MoE: returns (token_ids_object, audio)
            if thinking:
                text_ids, audio = model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=thinker_max_new_tokens,
                    use_audio_in_video=use_audio_in_video,
                )
            else:
                text_ids, audio = model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=thinker_max_new_tokens,
                    thinker_do_sample=False,
                    speaker="Ethan",
                    use_audio_in_video=use_audio_in_video,
                    return_audio=return_audio,
                )
            log_memory_usage("during inference")
            response = processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            # Qwen2.5-Omni: returns token_ids (or (token_ids, audio) if return_audio)
            #
            # The generation budget MUST be passed as `thinker_max_new_tokens`.
            # Qwen2_5OmniForConditionalGeneration.generate declares its own
            # `thinker_max_new_tokens: int = 1024`, seeds
            # `thinker_kwargs = {"max_new_tokens": thinker_max_new_tokens}`, and
            # merges caller kwargs with `if key not in thinker_kwargs` — so a
            # plain `max_new_tokens=` is silently DISCARDED and every note gets
            # cut off at 1024 tokens mid-sentence. Same failure shape as the
            # audio truncation this script fixes: a wrapper default quietly
            # overriding the caller.
            result = model.generate(
                **inputs,
                thinker_max_new_tokens=max_new_tokens,
                use_audio_in_video=use_audio_in_video,
                return_audio=return_audio,
            )
            if isinstance(result, tuple):
                text_ids, audio = result
            else:
                text_ids, audio = result, None
            log_memory_usage("during inference")
            # Decode token IDs to text
            response = processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        if thinking and has_unclosed_thinking(response):
            raise RuntimeError(
                "Generation ran out of tokens inside the <think> block "
                f"(thinker_max_new_tokens={thinker_max_new_tokens}); no SOAP "
                "note was produced. Raise thinker_max_new_tokens or lower "
                "--max-audio-seconds."
            )
        thinking_text = extract_thinking(response) if thinking else ""
        if thinking:
            response = strip_thinking(response)
        log_memory_usage("after inference")
        if audio is not None:
            audio = np.array(
                audio.reshape(-1).detach().cpu().numpy() * 32767
            ).astype(np.int16)
        return response, audio, thinking_text
    else:
        from vllm import SamplingParams

        if thinking:
            # Thinking model sampling params per HuggingFace model card
            sampling_params = SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384
            )
        else:
            sampling_params = SamplingParams(
                temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192
            )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video
        )
        vllm_inputs: dict[str, Any] = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {
                "use_audio_in_video": use_audio_in_video,
                # Same audio-truncation fix as the transformers branch above.
                # UNVERIFIED: vLLM filters mm_processor_kwargs against the
                # processor signature and may drop keys it only sees via
                # **kwargs. If you use --use-vllm, confirm the audio is not
                # being cut before trusting the results.
                "truncation": False,
            },
        }
        if images is not None:
            vllm_inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            vllm_inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            vllm_inputs["multi_modal_data"]["audio"] = audios
        log_memory_usage("during inference")
        outputs = model.generate(vllm_inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        thinking_text = extract_thinking(response) if thinking else ""
        if thinking:
            response = strip_thinking(response)
        log_memory_usage("after inference")
        return response, None, thinking_text


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

def process_audio_subprocess(
    sample_id: str,
    wav_path: str,
    queue: multiprocessing.Queue,  # type: ignore[type-arg]
    prompt_txt: str,
    model_name: str,
    device: str,
    use_vllm: bool,
    use_flash_attn2: bool,
    thinking: bool,
    max_new_tokens: int = 8192,
    thinker_max_new_tokens: int = 8192,
    mps_high_watermark_ratio: float = 0.0,
    max_audio_seconds: float | None = None,
    context_length: int | None = None,
    num_gpus: int = 1,
) -> None:
    """Process a single audio file in a subprocess for GPU memory isolation.

    Results are returned via the multiprocessing queue as a tuple:
    (success: bool, summary: str, elapsed_sec: float, error: str, thinking: str)
    """
    try:
        # GPU visibility.
        #
        # Default (num_gpus=1): restrict the subprocess to a single GPU so
        # device_map="auto" offloads excess layers to CPU instead of filling
        # every GPU with weights and leaving no room for the KV cache.
        #
        # num_gpus>1 shards the model across that many GPUs instead.
        # EXPERIMENTAL: only sound when the cards jointly fit the whole model
        # with headroom. 2x40GB for a ~60GB 30B model is marginal — observed
        # to hang past walltime or silently corrupt generation when MoE layers
        # spill to CPU. A single >=80GB card is the reliable configuration.
        if device in ("cuda", "auto"):
            if num_gpus and num_gpus > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in range(num_gpus)
                )
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif device not in ("cpu", "mps"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        if device == "mps":
            # Prevent MPS from silently consuming unlimited memory via disk swap.
            # When exceeded, PyTorch raises a clear RuntimeError instead.
            # Default 0.0 = let OS manage; override via env var or model config.
            os.environ.setdefault(
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                str(mps_high_watermark_ratio),
            )
        # Must be set BEFORE anything initializes CUDA — the allocator parses
        # this env var exactly once, on first CUDA touch (log_device_info below
        # already triggers _lazy_init). 7+ GiB was sitting
        # reserved-but-unallocated to fragmentation in the observed OOM;
        # expandable segments give that back.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        logger.info("Subprocess PID={} | device={}", os.getpid(), device)
        log_device_info("before model load")
        t0 = time.time()

        model, processor = _load_model_processor(
            model_name, device=device, use_vllm=use_vllm,
            use_flash_attn2=use_flash_attn2, num_gpus=num_gpus,
        )

        # Log where the model was placed after loading
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            logger.info("Model hf_device_map: {}", model.hf_device_map)
        elif hasattr(model, "device"):
            logger.info("Model device: {}", model.device)
        log_device_info("after model load")

        audio_ele: dict[str, Any] = {"type": "audio", "audio": wav_path}
        duration = get_audio_duration(wav_path)
        if duration is not None:
            logger.info("[{}] Audio duration: {:.1f}s", sample_id, duration)
        if max_audio_seconds is not None:
            # qwen_omni_utils.process_audio_info honours audio_end by passing
            # it to librosa.load(duration=...), so the cut happens at load
            # time rather than silently inside the feature extractor.
            audio_ele["audio_end"] = float(max_audio_seconds)
            if duration is not None and duration > max_audio_seconds:
                logger.warning(
                    "[{}] Audio is {:.1f}s but the model context only fits "
                    "{:.1f}s — dropping the last {:.1f}s. Raise "
                    "--max-audio-seconds only if the model context allows it.",
                    sample_id, duration, max_audio_seconds,
                    duration - max_audio_seconds,
                )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_txt},
                    audio_ele,
                ],
            }
        ]

        response, _, thinking_text = run_model(
            model, processor, messages,
            max_new_tokens=max_new_tokens,
            thinker_max_new_tokens=thinker_max_new_tokens,
            use_vllm=use_vllm,
            use_audio_in_video=True,
            return_audio=False,
            thinking=thinking,
            context_length=context_length,
        )

        elapsed = time.time() - t0
        # Ensure all values are plain Python types for pickling across processes
        queue.put((True, str(response), elapsed, "", str(thinking_text)))
    except Exception as e:
        elapsed = time.time() - t0 if "t0" in dir() else 0.0
        queue.put((False, "", elapsed, str(e), ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the omni step."""
    parser = argparse.ArgumentParser(
        description="Omni step: direct audio-to-summary using Qwen3-Omni"
    )

    # Data source (HF dataset is the default; manifest is the fallback)
    parser.add_argument(
        "--dataset",
        default="BeTraC/betrac-2026",
        help="HuggingFace dataset name (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to process (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (useful for testing)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N samples (for array job partitioning)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to CSV manifest file (overrides --dataset/--split)",
    )

    # Output
    parser.add_argument(
        "--output", required=True,
        help="Path to output JSONL file",
    )

    # Model
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="HuggingFace model name or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="CUDA device ID(s) or 'auto' (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Path to prompt YAML file with 'text' key (default: built-in SOAP prompt)",
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM backend instead of transformers",
    )
    parser.add_argument(
        "--use-flash-attn2", action="store_true",
        help="Enable Flash Attention 2 (requires compatible GPU)",
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Base directory for resolving relative audio paths in manifest",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip IDs already present in the output JSONL file",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Force thinking-model mode (auto-detected when 'thinking' is in model name)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help=(
            "GPUs to shard the model across (default: 1 + CPU offload). "
            "EXPERIMENTAL: only useful when the cards jointly hold the whole "
            "model with headroom. 2x40GB for a 30B model is marginal — "
            "observed to hang or corrupt output; prefer a single card with "
            ">=80GB. See README 'Audio length'."
        ),
    )
    parser.add_argument(
        "--max-audio-seconds", type=float, default=None,
        help=(
            "Longest audio (seconds) fed to the model. Default: derived from "
            "the model context window (~1130s for Qwen2.5-Omni, ~4400s for "
            "Qwen3-Omni). Use a negative value for no limit."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _process_one_sample(
    sample_id: str,
    wav_path: str,
    args: argparse.Namespace,
    prompt_txt: str,
    device: str,
    thinking: bool,
    output_path: Path,
    model_config: dict | None = None,
) -> bool:
    """Process a single audio sample via subprocess. Returns True on success."""
    mc = model_config or {}
    max_audio_seconds = compute_max_audio_seconds(
        mc, getattr(args, "max_audio_seconds", None)
    )
    audio_sec = get_audio_duration(wav_path)

    # --max-audio-seconds is derived from the *context window*, but on a small
    # GPU the binding constraint is device memory: a 30B model with CPU offload
    # OOMs on a 40GB card well before it runs out of context. Rather than lose
    # the sample, retry with progressively less audio. Each attempt runs in its
    # own subprocess, so the previous attempt's GPU memory is already released.
    attempt_caps: list[float | None] = [max_audio_seconds]
    if audio_sec is not None:
        effective = audio_sec if max_audio_seconds is None else min(audio_sec, max_audio_seconds)
        for factor in OOM_RETRY_FACTORS:
            shorter = effective * factor
            if shorter >= OOM_RETRY_FLOOR_SEC:
                attempt_caps.append(shorter)

    ok, summary, elapsed, error, thinking_text = (
        False, "", 0.0, "No result returned from subprocess", "",
    )
    used_cap = max_audio_seconds
    for attempt, cap in enumerate(attempt_caps):
        if attempt:
            logger.warning(
                "[{}] Retrying with audio capped at {:.0f}s (attempt {}/{}) "
                "after out-of-memory",
                sample_id, cap, attempt + 1, len(attempt_caps),
            )
        used_cap = cap
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()  # type: ignore[type-arg]
        p = ctx.Process(
            target=process_audio_subprocess,
            args=(sample_id, wav_path, queue, prompt_txt,
                  args.model, device, args.use_vllm, args.use_flash_attn2, thinking),
            kwargs={
                "max_new_tokens": mc.get("max_new_tokens", 8192),
                "thinker_max_new_tokens": mc.get("thinker_max_new_tokens", 8192),
                "mps_high_watermark_ratio": mc.get("mps_high_watermark_ratio", 0.0),
                "max_audio_seconds": cap,
                "context_length": mc.get("context_length"),
                "num_gpus": getattr(args, "num_gpus", 1),
            },
        )
        p.start()
        # Drain the queue BEFORE joining: a child whose result exceeds the pipe
        # buffer (~64-128 KB — reachable at the 16384-token thinking budget)
        # blocks in its queue feeder thread until the parent reads, and
        # p.join() would then deadlock forever.
        got = None
        while True:
            try:
                got = queue.get(timeout=10)
                break
            except Exception:  # queue.Empty
                if not p.is_alive():
                    # child exited; one final grab in case the payload landed
                    # between the timeout and the liveness check
                    try:
                        got = queue.get(timeout=5)
                    except Exception:
                        pass
                    break
        p.join()

        if got is not None:
            ok, summary, elapsed, error, thinking_text = got
        else:
            ok, summary, elapsed, error, thinking_text = (
                False, "", 0.0, "No result returned from subprocess", "",
            )

        if ok and len(summary.split()) < DEGENERATE_SUMMARY_MIN_WORDS:
            # Observed with 2x40GB sharded MoE inference: generation completed
            # "successfully" but emitted ~90 chars of noise. Never record that
            # as a usable result.
            ok = False
            error = (
                f"Degenerate output: {len(summary.split())} words "
                "(likely corrupted generation — check the GPU/offload config)"
            )

        if ok or not _is_oom_error(error):
            break

    max_audio_seconds = used_cap
    audio_used_sec = (
        None if audio_sec is None
        else (audio_sec if max_audio_seconds is None
              else min(audio_sec, max_audio_seconds))
    )

    result = {
        "id": sample_id,
        "summary": summary,
        "thinking": thinking_text,
        "omni_time_sec": round(elapsed, 3),
        "total_time_sec": round(elapsed, 3),
        # Audio coverage, so a silent truncation regression is visible in the
        # output itself rather than only in the logs.
        "audio_sec": None if audio_sec is None else round(audio_sec, 2),
        "audio_used_sec": (
            None if audio_used_sec is None else round(audio_used_sec, 2)
        ),
        "success": ok,
        "error": error,
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if ok:
        logger.info("[{}] Done in {:.1f}s", sample_id, elapsed)
    else:
        logger.error("[{}] Failed: {}", sample_id, error)
    return ok


def _process_hf_samples(
    samples: list[dict],
    args: argparse.Namespace,
    prompt_txt: str,
    device: str,
    thinking: bool,
    output_path: Path,
    completed_ids: set[str],
    model_config: dict | None = None,
) -> tuple[int, int, int]:
    """Process samples loaded from HuggingFace dataset.

    Returns (successes, failures, skipped).
    """
    successes = 0
    failures = 0
    skipped = 0
    total = len(samples)

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["id"]

        if sample_id in completed_ids:
            logger.info("[{}/{}] Skipping {} (already processed)", idx, total, sample_id)
            skipped += 1
            continue

        # Write audio bytes to temp file
        audio_path = write_audio_tempfile(sample["audio_bytes"])
        try:
            logger.info("[{}/{}] Processing: {}", idx, total, sample_id)
            ok = _process_one_sample(
                sample_id, audio_path, args, prompt_txt,
                device, thinking, output_path, model_config,
            )
            if ok:
                successes += 1
            else:
                failures += 1
        finally:
            Path(audio_path).unlink(missing_ok=True)

    return successes, failures, skipped


def _process_manifest_samples(
    samples: list[dict[str, str]],
    args: argparse.Namespace,
    prompt_txt: str,
    device: str,
    thinking: bool,
    output_path: Path,
    completed_ids: set[str],
    model_config: dict | None = None,
) -> tuple[int, int, int]:
    """Process samples from a CSV manifest (fallback path).

    Returns (successes, failures, skipped).
    """
    successes = 0
    failures = 0
    skipped = 0
    total = len(samples)

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["id"]
        wav_path = sample["audio_path"]

        if sample_id in completed_ids:
            logger.info("[{}/{}] Skipping {} (already processed)", idx, total, sample_id)
            skipped += 1
            continue

        logger.info("[{}/{}] Processing: {} ({})", idx, total, sample_id, wav_path)
        ok = _process_one_sample(
            sample_id, wav_path, args, prompt_txt,
            device, thinking, output_path, model_config,
        )
        if ok:
            successes += 1
        else:
            failures += 1

    return successes, failures, skipped


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Look up model-specific configuration
    model_config = get_model_config(args.model)
    logger.info("Model config: {}", model_config)

    # Detect thinking mode
    thinking = is_thinking_model(args.model, args.thinking)
    if thinking:
        logger.info("Thinking model detected — using thinking-mode generation")

    # Resolve device
    if args.device == "auto":
        device = detect_device()
        # Check if config has an MPS-specific override (e.g. 7B → CPU on MPS)
        if device == "mps" and "mps_device_map" in model_config:
            device = model_config["mps_device_map"]
            logger.info(
                "Model config overrides MPS → {} for {}", device, args.model
            )
        logger.info("Auto-detected device: {}", device)
    else:
        device = args.device

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load prompt
    prompt_txt = load_prompt(args.prompt)
    logger.info("Prompt loaded ({} chars)", len(prompt_txt))

    # Load samples from HuggingFace or CSV manifest
    use_hf = args.manifest is None
    if use_hf:
        hf_samples = load_hf_samples(args.dataset, args.split, offset=args.offset, limit=args.limit)
        logger.info(
            "Loaded {} samples from {} [{}]",
            len(hf_samples), args.dataset, args.split,
        )
    else:
        manifest_samples = read_manifest(args.manifest, args.base_dir)
        logger.info("Manifest: {} samples from {}", len(manifest_samples), args.manifest)

        # Validate audio files exist
        missing = [s for s in manifest_samples if not Path(s["audio_path"]).exists()]
        if missing:
            for s in missing:
                logger.error("Audio file not found: {} (id={})", s["audio_path"], s["id"])
            logger.error("{} audio file(s) missing. Aborting.", len(missing))
            return 1

    # Resume support
    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output)
        if completed_ids:
            logger.info("Resume: skipping {} already-processed IDs", len(completed_ids))

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process samples
    if use_hf:
        successes, failures, skipped = _process_hf_samples(
            hf_samples, args, prompt_txt, device, thinking,
            output_path, completed_ids, model_config,
        )
        total = len(hf_samples)
    else:
        successes, failures, skipped = _process_manifest_samples(
            manifest_samples, args, prompt_txt, device, thinking,
            output_path, completed_ids, model_config,
        )
        total = len(manifest_samples)

    # Summary
    logger.info("=== Summary ===")
    logger.info(
        "Processed: {}, Skipped: {}, Failed: {} (Total: {})",
        successes, skipped, failures, total,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
