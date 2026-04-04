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

MODEL_CONFIGS: dict[str, dict] = {
    "qwen2.5-omni-3b": {
        "family": "qwen2.5-omni",
        "device_map": "auto",
        "max_new_tokens": 4096,
    },
    "qwen2.5-omni-7b": {
        "family": "qwen2.5-omni",
        "device_map": "auto",
        "mps_device_map": "cpu",       # MPS causes >128GB memory blowup on 7B dense
        "max_new_tokens": 4096,
    },
    "qwen3-omni-30b-a3b-instruct": {
        "family": "qwen3-omni-moe",
        "device_map": "auto",          # MoE, ~3B active params — MPS fine
        "max_new_tokens": 8192,
        "thinker_max_new_tokens": 8192,
    },
    "qwen3-omni-30b-a3b-thinking": {
        "family": "qwen3-omni-moe",
        "device_map": "auto",
        "max_new_tokens": 32768,
        "thinker_max_new_tokens": 32768,
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
    }


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
    """Load IDs already processed from an existing JSONL output file."""
    completed: set[str] = set()
    if not Path(output_path).exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------


def load_hf_samples(
    dataset_name: str, split: str, limit: int | None = None
) -> list[dict]:
    """Load samples from a HuggingFace WebDataset.

    Returns a list of dicts with keys: id, audio_bytes, reference_soap.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
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
    try:
        os.write(fd, audio_bytes)
    finally:
        os.close(fd)
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

        if use_flash_attn2:
            model = model_cls.from_pretrained(
                model_name,
                dtype=model_dtype,
                attn_implementation="flash_attention_2",
                device_map=device_map,
            )
        else:
            model = model_cls.from_pretrained(
                model_name, device_map=device_map, dtype=model_dtype
            )
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
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(model.device).to(model.dtype)

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
            result = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
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
) -> None:
    """Process a single audio file in a subprocess for GPU memory isolation.

    Results are returned via the multiprocessing queue as a tuple:
    (success: bool, summary: str, elapsed_sec: float, error: str, thinking: str)
    """
    try:
        # Only restrict CUDA devices when explicit GPU IDs given
        if device not in ("auto", "cuda", "cpu", "mps"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        if device == "mps":
            # Prevent MPS from silently consuming unlimited memory via disk swap.
            # When exceeded, PyTorch raises a clear RuntimeError instead.
            # Default 0.0 = let OS manage; override via env var or model config.
            os.environ.setdefault(
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                str(mps_high_watermark_ratio),
            )
        logger.info("Subprocess PID={} | device={}", os.getpid(), device)
        log_device_info("before model load")
        t0 = time.time()

        model, processor = _load_model_processor(
            model_name, device=device, use_vllm=use_vllm,
            use_flash_attn2=use_flash_attn2,
        )

        # Log where the model was placed after loading
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            logger.info("Model hf_device_map: {}", model.hf_device_map)
        elif hasattr(model, "device"):
            logger.info("Model device: {}", model.device)
        log_device_info("after model load")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_txt},
                    {"type": "audio", "audio": wav_path},
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
        },
    )
    p.start()
    p.join()

    if not queue.empty():
        ok, summary, elapsed, error, thinking_text = queue.get()
    else:
        ok, summary, elapsed, error, thinking_text = (
            False, "", 0.0, "No result returned from subprocess", "",
        )

    result = {
        "id": sample_id,
        "summary": summary,
        "thinking": thinking_text,
        "omni_time_sec": round(elapsed, 3),
        "total_time_sec": round(elapsed, 3),
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
        hf_samples = load_hf_samples(args.dataset, args.split, limit=args.limit)
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
