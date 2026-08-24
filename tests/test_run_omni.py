"""Unit tests for steps/omni/run_omni.py

Note: Model loading and inference tests are skipped since they require
the actual Qwen3-Omni model (very large). These tests focus on the
manifest reading, output format, prompt loading, and resume logic.
"""

import json
import sys
from pathlib import Path

import pytest

# Add steps/omni to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "steps" / "omni"))

from run_omni import (
    DEFAULT_PROMPT,
    GPU_HEADROOM_GIB,
    MODEL_CONFIGS,
    build_max_memory,
    has_unclosed_thinking,
    OOM_RETRY_FACTORS,
    OOM_RETRY_FLOOR_SEC,
    _is_oom_error,
    compute_max_audio_seconds,
    get_model_config,
    load_completed_ids,
    load_prompt,
    read_manifest,
)


class TestReadManifest:
    """Tests for CSV manifest reading."""

    def test_reads_manifest_with_absolute_paths(self, sample_manifest_csv):
        """Test reading a manifest with absolute audio paths."""
        samples = read_manifest(str(sample_manifest_csv))
        assert len(samples) == 2
        assert samples[0]["id"] == "test_001"
        assert samples[1]["id"] == "test_002"
        assert Path(samples[0]["audio_path"]).is_absolute()

    def test_reads_manifest_with_relative_paths(self, sample_manifest_relative, temp_dir):
        """Test that relative paths are resolved against base_dir."""
        samples = read_manifest(str(sample_manifest_relative), base_dir=str(temp_dir))
        assert len(samples) == 1
        assert samples[0]["id"] == "rel_001"
        assert Path(samples[0]["audio_path"]).is_absolute()
        assert Path(samples[0]["audio_path"]).exists()

    def test_relative_paths_default_to_manifest_parent(self, sample_manifest_relative):
        """Test that relative paths default to manifest's parent directory."""
        samples = read_manifest(str(sample_manifest_relative))
        assert len(samples) == 1
        # Should resolve relative to manifest's parent dir
        assert Path(samples[0]["audio_path"]).exists()

    def test_empty_manifest_raises_no_error(self, temp_dir):
        """Test that an empty manifest (header only) returns empty list."""
        manifest = temp_dir / "empty.csv"
        manifest.write_text("id,audio_path\n")
        samples = read_manifest(str(manifest))
        assert samples == []


class TestLoadPrompt:
    """Tests for prompt loading."""

    def test_default_prompt_returned_when_none(self):
        """Test that default SOAP prompt is returned when no path given."""
        prompt = load_prompt(None)
        assert prompt == DEFAULT_PROMPT
        assert "SOAP" in prompt

    def test_loads_yaml_prompt(self, sample_prompt_yaml):
        """Test loading prompt from YAML file."""
        prompt = load_prompt(str(sample_prompt_yaml))
        assert "Test prompt" in prompt

    def test_invalid_yaml_raises_error(self, temp_dir):
        """Test that YAML without 'text' key raises ValueError."""
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("foo: bar\n")
        with pytest.raises(ValueError, match="text"):
            load_prompt(str(bad_yaml))


class TestLoadCompletedIds:
    """Tests for resume support."""

    def test_returns_empty_for_nonexistent_file(self):
        """Test that nonexistent file returns empty set."""
        ids = load_completed_ids("/nonexistent/path.jsonl")
        assert ids == set()

    def test_reads_existing_jsonl(self, temp_dir):
        """Successful post-fix records count as complete; failures do not."""
        output = temp_dir / "output.jsonl"
        output.write_text(
            json.dumps({"id": "a", "summary": "x", "success": True,
                        "audio_sec": 500.0, "audio_used_sec": 500.0}) + "\n"
            + json.dumps({"id": "b", "summary": "", "success": False,
                          "audio_sec": 500.0, "audio_used_sec": 500.0}) + "\n"
        )
        ids = load_completed_ids(str(output))
        assert ids == {"a"}, "a failed record must be retried, not skipped"

    def test_skips_records_predating_the_truncation_fix(self, temp_dir):
        """Records with no audio_sec were produced by the buggy code.

        Treating them as complete would make `--resume` a silent no-op after
        pulling the fix, leaving 300s/30s-truncated notes in place.
        """
        output = temp_dir / "output.jsonl"
        output.write_text(
            json.dumps({"id": "old", "summary": "x", "success": True}) + "\n"
            + json.dumps({"id": "new", "summary": "y", "success": True,
                          "audio_sec": 500.0, "audio_used_sec": 500.0}) + "\n"
        )
        ids = load_completed_ids(str(output))
        assert ids == {"new"}
        assert "old" not in ids

    def test_handles_malformed_lines(self, temp_dir):
        """Test that malformed JSONL lines are skipped."""
        output = temp_dir / "output.jsonl"
        output.write_text(
            json.dumps({"id": "valid", "success": True, "audio_sec": 1.0}) + "\n"
            + "not valid json\n"
            + json.dumps({"no_id_key": "x"}) + "\n"
        )
        ids = load_completed_ids(str(output))
        assert ids == {"valid"}


class TestOutputSchema:
    """Tests for output JSONL schema validation."""

    def test_expected_fields(self):
        """The record built by _process_one_sample must carry these fields."""
        expected_fields = {
            "id", "summary", "thinking", "omni_time_sec", "total_time_sec",
            "audio_sec", "audio_used_sec", "success", "error",
        }
        source = (Path(__file__).parent.parent / "steps" / "omni" / "run_omni.py").read_text()
        # the result-dict construction in _process_one_sample
        block = source.split("result = {", 1)[1].split("}", 1)[0]
        for field in expected_fields:
            assert f'"{field}"' in block, (
                f"output record no longer carries {field!r} — "
                "audio coverage would become invisible in the results"
            )

    def test_jsonl_round_trip(self, temp_dir):
        """Test writing and reading a JSONL result."""
        output = temp_dir / "test.jsonl"
        result = {
            "id": "sample_001",
            "summary": "Test summary",
            "omni_time_sec": 1.234,
            "total_time_sec": 1.234,
            "audio_sec": 512.5,
            "audio_used_sec": 512.5,
            "success": True,
            "error": "",
        }
        with open(output, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        ids = load_completed_ids(str(output))
        assert "sample_001" in ids

        with open(output) as f:
            loaded = json.loads(f.readline())
        assert loaded["id"] == "sample_001"
        assert loaded["summary"] == "Test summary"
        assert loaded["success"] is True


class TestAudioLength:
    """Regression tests for the silent audio-truncation bug.

    The Omni processors delegate framing to a WhisperFeatureExtractor that
    truncates to `chunk_length` seconds by default (300 s for Qwen2.5-Omni,
    30 s for Qwen3-Omni-MoE). run_omni.py must keep passing truncation=False.
    """

    def test_processor_call_disables_truncation(self):
        """run_model must pass truncation=False to the processor."""
        source = (Path(__file__).parent.parent / "steps" / "omni" / "run_omni.py").read_text()
        assert "truncation=False" in source, (
            "run_model no longer passes truncation=False — audio will be "
            "silently cut at 300s (Qwen2.5-Omni) or 30s (Qwen3-Omni)."
        )

    @pytest.mark.parametrize("model_id", sorted(MODEL_CONFIGS))
    def test_every_model_config_declares_an_audio_budget(self, model_id):
        """Each known model needs context_length + audio_tokens_per_second."""
        config = MODEL_CONFIGS[model_id]
        assert config["context_length"] > 0
        assert config["audio_tokens_per_second"] > 0

    def test_cap_exceeds_the_buggy_limits(self):
        """The derived cap must be well above the old hidden 300s/30s cuts."""
        qwen25 = compute_max_audio_seconds(get_model_config("Qwen/Qwen2.5-Omni-3B"))
        qwen3 = compute_max_audio_seconds(get_model_config("Qwen/Qwen3-Omni-30B-A3B-Instruct"))
        assert qwen25 > 1000, qwen25
        assert qwen3 > 4000, qwen3

    def test_cap_fits_the_context_window(self):
        """Prompt audio tokens plus generation must fit the context window."""
        for name, config in MODEL_CONFIGS.items():
            cap = compute_max_audio_seconds(config)
            gen = config.get("thinker_max_new_tokens", config["max_new_tokens"])
            audio_tokens = cap * config["audio_tokens_per_second"]
            assert audio_tokens + gen <= config["context_length"], name

    def test_explicit_override_wins(self):
        config = get_model_config("Qwen/Qwen2.5-Omni-3B")
        assert compute_max_audio_seconds(config, 600.0) == 600.0

    def test_negative_override_means_no_cap(self):
        config = get_model_config("Qwen/Qwen2.5-Omni-3B")
        assert compute_max_audio_seconds(config, -1.0) is None

    def test_unknown_model_still_gets_a_budget(self):
        cap = compute_max_audio_seconds(get_model_config("Some/Unknown-Omni-Model"))
        assert cap is not None and cap > 300


class TestMultiGpuSharding:
    """device_map="auto" packs GPUs full and then OOMs allocating 2 MiB."""

    def test_single_gpu_uses_default_device_map(self):
        assert build_max_memory(1) is None
        assert build_max_memory(0) is None

    def test_headroom_is_reserved(self):
        """Observed: GPU 1 filled to 39.48 of 39.49 GiB, leaving nothing."""
        assert GPU_HEADROOM_GIB >= 8, (
            "a 30B model needs room for activations and the KV cache on top "
            "of weights; too little headroom reproduces the load-time OOM"
        )

    def test_multi_gpu_budget_shape(self, monkeypatch):
        import run_omni

        class _Props:
            total_memory = 40 * 1024 ** 3

        monkeypatch.setattr(run_omni.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(run_omni.torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(run_omni.torch.cuda, "get_device_properties", lambda i: _Props())
        mm = build_max_memory(2)
        assert set(mm) == {0, 1, "cpu"}
        assert mm[0] == mm[1] == f"{40 - GPU_HEADROOM_GIB}GiB"
        assert mm["cpu"].endswith("GiB")


class TestGenerationBudget:
    """Guards against generation budgets being silently overridden."""

    def test_qwen25_passes_thinker_max_new_tokens(self):
        """Qwen2.5-Omni's generate() ignores a plain max_new_tokens.

        Qwen2_5OmniForConditionalGeneration.generate declares its own
        `thinker_max_new_tokens: int = 1024` and merges caller kwargs with
        `if key not in thinker_kwargs`, so an unprefixed `max_new_tokens=`
        is discarded and every note is cut off at 1024 tokens.
        """
        source = (Path(__file__).parent.parent / "steps" / "omni" / "run_omni.py").read_text()
        qwen25_call = source.split("# Qwen2.5-Omni: returns token_ids")[1]
        qwen25_call = qwen25_call.split("log_memory_usage")[0]
        assert "thinker_max_new_tokens=max_new_tokens" in qwen25_call, (
            "the Qwen2.5 branch must pass thinker_max_new_tokens, otherwise "
            "output is silently capped at 1024 tokens"
        )
        # and must NOT pass the ineffective unprefixed form
        assert "\n                max_new_tokens=" not in qwen25_call

    def test_thinking_model_has_headroom(self):
        """Reasoning grows with audio length; 8192 overflowed on full audio."""
        cfg = MODEL_CONFIGS["qwen3-omni-30b-a3b-thinking"]
        assert cfg["thinker_max_new_tokens"] >= 16384

    @pytest.mark.parametrize("text,expected", [
        ("<think>reasoning ran out of budget", True),
        ("<think>done</think>S: note", False),
        ("S: plain note with no thinking", False),
    ])
    def test_detects_unclosed_thinking(self, text, expected):
        assert has_unclosed_thinking(text) is expected


class TestDegenerateOutputGuard:
    """A completed generation of near-empty noise must not be success=true.

    Observed: a misconfigured 2x40GB sharded run emitted ~90 chars of noise
    with success=true, which --resume would then treat as done forever.
    """

    def test_threshold_is_below_any_legitimate_note(self):
        from run_omni import DEGENERATE_SUMMARY_MIN_WORDS
        # shortest real SOAP note across all four validation runs: 163 words
        assert 0 < DEGENERATE_SUMMARY_MIN_WORDS <= 100

    def test_guard_is_wired_into_the_retry_loop(self):
        source = (Path(__file__).parent.parent / "steps" / "omni" / "run_omni.py").read_text()
        assert "DEGENERATE_SUMMARY_MIN_WORDS" in source.split("def _process_one_sample")[1].split("def _process_hf_samples")[0]


class TestOomRetry:
    """The context-derived cap can still exceed GPU memory on a small card."""

    @pytest.mark.parametrize("msg", [
        "CUDA out of memory. Tried to allocate 4.08 GiB.",
        "MPS backend out of memory",
        "torch.OutOfMemoryError",
    ])
    def test_detects_oom(self, msg):
        assert _is_oom_error(msg)

    @pytest.mark.parametrize("msg", ["", "No result returned from subprocess",
                                     "FileNotFoundError: audio.opus"])
    def test_ignores_other_errors(self, msg):
        assert not _is_oom_error(msg)

    def test_retry_factors_shrink_monotonically(self):
        assert list(OOM_RETRY_FACTORS) == sorted(OOM_RETRY_FACTORS, reverse=True)
        assert all(0 < f < 1 for f in OOM_RETRY_FACTORS)

    def test_retries_stay_above_the_floor(self):
        """A 1096s dialog that OOMs should retry at lengths worth keeping."""
        caps = [1096.0 * f for f in OOM_RETRY_FACTORS
                if 1096.0 * f >= OOM_RETRY_FLOOR_SEC]
        assert caps, "no usable retry cap for a ~18 minute dialog"
        assert all(c >= OOM_RETRY_FLOOR_SEC for c in caps)
        # every retry must still beat the 300s cut the bug imposed
        assert all(c >= 300.0 for c in caps)


class TestModelLoading:
    """Tests for model loading (skipped without GPU/model)."""

    @pytest.mark.skip(reason="Requires Qwen3-Omni model")
    def test_load_model_transformers(self):
        pass

    @pytest.mark.skip(reason="Requires vLLM and GPU")
    def test_load_model_vllm(self):
        pass


class TestIntegration:
    """Integration tests requiring full environment."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires full model setup")
    def test_end_to_end_processing(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
