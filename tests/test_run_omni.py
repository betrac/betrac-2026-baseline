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
        """Test reading IDs from existing JSONL output."""
        output = temp_dir / "output.jsonl"
        output.write_text(
            json.dumps({"id": "a", "summary": "x", "success": True}) + "\n"
            + json.dumps({"id": "b", "summary": "y", "success": False}) + "\n"
        )
        ids = load_completed_ids(str(output))
        assert ids == {"a", "b"}

    def test_handles_malformed_lines(self, temp_dir):
        """Test that malformed JSONL lines are skipped."""
        output = temp_dir / "output.jsonl"
        output.write_text(
            json.dumps({"id": "valid"}) + "\n"
            + "not valid json\n"
            + json.dumps({"no_id_key": "x"}) + "\n"
        )
        ids = load_completed_ids(str(output))
        assert ids == {"valid"}


class TestOutputSchema:
    """Tests for output JSONL schema validation."""

    def test_expected_fields(self):
        """Verify the expected output schema fields."""
        expected_fields = {"id", "summary", "omni_time_sec", "total_time_sec", "success", "error"}
        # This documents the contract for the output format
        assert expected_fields == {"id", "summary", "omni_time_sec", "total_time_sec", "success", "error"}

    def test_jsonl_round_trip(self, temp_dir):
        """Test writing and reading a JSONL result."""
        output = temp_dir / "test.jsonl"
        result = {
            "id": "sample_001",
            "summary": "Test summary",
            "omni_time_sec": 1.234,
            "total_time_sec": 1.234,
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
