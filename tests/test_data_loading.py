"""Tests for HuggingFace dataset loading and OPUS audio handling.

Integration tests (marked @pytest.mark.integration) require HuggingFace
access and download real data from BeTraC/betrac-2026.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add steps/omni to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "steps" / "omni"))


# ---------------------------------------------------------------------------
# Unit tests (no network access required)
# ---------------------------------------------------------------------------


class TestWriteAudioTempfile:
    """Tests for writing audio bytes to a temporary file."""

    def test_returns_path_string(self):
        from run_omni import write_audio_tempfile

        fake_audio = b"\x00" * 100
        path = write_audio_tempfile(fake_audio)
        try:
            assert isinstance(path, str)
            assert Path(path).exists()
            assert Path(path).suffix == ".opus"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_contains_original_bytes(self):
        from run_omni import write_audio_tempfile

        fake_audio = b"\x01\x02\x03\x04\x05"
        path = write_audio_tempfile(fake_audio)
        try:
            assert Path(path).read_bytes() == fake_audio
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadHfSamples:
    """Tests for loading samples from HuggingFace dataset."""

    def test_returns_list_of_dicts(self):
        from run_omni import load_hf_samples

        mock_samples = [
            {
                "__key__": "dialog_0001_0001",
                "opus": b"\x00" * 100,
                "soap.txt": "S: Patient presents...",
                "json": json.dumps({"id": "dialog_0001_0001"}),
            },
            {
                "__key__": "dialog_0001_0002",
                "opus": b"\x00" * 200,
                "soap.txt": "S: Patient reports...",
                "json": json.dumps({"id": "dialog_0001_0002"}),
            },
        ]

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_samples
            samples = load_hf_samples("fake/dataset", "validation", limit=2)

        assert isinstance(samples, list)
        assert len(samples) == 2
        assert all(isinstance(s, dict) for s in samples)

    def test_sample_has_required_keys(self):
        from run_omni import load_hf_samples

        mock_samples = [
            {
                "__key__": "dialog_0001_0001",
                "opus": b"\x00" * 100,
                "soap.txt": "S: ...",
                "json": json.dumps({"id": "dialog_0001_0001"}),
            },
        ]

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_samples
            samples = load_hf_samples("fake/dataset", "validation", limit=1)

        sample = samples[0]
        assert "id" in sample
        assert "audio_bytes" in sample
        assert "reference_soap" in sample

    def test_limit_parameter(self):
        from run_omni import load_hf_samples

        mock_samples = [
            {
                "__key__": f"dialog_{i:04d}",
                "opus": b"\x00",
                "soap.txt": "soap",
                "json": json.dumps({"id": f"dialog_{i:04d}"}),
            }
            for i in range(10)
        ]

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_samples
            samples = load_hf_samples("fake/dataset", "validation", limit=3)

        assert len(samples) == 3

    def test_no_limit_returns_all(self):
        from run_omni import load_hf_samples

        mock_samples = [
            {
                "__key__": f"dialog_{i:04d}",
                "opus": b"\x00",
                "soap.txt": "soap",
                "json": json.dumps({"id": f"dialog_{i:04d}"}),
            }
            for i in range(5)
        ]

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_samples
            samples = load_hf_samples("fake/dataset", "validation")

        assert len(samples) == 5


class TestCLIArgParsing:
    """Tests for the updated CLI argument parsing."""

    def test_dataset_and_split_defaults(self):
        from run_omni import build_parser

        parser = build_parser()
        args = parser.parse_args(["--output", "out.jsonl"])
        assert args.dataset == "BeTraC/betrac-2026"
        assert args.split == "validation"

    def test_custom_dataset_and_split(self):
        from run_omni import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--output", "out.jsonl",
            "--dataset", "my/dataset",
            "--split", "train",
        ])
        assert args.dataset == "my/dataset"
        assert args.split == "train"

    def test_manifest_flag(self):
        from run_omni import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--output", "out.jsonl",
            "--manifest", "data/my.csv",
        ])
        assert args.manifest == "data/my.csv"

    def test_limit_parameter(self):
        from run_omni import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--output", "out.jsonl",
            "--limit", "5",
        ])
        assert args.limit == 5

    def test_output_is_required(self):
        from run_omni import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Integration tests (require HuggingFace access)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHFDatasetIntegration:
    """Integration tests that load real data from HuggingFace.

    These tests download the first 5 samples from the validation split of
    BeTraC/betrac-2026. They require network access and HuggingFace
    authentication (if the dataset is gated).

    Run with: pytest -m integration
    """

    def test_load_5_validation_samples(self):
        from run_omni import load_hf_samples

        samples = load_hf_samples("BeTraC/betrac-2026", "validation", limit=5)
        assert len(samples) == 5

    def test_samples_have_audio_bytes(self):
        from run_omni import load_hf_samples

        samples = load_hf_samples("BeTraC/betrac-2026", "validation", limit=1)
        assert len(samples[0]["audio_bytes"]) > 0

    def test_samples_have_soap(self):
        from run_omni import load_hf_samples

        samples = load_hf_samples("BeTraC/betrac-2026", "validation", limit=1)
        assert len(samples[0]["reference_soap"]) > 0

    def test_opus_tempfile_is_valid_audio(self):
        from run_omni import load_hf_samples, write_audio_tempfile

        samples = load_hf_samples("BeTraC/betrac-2026", "validation", limit=1)
        path = write_audio_tempfile(samples[0]["audio_bytes"])
        try:
            assert Path(path).stat().st_size > 0
            # Verify it's a valid Ogg/Opus file (starts with OggS magic bytes)
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == b"OggS", f"Expected OggS header, got {magic!r}"
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
