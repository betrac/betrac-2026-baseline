"""Pytest configuration and shared fixtures."""

import csv
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires network access and HuggingFace auth"
    )


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_manifest_csv(temp_dir):
    """Create a sample CSV manifest with dummy audio references."""
    manifest = temp_dir / "test_manifest.csv"
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"\x00" * 100)

    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "audio_path"]
        )
        writer.writeheader()
        writer.writerow(
            {"id": "test_001", "audio_path": str(audio_file)}
        )
        writer.writerow(
            {"id": "test_002", "audio_path": str(audio_file)}
        )
    return manifest


@pytest.fixture
def sample_manifest_relative(temp_dir):
    """Create a manifest with relative audio paths."""
    audio_dir = temp_dir / "audio"
    audio_dir.mkdir()
    audio_file = audio_dir / "sample.wav"
    audio_file.write_bytes(b"\x00" * 100)

    manifest = temp_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "audio_path"])
        writer.writeheader()
        writer.writerow({"id": "rel_001", "audio_path": "audio/sample.wav"})
    return manifest


@pytest.fixture
def sample_prompt_yaml(temp_dir):
    """Create a sample prompt YAML file."""
    prompt_file = temp_dir / "prompt.yaml"
    prompt_file.write_text("text: |\n  Test prompt for unit testing.\n")
    return prompt_file
