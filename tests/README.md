# Tests

## Running Tests

```bash
# Unit tests (no GPU or network required)
pytest tests/ -v -m "not integration"

# Integration tests (requires HuggingFace access)
pytest tests/ -v -m integration

# All tests
pytest tests/ -v
```

## Test Files

- `test_run_omni.py` — Tests for manifest reading, prompt loading, resume, output schema
- `test_data_loading.py` — Tests for HuggingFace loading, audio temp files, CLI parsing
- `conftest.py` — Shared fixtures (temp dirs, sample manifests, prompts)

## Smoke Tests (via Makefile)

```bash
make test              # Example manifest, local audio
make test-5            # 5 HuggingFace samples, Qwen2.5-Omni-3B
make test-hf LIMIT=20  # Configurable HuggingFace test
```
