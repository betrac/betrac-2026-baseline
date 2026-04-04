# Troubleshooting Guide

Common issues when running the BeTraC 2026 baseline.

---

## Setup

### `uv` not found

**Symptom:** `make: uv: No such file or directory`

```bash
# Install uv
pip install uv
# or on macOS
brew install uv
```

### `make setup` fails with Python version error

**Symptom:** `requires-python >=3.11,<3.12` error

```bash
python3 --version  # Should show 3.11.x
```

If you have multiple Python versions, point uv to the right one:
```bash
cd steps/omni && uv venv --python python3.11 .venv && uv pip install -e .
```

### Virtual Environment Not Found

**Symptom:** `ERROR: Omni venv not found`

```bash
make setup   # Creates steps/omni/.venv
```

---

## Model Loading

### Model Download Stalls in Subprocess

**Symptom:** `Fetching N files: 0%` hangs indefinitely during `make test-5` or `make run`

**Cause:** HuggingFace `snapshot_download` can deadlock inside `multiprocessing.spawn` subprocesses.

**Fix:** Pre-download the model in the main process before running inference:
```bash
steps/omni/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Omni-3B')  # change to your model
"
```

If `snapshot_download` also stalls, download model files manually with `curl`. Example for `Qwen/Qwen2.5-Omni-7B` (adjust URL and shard count for other models):
```bash
mkdir -p qwen2.5-omni-7b
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main"
# Download safetensor shards (check model.safetensors.index.json for shard count)
for i in 1 2 3 4 5; do
  curl -L -C - -o "qwen2.5-omni-7b/model-0000${i}-of-00005.safetensors" \
    "${MODEL_URL}/model-0000${i}-of-00005.safetensors"
done
# Download config and tokenizer files
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
         preprocessor_config.json generation_config.json special_tokens_map.json \
         added_tokens.json chat_template.json vocab.json merges.txt spk_dict.pt; do
  curl -L -o "qwen2.5-omni-7b/$f" "${MODEL_URL}/$f"
done
# Run with local path (directory name must contain model family for auto-detection)
make test-5 TEST_MODEL=./qwen2.5-omni-7b
```

### Wrong Model Classes

**Symptom:** `You are using a model of type qwen2_5_omni to instantiate a model of type qwen3_omni_moe`

**Cause:** Model family auto-detection uses the model name. If using a local path without "qwen3" or "qwen2.5" in it, detection fails.

**Fix:** Rename your local model directory to include the model family name, e.g. `qwen2.5-omni-7b/`.

---

## GPU and Memory

### MPS Memory Explosion (Apple Silicon)

**Symptom:** Disk fills with `mpsgraph` temp files, >128GB memory usage, `No result returned from subprocess`

**Cause:** Dense models (e.g. Qwen2.5-Omni-7B) exceed MPS shared memory. MPS silently swaps to disk via macOS unified memory.

**Fix:** The model config registry auto-routes known large dense models to CPU on MPS. To force CPU for any model:
```bash
make test-5 TEST_MODEL=Qwen/Qwen2.5-Omni-7B DEVICE=cpu
```

To set a memory limit instead of forcing CPU (causes a clear `RuntimeError` instead of silent disk swap):
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5  # cap at 50% of RAM; 0.0 = OS-managed (default)
```
If you get `invalid low watermark ratio`, use `0.0` instead (disables the hard cap, lets OS manage).

**Why the 30B model works on MPS:** Qwen3-Omni-30B is a Mixture-of-Experts (MoE) model with only ~3B active parameters per token — its memory footprint during inference is much smaller than the 7B dense model.

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

Options:
- Use a smaller model (`Qwen/Qwen2.5-Omni-3B`)
- Enable Flash Attention 2: add `--use-flash-attn2` (requires A100 or newer)
- Use vLLM backend: add `--use-vllm` (better memory management for multi-GPU)

### No Result Returned from Subprocess

**Symptom:** `Failed: No result returned from subprocess` with no other error

**Cause:** The subprocess crashed (OOM, segfault) before putting a result on the queue.

**Diagnosis:**
1. Check disk space: `df -h /` (MPS memory issues can fill the disk)
2. Try with `DEVICE=cpu` to rule out GPU memory issues
3. Try with a smaller model (`Qwen/Qwen2.5-Omni-3B`)
4. Check if the model downloaded completely (see "Model Download Stalls" above)

---

## Audio Processing

### Opus Decoding Fails

**Symptom:** Error loading `.opus` audio files

**Fix:** Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

---

## SLURM

### Module Load Fails

**Symptom:** `module: command not found` or `No module named python/3.11`

**Fix:** Edit `scripts/run_omni.slurm` — uncomment and adapt the module load section for your cluster:
```bash
# module purge
# module load python/3.11
# module load cuda/12.4
```

### Array Job Fails but Combine Still Runs

**Symptom:** Combine job says `MISSING: summaries_task_N.jsonl`

**Cause:** The combine job uses `--dependency=afterok` — it should only run if all array tasks succeed. If you see this, a task likely timed out or was cancelled.

**Fix:** Check failed task logs in `logs/slurm-betrac-omni_<JOB>_<TASK>.err`

---

## Data Loading

### HuggingFace Dataset Not Found

**Symptom:** `DatasetNotFoundError` or `FileNotFoundError`

```bash
# Verify dataset access
steps/omni/.venv/bin/python -c "
from datasets import load_dataset
ds = load_dataset('BeTraC/betrac-2026', split='validation', streaming=True)
print(next(iter(ds)).keys())
"
```

### Manifest Audio Files Missing

**Symptom:** `ERROR: N audio file(s) missing. Aborting.`

```bash
# Validate your manifest
python scripts/validate_manifest.py data/manifests/my_data.csv
```

Manifest paths are resolved relative to `--base-dir` (defaults to the manifest's parent directory). Use absolute paths if unsure.

---

## Performance

### Inference is slow

Each sample runs in a separate subprocess that loads the model from scratch. This is by design (GPU memory isolation), but it means:

- **First-time overhead:** ~10–45s per sample for model loading (cached after first download)
- **MPS (Apple Silicon):** ~70–570s per sample depending on model size
- **CPU:** Slower but more stable for large dense models

To speed up inference:
- Use a GPU with CUDA (`--device auto` detects CUDA automatically)
- Enable Flash Attention 2: `--use-flash-attn2` (A100 or newer)
- Use vLLM backend: `--use-vllm` (better throughput on multi-GPU)
- Use the 3B model for development, 30B for final results

### Resuming interrupted runs

If a long run is interrupted (Ctrl+C, OOM, timeout), re-run the same command with `--resume`:

```bash
# The Makefile run targets include --resume by default
make run SPLIT=validation

# Or explicitly with run_omni.py
steps/omni/.venv/bin/python steps/omni/run_omni.py \
    --dataset BeTraC/betrac-2026 --split validation \
    --output results/omni_output.jsonl --resume
```

This skips sample IDs already present in the output JSONL file.
