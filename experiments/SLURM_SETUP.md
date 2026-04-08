# Adapting BeTraC SLURM Scripts to Your Cluster

This guide walks through how the generic SLURM scripts in `scripts/` were
adapted into the per-experiment directories under `experiments/` for the
OSC Ascend cluster. Use it as a template for setting up your own cluster.

## TL;DR — Quick migration checklist

To run on your own SLURM cluster, edit each experiment's `slurm/run_omni_hf.slurm`
**and** `slurm/run_combine_omni.slurm` (look for lines marked `← CHANGE`):

1. **Change cluster/account headers** — replace `--cluster=ascend` and
   `--account=PAS2138` with your cluster's partition and account
2. **Update module loading** (in `run_omni_hf.slurm` only) — replace the
   `module load` lines with your environment setup
3. **Pre-cache models** — run `make setup` then pre-download the model
   before submitting jobs (see [HuggingFace caching](#4-set-up-huggingface-caching))

Then submit: `cd experiments/Exp0001-qwen25-3b && TOTAL=400 bash slurm/submit_omni.sh`

## What the generic scripts provide

The repository ships with cluster-agnostic SLURM scripts in `scripts/`:

| File | Purpose |
|------|---------|
| `scripts/run_omni.slurm` | Array job template (manifest-based) |
| `scripts/run_combine.slurm` | Combine per-task JSONL outputs |
| `scripts/submit_slurm.sh` | Orchestrator: calculates array size, submits both jobs |
| `scripts/run_local.sh` | Non-SLURM local runner |

These work with any SLURM cluster but require you to:
- Add your cluster/account/partition headers
- Set up module loading for your environment
- Configure GPU and memory allocations for your hardware
- Handle HuggingFace caching for parallel jobs

## What the experiment directories add

Each `experiments/ExpNNNN-<name>/` directory contains cluster-specific scripts
with hardcoded resource allocations and model defaults:

| File | Based on | Key changes |
|------|----------|-------------|
| `slurm/run_omni_hf.slurm` | `scripts/run_omni.slurm` | HF dataset mode (--offset/--limit), cluster headers, module loading, HF_HUB_OFFLINE=1 |
| `slurm/run_combine_omni.slurm` | `scripts/run_combine.slurm` | Cluster headers, deduplication by sample ID |
| `slurm/submit_omni.sh` | `scripts/submit_slurm.sh` | Pre-caches dataset+model, experiment-specific defaults |
| `run_local.sh` | `scripts/run_local.sh` | HF dataset mode, experiment-specific defaults |

## Step-by-step: Creating experiment directories for your cluster

### 1. Choose resource allocations per model

The 4 models have very different resource needs. Start with these estimates
and adjust based on your GPU type:

| Model | Min GPU RAM | GPUs (SBATCH) | Memory | Time/sample | Walltime (4 samples) |
|-------|-------------|---------------|--------|-------------|---------------------|
| Qwen2.5-Omni-3B | ~8 GB | 1 | 32 GB | ~45s | ~5 min |
| Qwen2.5-Omni-7B | ~16 GB | 1 | 80 GB | ~90s | ~8 min |
| Qwen3-Omni-30B-Instruct | ~34 GB | 2* | 80 GB | ~414s (~7 min) | ~30 min |
| Qwen3-Omni-30B-Thinking | ~34 GB | 2* | 80 GB | ~1275s (~21 min) | ~90 min |

*\*The 30B models request 2 GPUs in SBATCH but each subprocess is restricted to
a single GPU via `CUDA_VISIBLE_DEVICES=0`. The model uses `device_map=auto` which
offloads excess layers to CPU (not tensor parallelism across GPUs). This prevents
OOM on 40GB GPUs — the model uses ~34 GB on one GPU with ~6 GB free for inference.
The second GPU is reserved to prevent other jobs from co-scheduling on the node.
If your cluster has 80GB GPUs (e.g. A100-SXM4-80GB), you can change to
`--gpus-per-task=1` — the model will fit on a single 80GB GPU without CPU offload.*

**Tips:**
- Keep walltime under your cluster's preferential scheduling threshold (often 1 hour)
- Adjust `SAMPLES_PER_TASK` to control walltime: fewer samples = shorter jobs
- For Thinking models, use `SAMPLES_PER_TASK=2-4` to avoid timeouts (some samples take 40+ min)
- Request more memory than the model needs — the subprocess-per-sample design has overhead

### 2. Set cluster-specific SBATCH headers

Replace the generic `#SBATCH` lines with your cluster's requirements.

**Generic** (`scripts/run_omni.slurm`):
```bash
#SBATCH --job-name=betrac-omni
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x_%A_%a.out
#SBATCH --error=logs/slurm-%x_%A_%a.err
```

**OSC Ascend** (what we use):
```bash
#SBATCH --job-name=Exp0001-omni-qwen25-3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --cluster=ascend                    # OSC-specific
#SBATCH --account=PAS2138                   # OSC-specific
#SBATCH --output=slurm/logs-slurm/slurm-%x_%A_%a.out
#SBATCH --error=slurm/logs-slurm/slurm-%x_%A_%a.err
```

**Common cluster variations:**

| Cluster type | Instead of `--cluster` | Notes |
|-------------|------------------------|-------|
| Single-cluster SLURM | `--partition=gpu` | Most university clusters |
| AWS ParallelCluster | `--partition=gpu` | Plus `--constraint=` for instance type |
| GCP HPC Toolkit | `--partition=gpu` | Similar to standard SLURM |
| Multi-cluster (like OSC) | `--cluster=name` | Each cluster has own scheduler |
| PBS/Torque | N/A | Different syntax entirely — see below |

### 3. Configure module loading

The generic scripts comment out module loading. Uncomment and adapt for your cluster.

**OSC Ascend:**
```bash
module purge
module load python/3.10
module load cuda/12.4.1
```

**Common alternatives:**
```bash
# Conda-based cluster
source activate betrac

# Container-based (Singularity/Apptainer)
module load singularity
singularity exec --nv /path/to/container.sif bash -c "..."

# No module system (e.g. cloud VMs)
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Spack-based
spack load python@3.11 cuda@12.4
```

### 4. Set up HuggingFace caching

This is critical for parallel jobs. Without caching, N simultaneous tasks
all hit the HuggingFace API, causing rate limit errors (HTTP 429).

**Solution: pre-cache on the login node, run offline on compute nodes.**

The `submit_omni.sh` scripts do this automatically:
```bash
# Before submitting jobs (runs on login node with internet):
python -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
load_dataset('BeTraC/betrac-2026', split='validation')  # cache dataset
snapshot_download('Qwen/Qwen2.5-Omni-3B')               # cache model
"
```

The SLURM scripts set `HF_HUB_OFFLINE=1` so compute nodes never contact HuggingFace:
```bash
export HF_HUB_OFFLINE=1
```

**Disk space requirements:** Model weights are cached in `~/.cache/huggingface/`
by default. The 30B models are ~60 GB each; all 4 models together need ~100 GB.
If your home directory has a small quota, move the cache to scratch storage:

```bash
# Move cache to scratch (do this BEFORE running make setup or pre-caching)
mkdir -p /scratch/YOUR_PROJECT/$USER/huggingface_cache
ln -s /scratch/YOUR_PROJECT/$USER/huggingface_cache ~/.cache/huggingface

# Or set HF_HOME in your shell profile and SLURM scripts
export HF_HOME=/scratch/YOUR_PROJECT/$USER/huggingface_cache
```

**Shared cache for multi-user clusters:**
```bash
export HF_HOME=/shared/scratch/project/hf_cache
```

### 5. Choose SAMPLES_PER_TASK

This controls the tradeoff between parallelism and per-task walltime:

| SAMPLES_PER_TASK | Tasks (400 samples) | Walltime (3B) | Walltime (30B Instruct) | Walltime (30B Thinking) |
|-----------------|---------------------|---------------|-------------------------|-------------------------|
| 2 | 200 | ~3 min | ~15 min | ~45 min |
| 4 | 100 | ~5 min | ~30 min | ~90 min |
| 10 | 40 | ~10 min | ~70 min | ~210 min |
| 20 | 20 | ~18 min | ~140 min | ~420 min |

*Walltimes are estimates based on measured median per-sample times. Worst-case
samples can be 2-3x the median. The Thinking model is ~3x slower than Instruct
due to chain-of-thought generation.*

**Recommendations:**
- For 3B/7B models: `SAMPLES_PER_TASK=20` (good balance, fits under 30 min)
- For 30B Instruct: `SAMPLES_PER_TASK=4` (fits under 1h with headroom for slow samples)
- For 30B Thinking: `SAMPLES_PER_TASK=2-4` (some samples take 40+ min; a few may still timeout)
- More tasks = more scheduler overhead but better fault tolerance
- The `--resume` flag means failed/timed-out tasks only need to reprocess their missing samples

### 6. Data source: HF dataset vs manifest

The generic scripts use **manifest mode** (CSV files with audio paths):
```bash
# Generic: uses --manifest with manifest slicing
OMNI_ARGS=(--manifest "${TEMP_MANIFEST}" --output "${TASK_JSONL}" ...)
```

The experiment scripts use **HF dataset mode** (streaming with offset):
```bash
# Experiment: uses --dataset with --offset/--limit
OMNI_ARGS=(--dataset "${DATASET}" --split "${SPLIT}" --offset "${OFFSET}" --limit "${SAMPLES_PER_TASK}" ...)
```

If your data is in a local CSV manifest rather than on HuggingFace, use the
generic scripts or adapt the experiment scripts to use manifest slicing
(see `scripts/run_omni.slurm` lines 90-106 for the slicing logic).

## Key differences: generic vs experiment scripts

| Aspect | Generic (`scripts/`) | Experiment (`experiments/Exp*/`) |
|--------|---------------------|--------------------------------|
| Data source | Manifest (CSV) | HF dataset (streaming) |
| Partitioning | Manifest line slicing | `--offset` / `--limit` |
| Cluster config | Commented out, user adapts | Hardcoded for target cluster |
| Module loading | Commented out | Configured for cluster |
| HF caching | Not handled | Pre-cache + offline mode |
| Model defaults | Via env vars | Hardcoded per experiment |
| Output path | `results/<model>/` | `results/<split>/<model>/` |
| Combine | Simple cat | Cat + deduplicate by ID |
| Job naming | Generic `betrac-omni` | `ExpNNNN-omni-<model>` |

## Lessons learned from OSC deployment

### HuggingFace rate limits (HTTP 429)
With 100 parallel tasks, each subprocess checking model metadata on HuggingFace
simultaneously caused widespread rate limiting. **Fix:** pre-cache everything on
the login node and set `HF_HUB_OFFLINE=1` on compute nodes.

### NFS stale file handles (errno 116)
Parallel tasks reading from `~/.cache/huggingface/` on a shared NFS filesystem
occasionally hit stale file handles. This is a transient NFS issue, not a code bug.
The `--resume` flag handles this gracefully — just resubmit.

### Preferential scheduling under 1 hour
Many clusters (including OSC) give priority to jobs requesting less than 1 hour
of walltime. Keeping `SAMPLES_PER_TASK` low enough to finish within 1 hour
significantly reduces queue wait times.

### 30B models: single-GPU subprocess with CPU offload
The 30B MoE model loads ~34 GB and needs ~6 GB free for inference KV cache.
On 2x40GB GPUs, `device_map=auto` splits the model across both GPUs, leaving
only ~6 GB free on each — not enough for inference, causing OOM. The fix is
to restrict each subprocess to a single GPU (`CUDA_VISIBLE_DEVICES=0`), forcing
`device_map=auto` to offload excess layers to CPU. This is handled automatically
in `run_omni.py`. The SBATCH still requests 2 GPUs to reserve the full node.

### `/tmp` space on compute nodes
The inference script writes audio to temporary files in `/tmp`. If the compute
node's `/tmp` is small or shared with other jobs, this can fail with
`[Errno 28] No space left on device`. Fix by setting `TMPDIR` in the SLURM
script to point to scratch storage:
```bash
export TMPDIR=/fs/scratch/YOUR_PROJECT/$USER/tmp
mkdir -p "$TMPDIR"
```

### Duplicate results on retry with different SAMPLES_PER_TASK
If you rerun with a different `SAMPLES_PER_TASK`, old task files from the previous
run may still exist with different sample-to-task mappings. The combine script
deduplicates by sample ID to handle this. However, it's cleaner to move old
results aside before rerunning (e.g. `mv results/validation/model results/validation/model.backup`).

## Recovering from failed or timed-out tasks

### Check which tasks failed

```bash
# Show all non-completed tasks
sacct -j <ARRAY_JOB_ID> --format=JobID,State -n | grep -v "\." | grep -v COMPLETED

# Count by state
sacct -j <ARRAY_JOB_ID> --format=State -n | sort | uniq -c
```

### Resubmit specific tasks

```bash
cd experiments/Exp0004-qwen3-thinking/

# Resubmit tasks 42, 78, 91 (--resume skips already-completed samples)
sbatch --array=42,78,91 --time=02:00:00 \
  --export=ALL,MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking",MODEL_SHORT="qwen3-omni-30b-thinking",DATASET="BeTraC/betrac-2026",SPLIT="validation",FLASH_ATTN=0,THINKING=1,SAMPLES_PER_TASK=4 \
  slurm/run_omni_hf.slurm
```

### Handle failed records in output files

If a task wrote a failed record (e.g., `"success": false`) to the output
JSONL, `--resume` will skip it (it matches by ID, not by success status).
Remove failed records before resubmitting:

```bash
python3 -c "
import json
f = 'results/validation/MODEL_SHORT/summaries_task_66.jsonl'
lines = open(f).readlines()
good = [l for l in lines if json.loads(l).get('success', True)]
open(f, 'w').writelines(good)
print(f'Kept {len(good)}/{len(lines)} records')
"
```

### Combine results manually

If the SLURM combine job is stuck in the queue, combine manually:

```bash
python3 -c "
import json, glob
seen, records = set(), []
for f in sorted(glob.glob('results/validation/MODEL_SHORT/summaries_task_*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r['id'] not in seen:
            seen.add(r['id'])
            records.append(line)
open('results/validation/MODEL_SHORT/summaries.jsonl', 'w').writelines(records)
print(f'Combined {len(records)} records')
"
```

## Verifying results

After all tasks complete, verify you have a full, clean result set:

```bash
cd experiments/Exp0001-qwen25-3b/
python3 -c "
import json
total = failed = empty = 0
for line in open('results/validation/qwen25-omni-3b/summaries.jsonl'):
    r = json.loads(line); total += 1
    if not r.get('success', True): failed += 1
    if not r.get('summary','').strip(): empty += 1
print(f'Total: {total}, Failed: {failed}, Empty: {empty}')
# Expected: Total: 400, Failed: 0, Empty: 0
"
```

If you see failed or empty records, check the per-task files and SLURM logs
to identify the cause, then follow the recovery steps above.

## PBS/Torque equivalent

For clusters using PBS/Torque instead of SLURM, the key translations are:

| SLURM | PBS/Torque |
|-------|-----------|
| `#SBATCH --job-name=X` | `#PBS -N X` |
| `#SBATCH --nodes=1` | `#PBS -l nodes=1` |
| `#SBATCH --gpus-per-task=2` | `#PBS -l ngpus=2` |
| `#SBATCH --mem=80G` | `#PBS -l mem=80gb` |
| `#SBATCH --time=01:00:00` | `#PBS -l walltime=01:00:00` |
| `#SBATCH --account=X` | `#PBS -A X` |
| `#SBATCH --array=0-19` | `#PBS -J 0-19` |
| `$SLURM_ARRAY_TASK_ID` | `$PBS_ARRAY_INDEX` |
| `$SLURM_SUBMIT_DIR` | `$PBS_O_WORKDIR` |
| `sbatch script.slurm` | `qsub script.pbs` |
| `squeue -u $USER` | `qstat -u $USER` |
| `--dependency=afterok:ID` | `-W depend=afterok:ID` |
