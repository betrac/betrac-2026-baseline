# Validation-set baseline results

All numbers are on the public `BeTraC/betrac-2026` **validation** split (400
dialogs), scored with [betrac-metrics](https://github.com/betrac/betrac-metrics):

```bash
btc-eval evaluate --predictions summaries.jsonl --split validation \
                  --team <name> --bootstrap-ci --output <dir>
```

Brackets are 95 % bootstrap confidence intervals over dialogs.

---

## The audio-truncation bug

Until this fix, the baseline never heard most of each consultation. Both Omni
processor families frame audio with a `WhisperFeatureExtractor` whose `__call__`
defaults to `truncation=True` with `max_length = chunk_length * sampling_rate`,
and `chunk_length` is read from each checkpoint's `preprocessor_config.json`:

| Checkpoint | `chunk_length` | Hard cut |
|------------|----------------|----------|
| `Qwen2.5-Omni-3B`, `Qwen2.5-Omni-7B` | `300` | 300 s |
| `Qwen3-Omni-30B-A3B-Instruct`, `-Thinking` | *absent* → Whisper default `30` | **30 s** |

No warning is emitted — the tensor is simply shorter.

### How much audio that cost

The validation split totals **58.9 h** of audio: mean **530.4 s** (8 m 50 s),
median 504.6 s, range 168.2 s – 1473.9 s.

| Cut | Dialogs affected | Audio the model heard |
|-----|------------------|-----------------------|
| 300 s (Qwen2.5-Omni) | 371 / 400 (92.8 %) | **55.9 %** |
| 30 s (Qwen3-Omni) | 400 / 400 (100 %) | **5.7 %** |
| 1126 s (new Qwen2.5 cap) | 8 / 400 (2.0 %) | 99.2 % |
| 3742–4372 s (new Qwen3 caps) | 0 / 400 (0 %) | 100 % |

### Independent evidence, before any re-run

Per-dialog Concept F1 from the pre-fix runs falls steadily with the fraction of
audio that survived the cut. "Audio heard" is duration-weighted, matching the
55.9 % / 5.7 % headline figures.

**Qwen2.5-Omni-3B** — `corr(fraction heard, recall) = +0.49`

| Dialog length | n | Audio heard | Concept F1 | Recall |
|---------------|---|-------------|-----------|--------|
| 0–300 s   |  29 | 100.0 % | 0.2738 | 0.2776 |
| 300–450 s | 122 |  79.8 % | 0.2958 | 0.2857 |
| 450–600 s | 135 |  57.1 % | 0.2523 | 0.2379 |
| 600–800 s |  80 |  44.1 % | 0.2321 | 0.2065 |
| 800 s+    |  34 |  30.4 % | 0.2208 | 0.1890 |

**Qwen2.5-Omni-7B** — `corr(fraction heard, recall) = +0.42`

| Dialog length | n | Audio heard | Concept F1 | Recall |
|---------------|---|-------------|-----------|--------|
| 0–300 s   |  29 | 100.0 % | 0.2853 | 0.2724 |
| 300–450 s | 122 |  79.8 % | 0.2927 | 0.2622 |
| 450–600 s | 135 |  57.1 % | 0.2425 | 0.2190 |
| 600–800 s |  80 |  44.1 % | 0.2326 | 0.2047 |
| 800 s+    |  34 |  30.4 % | 0.2224 | 0.1839 |

**The trend is not monotonic across the whole range, and the exception matters.**
For both models the never-truncated 0–300 s bucket scores *below* the 300–450 s
bucket (3B: 0.2738 vs 0.2958). From 300 s onward the decline is strictly
monotonic in both F1 and recall, but the shortest dialogs break the pattern.

That is consistent with a competing explanation — very short consultations may
simply be intrinsically harder to summarise, or carry less documentable content
— so this table on its own does **not** establish causality. Only the re-run
does, via the byte-identical control group below. Treat this table as
suggestive, not probative.

Precision is roughly flat across buckets while recall falls, which is the shape
truncation predicts: the model is not getting worse at what it says, it just
never hears the rest of the visit.

---

## Before the fix (archived under `ExpNNNN-*/results-truncated-bug/`)

| Experiment | Model | Heard | Concept F1 | ROUGE-2 F | ROUGE-3 F |
|------------|-------|-------|-----------|-----------|-----------|
| Exp0001 | Qwen2.5-Omni-3B | first 300 s | 0.2604 [0.2544, 0.2662] | 0.0920 [0.0883, 0.0957] | 0.0344 [0.0322, 0.0368] |
| Exp0002 | Qwen2.5-Omni-7B | first 300 s | 0.2572 [0.2502, 0.2638] | 0.0950 [0.0908, 0.0991] | 0.0343 [0.0317, 0.0367] |
| Exp0003 | Qwen3-Omni-30B-A3B-Instruct | first 30 s | 0.1879 [0.1842, 0.1916] | 0.0558 [0.0541, 0.0575] | 0.0175 [0.0166, 0.0185] |
| Exp0004 | Qwen3-Omni-30B-A3B-Thinking | first 30 s | 0.1645 [0.1603, 0.1688] | 0.0472 [0.0454, 0.0489] | 0.0123 [0.0113, 0.0132] |

The 30 s cut explains why the two 30B models scored *below* the 3B one — they
were summarising the opening exchange of each consultation, not the visit.

---

## A second bug, and why the runs are split three ways

An adversarial review of the fix turned up a second silent truncation, this time
on the **output**. `Qwen2_5OmniForConditionalGeneration.generate` declares its own
`thinker_max_new_tokens: int = 1024`, seeds `thinker_kwargs` with it, and merges
caller kwargs with `if key not in thinker_kwargs` — so the `max_new_tokens=4096`
this repo passed was **discarded** and every Qwen2.5 note was cut at 1024 tokens
mid-sentence. Same failure shape as the audio bug: a wrapper default quietly
overriding the caller.

To keep the two effects separable, each Qwen2.5 experiment has three result
directories. **Every number below names the directory it came from.**

| Directory | Audio | Output budget |
|-----------|-------|---------------|
| `results-truncated-bug/` | cut at 300 s | 1024 tokens |
| `results-audiofix-only/` | full | 1024 tokens |
| `results/` (**current, reproducible**) | full | 4096 tokens |

## After both fixes — `results/`

This is what the documented reproduce command produces. All four models, 400
validation dialogs.

| Experiment | Model | Concept F1 | ROUGE-2 F | ROUGE-3 F |
|------------|-------|-----------|-----------|-----------|
| Exp0001 | Qwen2.5-Omni-3B | **0.3203** [0.3133, 0.3271] | **0.1251** | **0.0539** |
| Exp0002 | Qwen2.5-Omni-7B | **0.3278** [0.3202, 0.3350] | **0.1433** | **0.0623** |
| Exp0003 | Qwen3-Omni-30B-A3B-Instruct | **0.3110** [0.3054, 0.3167] | **0.1430** | **0.0615** |
| Exp0004 | Qwen3-Omni-30B-A3B-Thinking | **0.2844** [0.2792, 0.2897] | **0.1334** | **0.0547** |

## What the truncation cost

Paired bootstrap over 400 dialogs, 20 000 resamples: *p* < 0.00001 for every
model.

| Model | Was hearing | Concept F1 | Impact | ROUGE-2 | ROUGE-3 |
|-------|-------------|-----------|--------|---------|---------|
| Qwen2.5-Omni-3B | 300 s | 0.2604 → **0.3203** | **+23.0 %** | +35.9 % | +56.6 % |
| Qwen2.5-Omni-7B | 300 s | 0.2572 → **0.3278** | **+27.4 %** | +50.8 % | +81.8 % |
| Qwen3-Omni-30B-A3B-Instruct | 30 s | 0.1879 → **0.3110** | **+65.5 %** | +156.3 % | +251.4 % |
| Qwen3-Omni-30B-A3B-Thinking | 30 s | 0.1645 → **0.2844** | **+72.9 %** | +182.6 % | +344.7 % |

Precision **and** recall rise for every model — this is not a precision/recall
trade. The two 30B models roughly doubled their recall.

| Model | Precision | Recall |
|-------|-----------|--------|
| Qwen2.5-Omni-3B | 0.2891 → 0.3429 | 0.2450 → 0.3112 |
| Qwen2.5-Omni-7B | 0.3070 → 0.3786 | 0.2302 → 0.3003 |
| Qwen3-Omni-30B-A3B-Instruct | 0.1879 → 0.2558 | 0.1964 → 0.4101 |
| Qwen3-Omni-30B-A3B-Thinking | 0.1694 → 0.2347 | 0.1675 → 0.3714 |

### The published ranking was wrong, not just the numbers

| | Before (buggy) | After (fixed) |
|-|----------------|---------------|
| 1 | Qwen2.5-Omni-3B — 0.2604 | Qwen2.5-Omni-7B — 0.3278 |
| 2 | Qwen2.5-Omni-7B — 0.2572 | Qwen2.5-Omni-3B — 0.3203 |
| 3 | Qwen3-30B-Instruct — 0.1879 | Qwen3-30B-Instruct — 0.3110 |
| 4 | Qwen3-30B-Thinking — 0.1645 | Qwen3-30B-Thinking — 0.2844 |

The baseline told participants the 30B models scored far below a 3B. Almost all
of that gap was the bug hitting the two families unequally — 30 s of audio
versus 300 s. Corrected, all four sit within a 0.28–0.33 band. Teams that ruled
out the 30B models on the published numbers were reasoning from an artefact.

### Known limitations of these numbers

- **Long-dialog degradation: resolved.** On the 40 GB A100s, 8–9 dialogs per
  30B run initially ran at ~60 % audio (OOM retry ladder) and one failed
  outright. All were re-run at **full audio on a single 93 GB H100** and
  merged; both 30B results above are **100 % audio coverage, 0 failures**.
  On 40 GB hardware alone, expect the retry ladder to shorten the longest
  ~2 % of dialogs.
- **The Qwen3 numbers are SDPA-path — measured to be benign.** The Qwen3 audio
  encoder only honours its `cu_seqlens` windows under FlashAttention-2; on the
  default SDPA path the tower attends globally instead of in ~400-token
  windows, which is not the configuration the checkpoint was trained in. We
  quantified this on a paired 100-dialog subset (identical audio, Instruct
  model): FA2 scored Concept F1 0.2966 vs SDPA 0.3041 — paired delta
  **−0.0076** [−0.0147, −0.0005], i.e. the windowed FA2 path is, if anything,
  *marginally worse*, and the difference is small. (Measured as FA2-on-H100 vs
  SDPA-on-A100, so bf16 hardware numerics are folded into the delta; greedy
  decoding amplifies per-dialog variance either way.) Conclusion: the default
  SDPA configuration does not understate the model, and the headline numbers
  stand as the reproducible default.

## Which bug cost what (Qwen2.5 only)

The output-cap bug was found later, so the two Qwen2.5 experiments have all
three run stages and the effects can be separated. Concept F1:

| Step | 3B | 7B |
|------|-----|-----|
| `results-truncated-bug/` | 0.2604 | 0.2572 |
| `results-audiofix-only/` | 0.3203 | 0.3278 |
| `results/` | 0.3203 | 0.3278 |

| Attribution | 3B | 7B |
|-------------|-----|-----|
| **Audio truncation fix** | **+23.0 %** | **+27.4 %** |
| Output-cap fix | +0.0 % | −0.0 % |

**The audio truncation accounts for the entire measurable effect.** Lifting the
1024-token output cap changed exactly 39 of 400 summaries for the 3B and 11 for
the 7B — precisely the notes being clipped — and moved Concept F1 by a paired
delta of +0.00004 (95 % CI [−0.00009, +0.00023]) for the 3B. ROUGE-2 drops
slightly (0.1285 → 0.1251) because the recovered text lengthens notes 41 %
without adding matched concepts.

The output cap is still a real bug worth fixing — a baseline whose notes stop
mid-sentence is a poor example to build on — it simply is not what moved the
score.

### The change is causally the audio

Measured on `results-audiofix-only/` vs `results-truncated-bug/`, which isolates
the audio variable with the output budget held fixed at 1024 on both sides.

The 29 validation dialogs shorter than 300 s were never truncated, so the fix
cannot have changed them — and it did not. For **both** models:

- 29 / 29 never-truncated dialogs → **byte-identical** summaries before and after
- 371 / 371 truncated dialogs → all changed

Decoding is deterministic, so this rules out seed drift, library skew, or a
metric change. The only variable that moved is how much audio the model heard.

### Dose–response

Also measured on `results-audiofix-only/` vs `results-truncated-bug/`.
All "audio heard" figures below are duration-weighted, the same basis as the
table in the previous section, so the two are directly comparable.

| Dialog length | n | Audio heard before | 3B F1 | 7B F1 |
|---------------|---|--------------------|-------|-------|
| 0–300 s   |  29 | 100.0 % | 0.2738 → 0.2738 (**+0.0 %**) | 0.2853 → 0.2853 (**+0.0 %**) |
| 300–450 s | 122 |  79.8 % | 0.2958 → 0.3267 (+10.5 %) | 0.2927 → 0.3347 (+14.3 %) |
| 450–600 s | 135 |  57.1 % | 0.2523 → 0.3302 (+30.9 %) | 0.2425 → 0.3339 (+37.7 %) |
| 600–800 s |  80 |  44.1 % | 0.2321 → 0.3178 (+36.9 %) | 0.2326 → 0.3350 (+44.0 %) |
| 800 s+    |  34 |  30.4 % | 0.2208 → 0.3031 (+37.3 %) | 0.2224 → 0.2982 (+34.1 %) |

Before the fix, score fell steadily with dialog length from 300 s onward. After
it, both models sit near a flat 0.30–0.335 regardless of length — the
length-dependent penalty is gone.

Two caveats on the tails. The 7B's 800 s+ bucket gains less than the 600–800 s
bucket (+34.1 % vs +44.0 %); the 3B's does **not** show that pattern (+37.3 %,
its largest gain), so this is not explained by the 8 dialogs that meet the
1126 s cap — that would affect both models equally. The cause is unresolved.
The 0–300 s bucket is unchanged by construction: nothing was ever cut there.

### Cost

| Model | GPU-h `results-truncated-bug/` | GPU-h `results-audiofix-only/` | Audio coverage | Dialogs capped |
|-------|-------------------------------|-------------------------------|----------------|----------------|
| Qwen2.5-Omni-3B | 4.3 | 4.9 (1.14×) | 99.2 % | 8 / 400 |
| Qwen2.5-Omni-7B | 3.8 | 4.5 (1.19×) | 99.2 % | 8 / 400 |

Hearing ~1.8× more audio costs under 20 % more compute — model loading and
generation dominate, not audio prefill.

---

## Reproducing

```bash
# 1. confirm the bug and the fix on the real processors (no model weights)
python scripts/check_audio_truncation.py

# 2. re-run the validation split (SLURM)
cd experiments/Exp0001-qwen25-3b
TOTAL=400 bash slurm/submit_omni.sh --time=01:00:00 --mem=64G

# 3. score
btc-eval evaluate --predictions results/validation/qwen25-omni-3b/summaries.jsonl \
                  --split validation --team mine --bootstrap-ci
```

Every record carries `audio_sec` and `audio_used_sec`; if they differ for more
than a couple of dialogs, audio is being dropped.
