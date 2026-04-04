# BeTraC — Beyond Transcription Challenge

**IEEE SLT 2026 Grand Challenge** | December 13–16, 2026 | Palermo, Sicily, Italy

## Overview

The Beyond Transcription Challenge tests the limits of long-form audio understanding by requiring models to generate clinical SOAP notes directly from doctor–patient conversation audio — without intermediate transcription at any pipeline stage.

## Challenge Statement

**Can models perform simultaneous speaker diarization, noise filtering, and cross-modal reasoning to extract structured professional summaries from raw multi-speaker audio?**

## Motivation

Current end-to-end (E2E) speech models excel at transcription but struggle with higher-order cognitive tasks. This challenge pushes beyond simple speech-to-text to test true audio understanding and information extraction capabilities.

## Dataset

- **Name:** Synth-DoPaCo (hosted as [BeTraC/betrac-2026](https://huggingface.co/datasets/betrac) on HuggingFace)
- **Size:** 8,800 high-fidelity multi-speaker dialogues
- **Domain:** Clinical doctor–patient conversations
- **Audio:** Opus-compressed, 16 kHz mono, ~9 minutes average
- **Task:** Map raw audio → structured SOAP notes
- **Splits:** train (7,200), validation (400), test (1,200)

## Tracks

### Lightweight Track
- No tool use or agentic pipelines
- Baseline: Qwen2.5-Omni-3B

### Heavyweight Track
- Tool use and agentic architectures permitted
- MoE models: total (not active) parameter count applies
- Baseline: Qwen3-Omni-30B-A3B-Instruct

### Rules (both tracks)
- **Open-weight models only** — no proprietary APIs
- **No intermediate transcription** at any pipeline stage — including chain-of-thought steps or tool outputs
- Each audio file processed independently (no cross-file context)
- A cascaded reference topline (Whisper + LLM) is provided for comparison but is **not eligible** for ranking

## Evaluation

- **Primary:** Open Medical Concept F1 (MeSH keyword matching + NER via scispaCy)
- **Secondary:** ROUGE F1 (R-2, R-3, R-L against reference notes)
- **Post-competition:** Top 5 teams per track undergo LLM-as-judge evaluation (Faithfulness, Coverage, Structure, Conciseness) plus out-of-domain testing on real OSCE interviews

## Timeline

- **Challenge Launch / Dataset Release:** April 2, 2026
- **Model/Dataset Proposal Deadline:** May 4, 2026
- **Submission Deadline:** June 24, 2026
- **Results Announcement:** July 1, 2026
- **Paper Submission:** July 8, 2026
- **Workshop:** IEEE SLT 2026 (December 13–16, 2026, Palermo, Italy)

## Baseline System

This repository provides the end-to-end baseline:

```
Audio Input (Opus) → Qwen Omni Model → SOAP Note (plain text)
                         ↓
              No intermediate transcript
              No separate diarization
              No explicit ASR step
```

- **Models:** Qwen2.5-Omni-3B (lightweight) and Qwen3-Omni-30B (heavyweight)
- **Framework:** argparse CLI with YAML prompt templates
- **Output:** Plain-text SOAP notes (Subjective, Objective, Assessment, Plan)

## Submission Format

- Plain-text SOAP notes matching training data format
- System description paper required (due July 8, 2026)

## Resources

- **Baseline implementation:** This repository
- **Cascade topline:** [betrac-2026-cascade](https://github.com/betrac/betrac-2026-cascade)
- **Dataset:** [BeTraC/betrac-2026](https://huggingface.co/datasets/betrac) on HuggingFace
- **Evaluation scripts:** [betrac-metrics](https://github.com/betrac/betrac-metrics)
- **Leaderboard:** [betrac.github.io](https://betrac.github.io)
- **Discussion forum:** [BeTraC Zulip](https://betrac.zulipchat.com/join/4znrm2sqi7f2ng5hdcyi63lh/)
- **Contact:** betrac@googlegroups.com

## FAQ

### Can I use proprietary APIs like GPT-4 or Gemini?
No. Both tracks require open-weight models.

### Can I use an ASR system followed by an LLM?
No. Cascaded architectures are explicitly prohibited. The no-transcription rule applies to all pipeline stages, including chain-of-thought and tool outputs. A cascade topline is provided as a reference only.

### What audio preprocessing is allowed?
Standard preprocessing (normalization, resampling) is allowed. Manual editing, speaker separation, or noise removal requiring human intervention is not.

### Can I fine-tune the baseline model?
Yes. Fine-tuning, continued training, and architectural modifications are all encouraged.

### Can I use tool calling or agents?
Lightweight track: no. Heavyweight track: yes, but no tool may produce intermediate transcriptions.

### What about medical domain expertise?
Domain-specific fine-tuning and prompt engineering are both valid approaches.

## Challenge Organizers

- Andrew Perrault — The Ohio State University
- Jiyun (Amy) Chun — The Ohio State University
- Samuele Cornell — Carnegie Mellon University
- Siddhant Arora — Carnegie Mellon University
- Syed-Amad Hussain — The Ohio State University / Nationwide Children's Hospital
- Thomas Schaaf — Solventum / Carnegie Mellon University
- Markus Müller — Amazon
- Leibny Paola Garcia — Johns Hopkins University (CLSP)
- Ahmed Hassoon (MD, MPH) — Johns Hopkins University

**Questions:** betrac@googlegroups.com
**Updates:** [BeTraC Zulip](https://betrac.zulipchat.com/join/4znrm2sqi7f2ng5hdcyi63lh/)

---

**[IEEE SLT 2026](https://attend.ieee.org/slt-2026/) Grand Challenge** | [betrac.github.io](https://betrac.github.io)
