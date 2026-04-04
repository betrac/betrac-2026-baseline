# Qwen3 Omni Baseline – Direct audio-to-summary pipeline
# Uses a single omni model step in an isolated uv-managed Python environment.

SHELL := /bin/bash
ROOT  := $(shell pwd)

# --- Environment paths -------------------------------------------------
OMNI_DIR  := $(ROOT)/steps/omni
OMNI_VENV := $(OMNI_DIR)/.venv

# --- Default data source ------------------------------------------------
DATASET      ?= BeTraC/betrac-2026
SPLIT        ?= validation
LIMIT        ?=

# --- Default output paths ------------------------------------------------
RESULTS_DIR  ?= $(ROOT)/results
OUTPUT       ?= $(RESULTS_DIR)/omni_output.jsonl

# --- Model defaults ----------------------------------------------------
MODEL          ?= Qwen/Qwen3-Omni-30B-A3B-Instruct
THINKING_MODEL ?= Qwen/Qwen3-Omni-30B-A3B-Thinking
DEVICE         ?= auto
SEED           ?= 42

# --- Manifest fallback (for custom data) ----------------------------------
MANIFEST  ?=

# =======================================================================
# Environment setup
# =======================================================================

.PHONY: setup run run-manifest test test-5 test-hf \
        clean help lint lint-fix format check

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

setup: $(OMNI_VENV)/.sentinel  ## Create virtual environment

$(OMNI_VENV)/.sentinel: $(OMNI_DIR)/pyproject.toml
	@echo "==> Setting up omni environment in $(OMNI_VENV)"
	cd $(OMNI_DIR) && uv venv $(OMNI_VENV) && uv pip install -e .
	@touch $@

# =======================================================================
# Running the pipeline (HuggingFace dataset — default)
# =======================================================================

run: setup  ## Run omni step (HuggingFace dataset → SOAP notes JSONL)
	@mkdir -p $(RESULTS_DIR)
	@echo "==> Running omni step: $(DATASET) [$(SPLIT)] → $(OUTPUT)"
	cd $(OMNI_DIR) && $(OMNI_VENV)/bin/python run_omni.py \
		--dataset  $(DATASET) \
		--split    $(SPLIT) \
		$(if $(LIMIT),--limit $(LIMIT)) \
		--output   $(OUTPUT) \
		--model    $(MODEL) \
		--device   $(DEVICE) \
		--seed     $(SEED) \
		--resume

# =======================================================================
# Running the pipeline (CSV manifest fallback — for custom data)
# =======================================================================

run-manifest: setup  ## Run omni on a CSV manifest (custom data)
	@if [ -z "$(MANIFEST)" ]; then \
		echo "ERROR: MANIFEST not set. Usage: make run-manifest MANIFEST=path/to/data.csv"; \
		exit 1; \
	fi
	@mkdir -p $(RESULTS_DIR)
	@echo "==> Running omni step: $(MANIFEST) → $(OUTPUT)"
	cd $(OMNI_DIR) && $(OMNI_VENV)/bin/python run_omni.py \
		--manifest $(MANIFEST) \
		--output   $(OUTPUT) \
		--model    $(MODEL) \
		--device   $(DEVICE) \
		--seed     $(SEED) \
		--base-dir $(ROOT) \
		--resume

# =======================================================================
# Testing
# =======================================================================

test: setup  ## Smoke test with example manifest (short audio)
	@echo "==> Smoke test: running omni on example manifest"
	@mkdir -p $(RESULTS_DIR)
	cd $(OMNI_DIR) && $(OMNI_VENV)/bin/python run_omni.py \
		--manifest $(ROOT)/data/manifests/example.csv \
		--output   $(RESULTS_DIR)/test_omni_output.jsonl \
		--model    $(MODEL) \
		--device   $(DEVICE) \
		--seed     $(SEED) \
		--base-dir $(ROOT)
	@echo ""
	@echo "==> Omni smoke test result:"
	@cat $(RESULTS_DIR)/test_omni_output.jsonl
	@echo ""
	@echo "==> Omni smoke test passed."

TEST_MODEL ?= Qwen/Qwen2.5-Omni-3B

test-5: setup  ## Smoke test: omni on 5 HuggingFace validation samples (3B model)
	@echo "==> Smoke test: $(TEST_MODEL) on 5 validation samples"
	@mkdir -p $(RESULTS_DIR)
	@rm -f $(RESULTS_DIR)/test_omni_5.jsonl
	cd $(OMNI_DIR) && $(OMNI_VENV)/bin/python run_omni.py \
		--dataset  $(DATASET) \
		--split    validation \
		--limit    5 \
		--output   $(RESULTS_DIR)/test_omni_5.jsonl \
		--model    $(TEST_MODEL) \
		--device   $(DEVICE) \
		--seed     $(SEED)
	@echo ""
	@echo "==> Omni 5-sample smoke test result:"
	@cat $(RESULTS_DIR)/test_omni_5.jsonl
	@echo ""
	@echo "==> Omni 5-sample smoke test passed."

test-hf: setup  ## Run omni on N HuggingFace samples (MODEL and LIMIT configurable)
	@echo "==> Running omni on $(or $(LIMIT),all) HuggingFace [$(SPLIT)] samples"
	@mkdir -p $(RESULTS_DIR)
	cd $(OMNI_DIR) && $(OMNI_VENV)/bin/python run_omni.py \
		--dataset  $(DATASET) \
		--split    $(SPLIT) \
		$(if $(LIMIT),--limit $(LIMIT)) \
		--output   $(RESULTS_DIR)/test_hf_output.jsonl \
		--model    $(MODEL) \
		--device   $(DEVICE) \
		--seed     $(SEED) \
		--resume
	@echo ""
	@echo "==> HuggingFace test result:"
	@cat $(RESULTS_DIR)/test_hf_output.jsonl
	@echo ""
	@echo "==> HuggingFace test passed."

# =======================================================================
# Code quality
# =======================================================================

lint: setup  ## Run ruff linter and mypy type checker
	cd $(OMNI_DIR) && uv pip install ruff mypy 2>/dev/null || true
	$(OMNI_VENV)/bin/ruff check $(OMNI_DIR)/run_omni.py
	$(OMNI_VENV)/bin/mypy $(OMNI_DIR)/run_omni.py --ignore-missing-imports

lint-fix: setup  ## Auto-fix linting issues and format code
	cd $(OMNI_DIR) && uv pip install ruff 2>/dev/null || true
	$(OMNI_VENV)/bin/ruff check --fix $(OMNI_DIR)/run_omni.py
	$(OMNI_VENV)/bin/ruff format $(OMNI_DIR)/run_omni.py

format: setup  ## Format code with ruff
	cd $(OMNI_DIR) && uv pip install ruff 2>/dev/null || true
	$(OMNI_VENV)/bin/ruff format $(OMNI_DIR)/run_omni.py

check: lint  ## Run lint + unit tests (recommended before commit)
	cd $(OMNI_DIR) && uv pip install pytest 2>/dev/null || true
	$(OMNI_VENV)/bin/python -m pytest tests/ -v -m "not integration"
	@echo ""
	@echo "==> All checks passed!"

# =======================================================================
# Cleanup
# =======================================================================

clean:  ## Remove virtual environment and results
	rm -rf $(OMNI_VENV)
	rm -rf $(RESULTS_DIR)
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
