# Layered Uncertainty Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export per-signal, per-representative-layer uncertainty evaluation metrics and generate line plots for analysis.

**Architecture:** Extend `pipeline/evaluate_uncertainty.py` with small helper functions that reshape already-computed layer features into an analysis table, evaluate each layer-feature signal with the existing metric pipeline, and render plots to the chosen output directory without changing current summary outputs.

**Tech Stack:** Python 3.10, pandas, numpy, matplotlib, scikit-learn

---

### Task 1: Add failing tests for layered analysis helpers

**Files:**
- Create: `test_evaluate_uncertainty_layer_analysis.py`
- Modify: `pipeline/evaluate_uncertainty.py`

- [ ] Write tests for representative-layer sorting, analysis-table layout, and plot file creation.
- [ ] Run the focused test file in the `inside-umpire` conda environment and confirm it fails because helper functions do not exist yet.

### Task 2: Implement layered analysis exports

**Files:**
- Modify: `pipeline/evaluate_uncertainty.py`
- Test: `test_evaluate_uncertainty_layer_analysis.py`

- [ ] Add helper functions for layer ordering, method evaluation across layer-feature signals, result-table export, and per-signal line-plot rendering.
- [ ] Integrate the helper flow into the main CLI while preserving existing summary outputs.
- [ ] Re-run the focused test file and confirm it passes.

### Task 3: Verify on the existing experiment output

**Files:**
- Modify: `pipeline/evaluate_uncertainty.py`

- [ ] Run `pipeline/evaluate_uncertainty.py` in the `inside-umpire` environment against `test_output/full_gpu2_tuned/generations_with_uncertainty.pkl`.
- [ ] Confirm new CSV/JSON/PNG outputs appear under `test_output/full_gpu2_tuned_eval`.
- [ ] Spot-check one layered metric entry against the generated table to ensure the output shape is correct.
