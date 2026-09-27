# Compact Raw-IMU Activity Recognition for Streaming Wake-Up Processing

EPFL take-home for the internship "Streaming Wake-Up Networks for Low-Power Sensor Processing." A
compact 1D CNN (under 20,000 parameters) classifies human activity from raw UCI HAR accelerometer/
gyroscope windows. **The experiment is complete and frozen** — see `report/report.md` for the full
writeup and `results/` for the saved artifacts described below.

## Repository status

The official test set has **already been evaluated exactly once** for the submitted experiment. The
frozen result files under `results/metrics/` and `results/figures/` are the artifacts to inspect; there
is no need to rerun anything to reproduce the numbers in `report/report.md`. Commands below are provided
for reproducibility/inspection, not as an invitation to re-evaluate the test set (see "Reproducing each
stage" below).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --out_dir data      # optional; or extract the UCI HAR zip yourself
export UCI_HAR_DIR="data/UCI HAR Dataset"           # or pass --data_dir to every script below
```

## Input representation

Six raw channels per 128-sample (2.56 s @ 50 Hz) window, `(6, 128)` float32, fixed order:
`total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z`. The handcrafted feature
files (`X_train.txt`, `X_test.txt`, `features*.txt`) are never read by this pipeline — enforced in
`src/data.py` and covered by `tests/test_data_pipeline.py`.

## Project structure

```
configs/baseline.yaml       All data/split/preprocessing/training hyperparameters
src/
  data.py                   Raw UCI HAR loader (train-only / test-only / both), feature-file guard
  preprocessing.py          Subject-held-out split, train-only normalization
  dataset.py                PyTorch Dataset/DataLoader wrappers
  architectures.py          Declarative specs for Models A/B/C (single source of truth)
  model.py                  PyTorch model built from the specs
  shape_calc.py             Tensor-free params/MACs/receptive-field/activation-size formulas
  complexity.py             Torch hook-based measurement, cross-checked against shape_calc.py
  np_reference.py           Pure-NumPy reference forward pass (no torch dependency)
  train.py                  Training loop, optimizer/scheduler, early stopping, checkpointing
  evaluate.py               Checkpoint loading + inference + confusion-matrix plotting
  metrics.py                Accuracy/F1/confusion-matrix/error-group-breakdown metrics
  experiment_log.py         Phase-1 CSV logging + architecture/epoch-budget selection logic
scripts/
  inspect_data.py           Dataset layout/consistency inspection
  check_preprocessing.py    Split + normalization sanity checks, sample-window plot
  analyze_models.py         Model complexity comparison (params/MACs/RF/activation memory)
  smoke_test.py             Short pre-flight pipeline check (not part of the frozen record)
  run_experiments.py        Phase 1: 3 seeds x A/B/C architecture comparison
  train_final.py            Phase 2: retrain the selected architecture on all training data
  evaluate_final.py         ONE-TIME official test evaluation (guarded, see below)
  verify_report_numbers.py  Static consistency checks against the frozen result files
tests/                      Unit tests (synthetic-data based, no real dataset/torch required for most)
results/
  checkpoints/              Phase-1 (per seed) and Phase-2 (final) model checkpoints
  metrics/                  experiment_log.csv, architecture_selection.json, final_model_info.json,
                            final_test_evaluation.json, final_test_predictions.csv,
                            streaming_analysis.json, model_analysis.json
  figures/                  confusion_matrix_final.png, sample_windows.png
report/
  report.md                    Final report (submit this)
  step6_streaming_analysis.md  Extended streaming/embedded analysis (report.md Section 8 condenses this)
```

## Reproducing each stage

```bash
# Dataset inspection and preprocessing sanity checks
python scripts/inspect_data.py --data_dir "$UCI_HAR_DIR"
python scripts/check_preprocessing.py --data_dir "$UCI_HAR_DIR" --config configs/baseline.yaml

# Model complexity analysis (params/MACs/receptive field/activation memory for A/B/C)
python scripts/analyze_models.py

# Pre-flight pipeline check (short, writes to results/smoke_test/, does not touch the frozen record)
python scripts/smoke_test.py --config configs/baseline.yaml --model A --seed 42 --epochs 2 --device auto

# Phase 1: architecture comparison (3 seeds x A/B/C = 9 runs; official test untouched)
python scripts/run_experiments.py --config configs/baseline.yaml --device auto

# Phase 2: retrain the selected architecture from scratch on all official-training data
python scripts/train_final.py --config configs/baseline.yaml --device auto
```

**Final evaluation — already performed for this submission, one-time only.** The command below is
documented for reproducibility; it is guarded against accidental re-use (`scripts/evaluate_final.py`
refuses to overwrite `results/metrics/final_test_evaluation.json` unless `--force` is passed). For this
submission, **do not run it and do not pass `--force`** — inspect the saved result files instead:
`results/metrics/final_test_evaluation.json`, `results/metrics/final_test_predictions.csv`,
`results/figures/confusion_matrix_final.png`.

```bash
# For reference only -- already executed for this submission:
# python scripts/evaluate_final.py --config configs/baseline.yaml --device cuda
```

## Tests and consistency checks

```bash
python tests/test_data_pipeline.py           # data loading/split/normalization (synthetic data)
python tests/test_model.py                   # architecture/complexity correctness
python tests/test_step4.py                   # training loop, logging, Phase-1/2 isolation
python scripts/verify_report_numbers.py      # static arithmetic checks against results/metrics/*
```

`verify_report_numbers.py` runs no training or inference: it recomputes accuracy/F1/error-group counts
from the saved confusion matrix and recounts the confusion matrix from `final_test_predictions.csv`,
checking both against `final_test_evaluation.json`; it also checks `architecture_selection.json` and
`final_model_info.json` for consistency (selected model, seed, epoch budget) and cross-checks
`streaming_analysis.json` against a fresh recomputation from `src/architectures.py`. If any of those
result files are absent from a given checkout, the corresponding check is skipped and reported as such.

## Where the report and results live

- **Report:** `report/report.md` (main), `report/step6_streaming_analysis.md` (extended streaming
  analysis referenced by report.md Section 8).
- **Final metrics:** `results/metrics/final_test_evaluation.json`, `results/metrics/final_test_predictions.csv`.
- **Confusion matrix figure:** `results/figures/confusion_matrix_final.png`.
- **Phase-1 log:** `results/experiment_log.csv`; selection outcome: `results/metrics/architecture_selection.json`.
- **Phase-2 metadata:** `results/metrics/final_model_info.json`; checkpoint: `results/checkpoints/phase2_final_C_seed123.pt`.
- **Model complexity:** `results/metrics/model_analysis.json`; streaming derivation: `results/metrics/streaming_analysis.json`.
