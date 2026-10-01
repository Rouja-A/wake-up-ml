# Compact Raw-IMU Activity Recognition (UCI HAR)

Take-home exercise for the research assistantship at the Embedded Systems Laboratory (ESL), EPFL.
A compact 1D CNN (**fewer than 20,000 trainable parameters**) classifies human activity from **raw**
accelerometer and gyroscope windows of the UCI HAR dataset.

**Report: `report/final_report.ipynb`** (rendered copies: `report/final_report.html`, `report/final_report.pdf`).

## Result in one paragraph

Three CNNs (A: 3,766 params, B: 14,390, C: 8,862) were compared on a subject-held-out validation split; C was selected,
retrained on all 21 training subjects for 17 epochs, and evaluated once on the 9 official test subjects:
**accuracy 0.9240, macro-F1 0.9231** (2,947 windows). A strictly causal Model D (3,906 params) and a linear baseline
were then trained (validation only) to build the accuracy-vs-cost Pareto analysis. Dominant failure: SITTING vs STANDING,
concentrated in a few subjects. Details, confusion matrix and discussion are in the report.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --out_dir data          # or extract the UCI HAR zip yourself
export UCI_HAR_DIR="data/UCI HAR Dataset"                # or pass --data_dir to the scripts
```

## Input representation

Six raw channels per 128-sample window (2.56 s @ 50 Hz), shape `(6, 128)`:
`total_acc_x/y/z, body_gyro_x/y/z`. The handcrafted feature files (`X_train.txt`, `X_test.txt`, `features*.txt`) are never
read (enforced in `src/data.py`, tested in `tests/test_data_pipeline.py`).

## Layout

```
configs/baseline.yaml      all data / split / preprocessing / training hyperparameters
src/                       data loading, preprocessing, models A/B/C, training, metrics, complexity (params/MACs)
experiments/research_extension/   Model D (causal), linear baseline, sensor ablation, Pareto (validation only)
scripts/                   run_experiments.py (Phase 1), train_final.py (Phase 2), evaluate_final*.py (test), ...
tests/                     unit tests (synthetic data)
results/metrics, figures   saved metrics, curves, final test evaluation, confusion matrix
results/checkpoints        Phase-1 and final (17-epoch) checkpoints
results/research_extension Model D / Linear / ablation / Pareto outputs
report/                    final_report.ipynb (+ .html/.pdf), step6_streaming_analysis.md (supporting derivation)
```

## Reproducing

```bash
python scripts/run_experiments.py --config configs/baseline.yaml --device auto       # Phase 1: A/B/C x 3 seeds (validation)
python scripts/train_final.py     --config configs/baseline.yaml --device auto       # Phase 2: 17 epochs, seed 123
python scripts/evaluate_final.py  --config configs/baseline.yaml                      # single official-test evaluation
python scripts/run_research_extension.py --config configs/baseline.yaml --device auto # Model D, Linear, ablation, Pareto
python tests/test_data_pipeline.py; python tests/test_model.py; python tests/test_step4.py
```

`evaluate_final.py` refuses to run if `results/metrics/final_test_evaluation.json` already exists (single-evaluation guard).
The submitted test numbers were produced with `scripts/evaluate_final_numpy.py`, a torch-free equivalent that reads the weights
from the `.pt` file; `--verify_against <predictions.csv>` shows it reproduces PyTorch predictions exactly (checked on 2,947/2,947
windows of an earlier checkpoint). On a machine with PyTorch, `evaluate_final.py` should give the same numbers.

## Protocol notes

* Validation is split **by subject** (17 train / 4 val subjects) because windows overlap 50 %.
* Normalisation statistics are fit on training data only and stored in each checkpoint.
* Phase-2 epoch budget = `round(mean(Phase-1 epochs_ran))` = `round(mean([17, 17, 18]))` = 17.
* An earlier 1-epoch draft of Phase 2 is archived under `results/historical_original_protocol/` for transparency only; it is not the submitted model.
