# Research extension — streaming wake-up study (validation-only)

Isolated, **training-subjects-only** exploratory study. It never loads or evaluates the official UCI test
split, never touches Models A/B/C, their checkpoints, or any frozen result, and writes only under
`results/research_extension/`. The primary assignment (`report/final_report.ipynb`) is unchanged.

**Status: run to completion (validation only). Results: `results/research_extension/metrics/research_extension_tables.md`, summarised in `report/final_report.ipynb`.**

## What it does (18 training runs, seeds 0/1/2, Phase-1 split and protocol)

| Runs | What |
|---|---|
| 3 | **Linear** baseline: flatten 6x128 -> `Linear(768,6)` (RQ1) |
| 3 | **Model D**: tiny causal dilated depthwise-separable CNN, 3,906 params (RQ2) |
| 12 | **Sensor ablation** (Model-A architecture, fixed beforehand): full 6-ch, accelerometer-only, gyroscope-only, body_acc+gyro; 3 seeds each (RQ4) |

Then (no extra training): per-validation-subject metrics for A, C (frozen Phase-1 checkpoints, read-only),
D and Linear; complexity table; Pareto plots (RQ3).

## Run locally (Windows, Conda `wake_up_ml`, CUDA GPU)

```bat
cd C:\Users\Rouja\Downloads\project
conda activate wake_up_ml

:: 0. snapshot the frozen files BEFORE anything (authoritative baseline for your copy)
python scripts\verify_frozen_integrity.py --write_snapshot results\research_extension\frozen_before.txt

:: 1. tests BEFORE training (torch-gated tests must show PASS, not "skipped")
python tests\test_research_extension.py
python tests\test_model.py
python tests\test_step4.py
python tests\test_data_pipeline.py

:: 2. plan only (no data loaded, nothing written)
python scripts\run_research_extension.py --config configs\baseline.yaml --device cuda --dry_run

:: 3. optional pipeline smoke test (2 epochs, 1 seed, outputs in results\research_extension\_smoke\; NOT results)
python scripts\run_research_extension.py --config configs\baseline.yaml --device cuda --smoke

:: 4. the real run
python scripts\run_research_extension.py --config configs\baseline.yaml --device cuda

:: 5. frozen-artifact integrity AFTER training
python scripts\verify_frozen_integrity.py --snapshot results\research_extension\frozen_before.txt
```

`experiments\research_extension\frozen_manifest_sha256.txt` is a second reference: hashes of the frozen files as
they were in the uploaded `project.zip`. Use `--snapshot experiments\research_extension\frozen_manifest_sha256.txt`
to compare against it instead (a mismatch would then also reveal edits you made to your copy yourself).

## Safety guarantees (all enforced in code and tested)

- Data only via `data_ext.py` -> `src.data.load_train_only`. `load_dataset`, `load_test_only`,
  `evaluate_final`, and the test-loader builders are forbidden symbols; a static scan
  (`official_test_firewall.py`) fails the tests if any extension module or the entry script references them.
- Train/val subjects are asserted equal to the frozen `results/metrics/split.json`; normalization is refit on
  each channel subset's own training windows.
- `output_guard.py` refuses any output directory outside `results/research_extension/` or overlapping a frozen
  path, and refuses to start if extension outputs already exist (`--overwrite_extension_outputs` replaces only those).
- Run names all start with `research_extension_`, so they cannot collide with `phase1_*`/final names.
- Frozen files are only read: Phase-1 A/C checkpoints (per-subject analysis), `split.json`,
  `architecture_selection.json` (reference A/B/C validation numbers).

## Outputs (`results/research_extension/`)

- `checkpoints/research_extension_{Linear,D,ablation_<arm>}_seed{0,1,2}.pt` (18)
- `metrics/research_extension_*_curves.json` (18), `research_extension_summary.json`,
  `research_extension_tables.md` (auto-generated result tables)
- `figures/pareto_macs.png`, `pareto_params.png`, `per_subject_macro_f1.png`
- `frozen_before.txt` (your snapshot from step 0)

## After the run

Copy the values from `metrics/research_extension_tables.md` into the pending cells of
`report/research_extension.md` (Sec. 7–10) and write the discussion from the measured numbers. Do not
tune D or change hypotheses after seeing results.
