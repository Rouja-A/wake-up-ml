# Research-extension results (auto-generated from measured runs; validation subjects only)

## Models

| Model | Mean val macro-F1 | Std (3 seeds) | Params | MAC/window | RF | Causal? |
|---|---:|---:|---:|---:|---:|---|
| Linear | 0.6203 | 0.0013 | 4,614 | 4,608 | 128 | no |
| A | 0.9890 | 0.0006 | 3,766 | 274,624 | 18 | no |
| B | 0.9811 | 0.0009 | 14,390 | 602,496 | 36 | no |
| C | 0.9898 | 0.0005 | 8,862 | 415,296 | 60 | no |
| D | 0.9778 | 0.0062 | 3,906 | 406,560 | 61 | yes (backbone) |

A/B/C rows are read from the frozen Phase-1 summary; Linear/D are newly trained in this extension.

## Sensor ablation (Model-A architecture)

| Arm | Channels | Params | MAC/window | Mean val macro-F1 | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| acc_gyro_full | 6 | 3,766 | 274,624 | 0.9890 | 0.0006 | 0.9883 | 0.9898 |
| acc_only | 3 | 3,334 | 219,328 | 0.9684 | 0.0039 | 0.9643 | 0.9737 |
| gyro_only | 3 | 3,334 | 219,328 | 0.7338 | 0.0011 | 0.7329 | 0.7353 |
| body_acc_gyro | 6 | 3,766 | 274,624 | 0.8485 | 0.0085 | 0.8419 | 0.8605 |

## Per-validation-subject macro-F1 (mean over seeds)

### A

| Subject | Windows | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 347 | 0.9990 | 0.9992 |
| 3 | 341 | 0.9756 | 0.9761 |
| 25 | 409 | 0.9829 | 0.9823 |
| 27 | 376 | 0.9991 | 0.9990 |

mean=0.9891 std=0.0102 min=0.9761 (subject 3) max=0.9992 (subject 1)

### C

| Subject | Windows | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 347 | 1.0000 | 1.0000 |
| 3 | 341 | 0.9892 | 0.9894 |
| 25 | 409 | 0.9731 | 0.9721 |
| 27 | 376 | 1.0000 | 1.0000 |

mean=0.9904 std=0.0114 min=0.9721 (subject 25) max=1.0000 (subject 1)

### D

| Subject | Windows | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 347 | 0.9962 | 0.9964 |
| 3 | 341 | 0.9912 | 0.9910 |
| 25 | 409 | 0.9356 | 0.9304 |
| 27 | 376 | 0.9982 | 0.9976 |

mean=0.9789 std=0.0281 min=0.9304 (subject 25) max=0.9976 (subject 27)

### Linear

| Subject | Windows | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 1 | 347 | 0.5178 | 0.5479 |
| 3 | 341 | 0.6794 | 0.6570 |
| 25 | 409 | 0.6129 | 0.5479 |
| 27 | 376 | 0.7420 | 0.6795 |

mean=0.6081 std=0.0607 min=0.5479 (subject 1) max=0.6795 (subject 27)

## Pareto dominance (validation macro-F1 vs cost)

- vs macs: Linear: non-dominated; D: dominated by A; A: non-dominated; B: dominated by A; C: non-dominated
- vs params: Linear: dominated by D; D: dominated by A; A: non-dominated; B: dominated by A; C: non-dominated
