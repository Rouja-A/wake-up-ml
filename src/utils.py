"""Small shared helpers: seeding, config/JSON I/O, and a tiny pass/fail reporter."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def default_data_dir() -> str:
    """Dataset location when --data_dir is not given: $UCI_HAR_DIR, else ./data/UCI HAR Dataset."""
    return os.environ.get("UCI_HAR_DIR", "data/UCI HAR Dataset")


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


class Checker:
    """Prints [PASS]/[FAIL]/[WARN] lines and remembers failures (used by the scripts)."""

    def __init__(self) -> None:
        self.n_checks = 0
        self.failures: list[str] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.n_checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(name)
        return bool(ok)

    @staticmethod
    def warn(name: str, detail: str = "") -> None:
        print(f"  [WARN] {name}" + (f"  ({detail})" if detail else ""))

    def summary(self) -> int:
        """Print a summary; return a process exit code (0 = all passed)."""
        if self.failures:
            print(f"\n{len(self.failures)} of {self.n_checks} checks FAILED:")
            for f in self.failures:
                print(f"  - {f}")
            return 1
        print(f"\nAll {self.n_checks} checks passed.")
        return 0
