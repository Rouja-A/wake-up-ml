"""PyTorch Dataset / DataLoader wrappers (the only module that requires torch in the data pipeline)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import ChannelStats, FinalPreparedData, FinalTrainOnlyData, PreparedData, standardize


class HARDataset(Dataset):
    """Item = (x, y): x float32 tensor (C, 128), y int64 scalar class index in [0, K-1]."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3 or len(x) != len(y):
            raise ValueError(f"x must be (N, C, T) with N == len(y); got x={x.shape}, y={y.shape}")
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def make_loaders(p: PreparedData, batch_size: int = 64, seed: int = 42, num_workers: int = 0) -> dict:
    """Train (shuffled with a seeded generator) and validation loaders. Test data is deliberately excluded."""
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        "train": DataLoader(HARDataset(p.x_train, p.y_train), batch_size=batch_size, shuffle=True,
                            generator=g, num_workers=num_workers),
        "val": DataLoader(HARDataset(p.x_val, p.y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def make_test_loader(p: PreparedData, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
    """ONLY for the final evaluation of the already-selected model. Never use for model selection."""
    return DataLoader(HARDataset(p.x_test, p.y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_final_train_loader(p, batch_size: int = 64, seed: int = 42, num_workers: int = 0) -> DataLoader:
    """Phase 2: shuffled loader over the Phase-2 training data. Accepts either FinalPreparedData
    (train+val combined, test also standardized) or FinalTrainOnlyData (train only, no test access);
    only `.x_train`/`.y_train` are read, so this never touches a test field on either type."""
    if not isinstance(p, (FinalPreparedData, FinalTrainOnlyData)):
        raise TypeError(f"expected FinalPreparedData or FinalTrainOnlyData, got {type(p)}")
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(HARDataset(p.x_train, p.y_train), batch_size=batch_size, shuffle=True,
                      generator=g, num_workers=num_workers)


def make_final_test_loader(p: FinalPreparedData, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
    """Phase 2: the official test set, from a FinalPreparedData that already loaded+standardized it.
    Intended to be consumed exactly once, by scripts/evaluate_final.py."""
    return DataLoader(HARDataset(p.x_test, p.y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_test_loader_from_stats(x_test_raw, y_test, stats: ChannelStats, batch_size: int = 64,
                                num_workers: int = 0) -> DataLoader:
    """Standardize raw test windows with SAVED (not recomputed) normalization statistics and wrap them in
    a DataLoader. Used by scripts/evaluate_final.py, which loads the test split independently (via
    `src.data.load_test_only`) and must not recompute normalization from any training data it does not
    have loaded."""
    x = standardize(x_test_raw, stats)
    return DataLoader(HARDataset(x, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers)
