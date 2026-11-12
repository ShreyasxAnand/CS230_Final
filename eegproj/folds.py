"""Cross-subject splits and fold-safe normalisation.

Protocol (guide, Stage 1): train on the other people's session 1, test on the
held-out people's session 2, so every test is a new person and a new day.

- Development folds: 6 groups of 9 people (fixed seed), used while building.
- Final folds: all 54 leave-one-subject-out folds, run with 5 repeats.

A few training subjects are set aside as validation for early stopping, so
the test people never influence training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALL_SUBJECTS = list(range(1, 55))


@dataclass
class Fold:
    name: str
    train: list[int]
    val: list[int]
    test: list[int]


def _split_val(train: list[int], n_val: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    perm = rng.permutation(train)
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


def dev_folds(subjects=ALL_SUBJECTS, n_groups: int = 6, n_val: int = 5, seed: int = 0) -> list[Fold]:
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(subjects), min(n_groups, len(subjects)))
    folds = []
    for i, g in enumerate(groups):
        test = sorted(g.tolist())
        rest = [s for s in subjects if s not in test]
        train, val = _split_val(rest, n_val, rng)
        folds.append(Fold(f"dev{i}", train, val, test))
    return folds


def loso_folds(subjects=ALL_SUBJECTS, n_val: int = 5, seed: int = 0) -> list[Fold]:
    rng = np.random.default_rng(seed)
    folds = []
    for s in subjects:
        rest = [t for t in subjects if t != s]
        train, val = _split_val(rest, n_val, rng)
        folds.append(Fold(f"loso{s:02d}", train, val, [s]))
    return folds


class ChannelScaler:
    """Per-channel mean/std, fitted on training subjects only (leakage rule)."""

    def fit(self, X: np.ndarray) -> "ChannelScaler":
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)
