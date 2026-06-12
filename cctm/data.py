"""Synthetic datasets used by the CCTM experiments.

We deliberately bundle every dataset so the paper is end-to-end reproducible
without any external data download.  All datasets are Boolean-valued so they
flow straight into TM training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Synthetic Structural Causal Model
# ---------------------------------------------------------------------- #
def generate_scm_dataset(
    n: int = 2000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], int]]:
    """A small SCM with five binary features and a Boolean label.

    Causal structure::

        x0 -- treatment (causal)
        x1 -- confounder (causal AND correlates with x0)
        x2 -- mediator (caused BY x0, NOT independently causal once we condition)
        x3 -- noise (uncorrelated, NOT causal)
        x4 -- weak causal (XOR with x0)

    True label::

        y = (x0 AND x1) OR (x4 XOR x0)

    Ground-truth feature ATE (under do-calculus on a single feature with all
    others held at their observed value, averaged over the population):
        ATE[i] = E[ y | do(x_i := 1) ] - E[ y | do(x_i := 0) ]

    Returns (X, y, ATE, oracle_label_fn).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 5), dtype=np.int8)
    # x0: treatment, 50/50
    X[:, 0] = rng.integers(0, 2, size=n)
    # x1: confounder, correlated with x0 (P(x1 = 1 | x0) = 0.7 if x0=1 else 0.3)
    p = np.where(X[:, 0] == 1, 0.7, 0.3)
    X[:, 1] = (rng.random(size=n) < p).astype(np.int8)
    # x2: mediator (caused by x0 noisily)
    p_mediator = np.where(X[:, 0] == 1, 0.8, 0.2)
    X[:, 2] = (rng.random(size=n) < p_mediator).astype(np.int8)
    # x3: independent noise
    X[:, 3] = rng.integers(0, 2, size=n)
    # x4: independent
    X[:, 4] = rng.integers(0, 2, size=n)

    def oracle(x_row: np.ndarray) -> int:
        x = np.asarray(x_row, dtype=int).reshape(-1)
        return int((x[0] & x[1]) | (x[4] ^ x[0]))

    y = np.array([oracle(X[i]) for i in range(n)], dtype=np.int8)

    # Empirically compute population ATE for each feature by intervening on
    # everyone and averaging y over the rest of X.  This is the true causal
    # effect under intervention (and so the right yardstick).
    ATE = np.zeros(5, dtype=np.float32)
    for f in range(5):
        Xa = X.copy(); Xa[:, f] = 1
        Xb = X.copy(); Xb[:, f] = 0
        ya = np.array([oracle(Xa[i]) for i in range(n)], dtype=np.float32)
        yb = np.array([oracle(Xb[i]) for i in range(n)], dtype=np.float32)
        ATE[f] = float((ya - yb).mean())
    return X, y, ATE, oracle


# ---------------------------------------------------------------------- #
# Synthetic loan-denial dataset (lending-style)
# ---------------------------------------------------------------------- #
@dataclass
class LoanColumns:
    """Index reference for the synthetic loan dataset."""
    high_income: int = 0
    has_stable_job: int = 1
    paid_past_loans: int = 2
    has_high_education: int = 3
    age_over_25: int = 4
    age_over_45: int = 5
    gender_male: int = 6                # protected (immutable, never recourse)
    citizen: int = 7                    # immutable
    n_features: int = 8


def generate_loan_dataset(
    n: int = 2000,
    seed: int = 0,
    bias_strength: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, LoanColumns, Callable[[np.ndarray], int]]:
    """A bundled synthetic 8-feature 'loan-denial' dataset.

    True decision rule (ground truth):

        y = 1  iff  (high_income AND paid_past_loans)
                    OR (has_stable_job AND paid_past_loans AND age_over_25)
                    OR (has_high_education AND has_stable_job AND age_over_25)

    Plus a small *biased* bonus toward gender_male (the bias_strength
    parameter) that the fairness audit experiment will detect and prune.
    Gender and citizenship are demographic features and must not be used
    in recourse (they're set to immutable in ``feasibility``).

    Returns (X, y, columns, oracle).
    """
    cols = LoanColumns()
    rng = np.random.default_rng(seed)
    X = np.zeros((n, cols.n_features), dtype=np.int8)
    # Marginals chosen so the dataset is balanced but realistic.
    X[:, cols.high_income] = (rng.random(n) < 0.35).astype(np.int8)
    X[:, cols.has_stable_job] = (rng.random(n) < 0.60).astype(np.int8)
    X[:, cols.paid_past_loans] = (rng.random(n) < 0.55).astype(np.int8)
    X[:, cols.has_high_education] = (rng.random(n) < 0.40).astype(np.int8)
    X[:, cols.age_over_25] = (rng.random(n) < 0.75).astype(np.int8)
    X[:, cols.age_over_45] = ((rng.random(n) < 0.40) & (X[:, cols.age_over_25] == 1)).astype(np.int8)
    X[:, cols.gender_male] = (rng.random(n) < 0.50).astype(np.int8)
    X[:, cols.citizen] = (rng.random(n) < 0.85).astype(np.int8)

    def oracle(x_row: np.ndarray) -> int:
        x = np.asarray(x_row, dtype=int).reshape(-1)
        base = (
            (x[cols.high_income] and x[cols.paid_past_loans]) or
            (x[cols.has_stable_job] and x[cols.paid_past_loans] and x[cols.age_over_25]) or
            (x[cols.has_high_education] and x[cols.has_stable_job] and x[cols.age_over_25])
        )
        return 1 if base else 0

    # Apply bias as label flips skewed by gender_male.  Used for audit demos.
    y = np.array([oracle(X[i]) for i in range(n)], dtype=np.int8)
    if bias_strength > 0:
        flip_to_approve = (X[:, cols.gender_male] == 1) & (y == 0)
        mask = flip_to_approve & (rng.random(n) < bias_strength)
        y[mask] = 1
        flip_to_deny = (X[:, cols.gender_male] == 0) & (y == 1)
        mask = flip_to_deny & (rng.random(n) < bias_strength * 0.5)
        y[mask] = 0
    return X, y, cols, oracle


def split_train_test(X, y, frac_train=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_tr = int(frac_train * n)
    return (X[idx[:n_tr]], y[idx[:n_tr]],
            X[idx[n_tr:]], y[idx[n_tr:]])
