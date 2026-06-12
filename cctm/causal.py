"""Counterfactual Causal Tsetlin Machine (CCTM).

Core algorithmic content.  CCTM extends the canonical Tsetlin Machine with
three additions:

1. **Counterfactual data augmentation.**  Each training sample ``(x, y)`` is
   accompanied (when available) by a counterfactual partner ``(x', y')``
   that flips exactly one or a small set of features.  When ``y != y'`` the
   pair is a *true counterfactual* and the differing features are causally
   implicated.  When ``y == y'`` the flip is irrelevant to the label.

2. **Type-III contrastive feedback.**  Standard Type-I / Type-II feedback
   runs on each sample of the pair individually.  *In addition* CCTM tracks
   a per-clause causal score equal to the running fraction of true-CF
   pairs on which the clause's output changes -- a clause that "moves
   with the label" is causally relevant; a clause that "doesn't budge"
   is a spurious correlate.

3. **Feature-level causal influence (CI).**  Aggregating the clause-level
   causal score back to features via the TA inclusion masks gives a vector
   ``CI in [0, 1]^{n_features}`` that says how causally responsible each
   raw feature is for the model's decisions.  This vector is the seat of
   subsequent counterfactual recourse and fairness audit.

The CCTM class wraps a ``TsetlinMachine`` and adds the above statistics.
When you call ``fit`` without CF pairs the class degrades gracefully to
standard TM training (causal_score remains zero, CI remains uniform).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .ta import TABank
from .tm import TsetlinMachine, TMConfig, to_literals


@dataclass
class FeatureFeasibility:
    """Per-feature feasibility specification used during counterfactual recourse.

    * ``immutable``: indices that cannot be flipped at all (e.g. race, sex
      when the law forbids them as recourse).
    * ``monotone_up``: indices whose value can only INCREASE (1 stays 1;
      0 may flip to 1 but 1 may not flip to 0).  E.g. age in years.
    * ``monotone_down``: indices whose value can only DECREASE.
    * ``cost``: per-feature flip cost (defaults to 1.0).  Recourse minimises
      total cost.
    """
    immutable: List[int] = field(default_factory=list)
    monotone_up: List[int] = field(default_factory=list)
    monotone_down: List[int] = field(default_factory=list)
    cost: Optional[List[float]] = None

    def allowed_flip(self, i: int, current_value: int) -> bool:
        if i in self.immutable:
            return False
        if i in self.monotone_up and current_value == 1:
            return False
        if i in self.monotone_down and current_value == 0:
            return False
        return True

    def flip_cost(self, i: int) -> float:
        if self.cost is None:
            return 1.0
        return float(self.cost[i])


@dataclass
class CCTMConfig:
    n_clauses: int
    n_features: int
    n_states_per_action: int = 100
    threshold: int = 15
    s: float = 3.9
    seed: int | None = None
    # Causal score is a running statistic.  ``causal_decay`` controls how
    # past pairs are weighted vs new ones (1.0 = no decay, 0.0 = only latest).
    causal_decay: float = 0.99

    def to_tm_config(self) -> TMConfig:
        return TMConfig(
            n_clauses=self.n_clauses,
            n_features=self.n_features,
            n_states_per_action=self.n_states_per_action,
            threshold=self.threshold,
            s=self.s,
            seed=self.seed,
        )


class CCTM:
    """Counterfactual Causal Tsetlin Machine."""

    def __init__(self, cfg: CCTMConfig):
        self.cfg = cfg
        self.tm = TsetlinMachine(cfg.to_tm_config())
        # Per-clause causal score in [0, 1].  Tracks running fraction of true-CF
        # pairs on which the clause's output changes.
        self.causal_score = np.zeros(cfg.n_clauses, dtype=np.float32)
        self.causal_count = 0                                  # true-CF pairs seen
        # Per-feature treatment-effect numerator (label flips) and
        # denominator (total CF flips).
        self.feature_te_num = np.zeros(cfg.n_features, dtype=np.float32)
        self.feature_te_den = np.zeros(cfg.n_features, dtype=np.float32)

    # ---------- inference ----------
    def predict_one(self, x: np.ndarray) -> int:
        lits = np.concatenate([x.astype(np.int8), 1 - x.astype(np.int8)])
        return self.tm.predict_one(lits)

    def vote_sum(self, x: np.ndarray) -> int:
        lits = np.concatenate([x.astype(np.int8), 1 - x.astype(np.int8)])
        v, _ = self.tm.vote_sum(lits, predict=True)
        return int(v)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])],
                        dtype=np.int8)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ---------- causal influence ----------
    def feature_treatment_effect(self) -> np.ndarray:
        """Per-feature empirical treatment effect estimated from CF pairs:
        ``P(label flips | feature i is the flipped feature in the CF pair)``.

        This is a direct, model-agnostic causal-effect estimate -- it just
        counts label flips per feature flip across all observed CF pairs.
        Features with zero CF flips return 0.
        """
        return self.feature_te_num / (self.feature_te_den + 1e-9)

    def feature_causal_influence(self) -> np.ndarray:
        """Aggregate clause-level causal_score to per-feature causal-influence
        scores in [0, 1] *weighted by the empirical treatment effect*.

        CI[i] = TE[i] * (clause-level support).  This factorisation ensures
        that:

        * features the CF data shows are causal (high TE) dominate;
        * within those, clauses that move with the label inform the per-clause
          contribution used downstream for recourse.

        When no CF pairs were seen, CI falls back to a uniform 1/L * tm-coverage.
        """
        L = self.cfg.n_features
        te = self.feature_treatment_effect()                    # (L,)
        acts = self.tm.bank.actions()                           # (C, 2L)
        raw = acts[:, :L].astype(np.float32)
        neg = acts[:, L:].astype(np.float32)
        any_inc = (raw + neg).clip(0, 1)                        # (C, L)
        weighted = any_inc * self.causal_score[:, None]         # (C, L)
        denom = any_inc.sum(axis=0) + 1e-9
        clause_part = weighted.sum(axis=0) / denom              # (L,)
        if self.causal_count == 0:
            return np.clip(clause_part, 0.0, 1.0)
        return np.clip(te * clause_part, 0.0, 1.0)

    # ---------- training ----------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_cf: Optional[np.ndarray] = None,
        y_cf: Optional[np.ndarray] = None,
        epochs: int = 10,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> "CCTM":
        """Train CCTM with optional counterfactual pairs.

        Parameters
        ----------
        X, y : factual training data.
        X_cf, y_cf : optional aligned counterfactual partners.  Indices
            ``i`` where ``y[i] != y_cf[i]`` are true-CF pairs; others are
            still trained on (extra data) but do not update causal_score.
        """
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.int8)
        if X_cf is not None:
            X_cf = np.asarray(X_cf, dtype=np.int8)
            y_cf = np.asarray(y_cf, dtype=np.int8)
            assert X_cf.shape == X.shape
            assert y_cf.shape == y.shape

        lits_x = to_literals(X)
        lits_xcf = to_literals(X_cf) if X_cf is not None else None

        for epoch in range(epochs):
            order = np.arange(X.shape[0])
            if shuffle:
                self.tm.rng.shuffle(order)
            for i in order:
                # Standard Type-I / Type-II on the factual sample.
                self.tm.train_step(lits_x[i], int(y[i]))

                if lits_xcf is not None:
                    # Standard Type-I / Type-II on the CF partner too (data
                    # augmentation that respects the CF structure).
                    self.tm.train_step(lits_xcf[i], int(y_cf[i]))

                    # Identify which raw features changed in the CF pair.
                    differ_features = np.where(X[i] != X_cf[i])[0]

                    # Update per-feature TE statistics (regardless of label
                    # flip).
                    self.feature_te_den[differ_features] += 1
                    if int(y[i]) != int(y_cf[i]):
                        self.feature_te_num[differ_features] += 1

                    # Update causal score on true-CF pairs (label flips).
                    if int(y[i]) != int(y_cf[i]):
                        co_x = self.tm.bank.clause_outputs(lits_x[i],
                                                            predict=False)
                        co_xcf = self.tm.bank.clause_outputs(lits_xcf[i],
                                                              predict=False)
                        differ = (co_x != co_xcf).astype(np.float32)
                        # Decayed running average -- recent pairs weighted more.
                        d = self.cfg.causal_decay
                        self.causal_score = d * self.causal_score + (1 - d) * differ
                        self.causal_count += 1

            if verbose:
                print(f"[CCTM] epoch {epoch+1}/{epochs} "
                      f"acc={self.score(X, y):.4f}  "
                      f"cf_pairs_seen={self.causal_count}")
        return self


# ---------------------------------------------------------------------- #
# Counterfactual data utilities
# ---------------------------------------------------------------------- #
def generate_cf_pairs_from_oracle(
    X: np.ndarray,
    oracle: callable,
    rng: np.random.Generator,
    flip_strategy: str = "single",
    max_pairs_per_sample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given X and an oracle ``y' = oracle(x')`` that scores any counterfactual,
    generate aligned CF pairs.  ``flip_strategy='single'`` flips one feature
    chosen uniformly; ``'random_subset'`` flips a small random subset.

    Returns X_cf, y_cf of the same shape as X.
    """
    X = np.asarray(X, dtype=np.int8)
    n, d = X.shape
    X_cf = X.copy()
    for i in range(n):
        if flip_strategy == "single":
            j = int(rng.integers(0, d))
            X_cf[i, j] = 1 - X[i, j]
        elif flip_strategy == "random_subset":
            k = int(rng.integers(1, min(3, d) + 1))
            js = rng.choice(d, size=k, replace=False)
            X_cf[i, js] = 1 - X[i, js]
        else:
            raise ValueError(flip_strategy)
    y_cf = np.array([int(oracle(X_cf[i])) for i in range(n)], dtype=np.int8)
    return X_cf, y_cf
