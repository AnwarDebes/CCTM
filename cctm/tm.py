"""Base binary Tsetlin Machine (canonical Granmo 2018).

Used as the substrate for the Counterfactual Causal Tsetlin Machine (CCTM).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .ta import TABank


def to_literals(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.int8)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be binary 0/1")
    return np.concatenate([X, 1 - X], axis=1)


@dataclass
class TMConfig:
    n_clauses: int
    n_features: int
    n_states_per_action: int = 100
    threshold: int = 15
    s: float = 3.9
    seed: int | None = None


class TsetlinMachine:
    """Binary Tsetlin Machine.  Half +polarity, half -polarity clauses."""

    def __init__(self, cfg: TMConfig):
        if cfg.n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even.")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.bank = TABank(
            n_clauses=cfg.n_clauses,
            n_literals=2 * cfg.n_features,
            n_states_per_action=cfg.n_states_per_action,
            rng=self.rng,
        )
        self.polarity = np.empty(cfg.n_clauses, dtype=np.int8)
        self.polarity[: cfg.n_clauses // 2] = 1
        self.polarity[cfg.n_clauses // 2:] = -1

    def vote_sum(self, literals_row, predict: bool):
        co = self.bank.clause_outputs(literals_row, predict=predict)
        v = int((self.polarity.astype(np.int32) * co.astype(np.int32)).sum())
        return v, co

    def predict_one(self, literals_row) -> int:
        v, _ = self.vote_sum(literals_row, predict=True)
        return 1 if v > 0 else 0

    def predict(self, X) -> np.ndarray:
        lits = to_literals(X)
        return np.array([self.predict_one(lits[i]) for i in range(lits.shape[0])],
                        dtype=np.int8)

    def score(self, X, y) -> float:
        return float((self.predict(X) == y).mean())

    def _feedback_subset(self, clause_outputs, literals, polarity, ftype, p):
        if p <= 0:
            return
        idx = np.where(self.polarity == polarity)[0]
        if idx.size == 0:
            return
        chosen = self.rng.random(idx.size) < p
        mask = np.zeros(self.cfg.n_clauses, dtype=bool)
        mask[idx[chosen]] = True
        if not mask.any():
            return
        if ftype == "I":
            self.bank.type_I_feedback(clause_outputs, literals, s=self.cfg.s,
                                      feedback_mask=mask)
        else:
            self.bank.type_II_feedback(clause_outputs, literals,
                                       feedback_mask=mask)

    def train_step(self, literals_row, y: int) -> None:
        v, co = self.vote_sum(literals_row, predict=False)
        T = self.cfg.threshold
        v_clip = max(-T, min(T, v))
        if y == 1:
            self._feedback_subset(co, literals_row, +1, "I",
                                  (T - v_clip) / (2.0 * T))
            self._feedback_subset(co, literals_row, -1, "II",
                                  (T + v_clip) / (2.0 * T))
        else:
            self._feedback_subset(co, literals_row, +1, "II",
                                  (T + v_clip) / (2.0 * T))
            self._feedback_subset(co, literals_row, -1, "I",
                                  (T - v_clip) / (2.0 * T))

    def fit(self, X, y, epochs=10, shuffle=True, verbose=False):
        lits = to_literals(X)
        y = np.asarray(y, dtype=np.int8)
        for epoch in range(epochs):
            order = np.arange(lits.shape[0])
            if shuffle:
                self.rng.shuffle(order)
            for i in order:
                self.train_step(lits[i], int(y[i]))
            if verbose:
                print(f"[TM] epoch {epoch+1}/{epochs} acc={self.score(X, y):.4f}")
        return self
