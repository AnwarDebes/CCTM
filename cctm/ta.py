"""Tsetlin Automaton bank (canonical, no trace).

Self-contained reimplementation of the standard 2-action Tsetlin Automaton
bank used by every TM in the family.  We deliberately do not import from
``thgtm`` so this repo can be reviewed independently.

State semantics (Granmo 2018):
    state in {1, ..., 2N}   ->  include literal iff state > N
"""

from __future__ import annotations

import numpy as np


class TABank:
    """Vectorised bank of canonical Tsetlin Automata.

    One TA per (clause, literal) pair.  No eligibility trace -- that lives
    in THGTM's ETTA.
    """

    def __init__(
        self,
        n_clauses: int,
        n_literals: int,
        n_states_per_action: int = 100,
        rng: np.random.Generator | None = None,
    ):
        self.n_clauses = int(n_clauses)
        self.n_literals = int(n_literals)
        self.N = int(n_states_per_action)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Start at exclude boundary.
        self.state = np.full(
            (self.n_clauses, self.n_literals), self.N, dtype=np.int32
        )

    def actions(self) -> np.ndarray:
        return self.state > self.N

    def clause_outputs(self, literals: np.ndarray, predict: bool = False) -> np.ndarray:
        literals = np.asarray(literals, dtype=bool)
        inc = self.actions()
        bad = inc & ~literals[np.newaxis, :]
        any_bad = bad.any(axis=1)
        out = (~any_bad).astype(np.int8)
        if predict:
            empty = ~inc.any(axis=1)
            out = np.where(empty, np.int8(0), out)
        return out

    def _apply_delta(self, delta: np.ndarray) -> None:
        np.clip(self.state + delta, 1, 2 * self.N, out=self.state)

    def type_I_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        s: float,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        c = np.asarray(clause_outputs, dtype=bool)
        x = np.asarray(literals, dtype=bool)
        C, L = self.n_clauses, self.n_literals
        boost_p = (s - 1.0) / s
        forget_p = 1.0 / s
        boost_mask = c[:, None] & x[None, :]
        forget_mask = ~boost_mask
        rand = self.rng.random((C, L)).astype(np.float32)
        delta = np.zeros((C, L), dtype=np.int32)
        delta[boost_mask & (rand < boost_p)] = 1
        delta[forget_mask & (rand < forget_p)] = -1
        if feedback_mask is not None:
            delta[~np.asarray(feedback_mask, dtype=bool), :] = 0
        self._apply_delta(delta)

    def type_II_feedback(
        self,
        clause_outputs: np.ndarray,
        literals: np.ndarray,
        feedback_mask: np.ndarray | None = None,
    ) -> None:
        c = np.asarray(clause_outputs, dtype=bool)
        x = np.asarray(literals, dtype=bool)
        inc = self.actions()
        mask = c[:, None] & (~x[None, :]) & (~inc)
        delta = mask.astype(np.int32)
        if feedback_mask is not None:
            delta[~np.asarray(feedback_mask, dtype=bool), :] = 0
        self._apply_delta(delta)
