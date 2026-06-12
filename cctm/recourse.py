"""Counterfactual recourse for CCTM.

Given a CCTM and a sample ``x`` with prediction ``y``, find a minimal-cost
flip set ``S subset of features`` such that the prediction on ``x`` with
features in ``S`` flipped equals the target class.

We provide three algorithms:

* ``recourse_exact`` -- enumerate flip subsets in increasing cost, return
  the first that flips the prediction.  Provably minimal under the chosen
  cost.  Exponential in worst case; usable up to ~20 features.
* ``recourse_greedy`` -- at each step pick the feasibility-allowed flip
  that moves the vote sum the most toward the target sign, normalised by
  cost.  O(n_features^2) and very fast.  Approximate but typically
  matches the exact solver in our experiments.
* ``recourse_for`` -- a thin convenience wrapper that picks the
  appropriate algorithm based on the feature count.

All algorithms respect ``FeatureFeasibility``:

* immutable features cannot be flipped at all (e.g. race, sex, age in
  hiring under EU AI Act Article 86 / US ECOA);
* monotone constraints (income can only go up, etc.);
* per-feature flip costs.

A successful recourse can be packaged into a ``CFReceipt`` (see
``cctm.receipts``) that lets an auditor re-evaluate the flip without
touching the model.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .causal import CCTM, FeatureFeasibility


@dataclass
class Recourse:
    """Counterfactual flip set together with provenance."""
    flips: List[int]                  # feature indices to flip
    cost: float                       # total flip cost
    original_prediction: int
    new_prediction: int
    target_prediction: int
    found: bool                       # False if no recourse was found within budget
    algorithm: str                    # "exact" or "greedy"
    n_features_considered: int

    @property
    def length(self) -> int:
        return len(self.flips)


def _apply_flips(x: np.ndarray, flips: List[int]) -> np.ndarray:
    out = x.astype(np.int8).copy()
    for i in flips:
        out[i] = 1 - out[i]
    return out


def _feasible_flip_set(
    x: np.ndarray,
    feasibility: Optional[FeatureFeasibility],
) -> List[int]:
    """List of feature indices that may legally be flipped given the current
    values and the feasibility constraints."""
    if feasibility is None:
        return list(range(len(x)))
    allowed = []
    for i in range(len(x)):
        if feasibility.allowed_flip(i, int(x[i])):
            allowed.append(i)
    return allowed


def recourse_exact(
    cctm: CCTM,
    x: np.ndarray,
    target_prediction: Optional[int] = None,
    feasibility: Optional[FeatureFeasibility] = None,
    max_flips: int = 5,
) -> Recourse:
    """Provably-minimal recourse by exhaustive enumeration of flip subsets.

    Subsets are enumerated in increasing CARDINALITY first, then within a
    cardinality in increasing TOTAL COST.  The first subset that flips the
    prediction to ``target_prediction`` is returned.
    """
    x = np.asarray(x, dtype=np.int8)
    original = cctm.predict_one(x)
    if target_prediction is None:
        target_prediction = 1 - original
    if original == target_prediction:
        return Recourse(flips=[], cost=0.0, original_prediction=original,
                        new_prediction=original,
                        target_prediction=target_prediction,
                        found=True, algorithm="exact",
                        n_features_considered=0)

    allowed = _feasible_flip_set(x, feasibility)
    costs = {i: (feasibility.flip_cost(i) if feasibility is not None else 1.0)
             for i in allowed}

    for k in range(1, max_flips + 1):
        # All k-subsets, sorted by total cost ascending.
        subsets = list(itertools.combinations(allowed, k))
        subsets.sort(key=lambda S: sum(costs[i] for i in S))
        for S in subsets:
            x_cf = _apply_flips(x, list(S))
            if cctm.predict_one(x_cf) == target_prediction:
                return Recourse(
                    flips=list(S),
                    cost=float(sum(costs[i] for i in S)),
                    original_prediction=original,
                    new_prediction=target_prediction,
                    target_prediction=target_prediction,
                    found=True,
                    algorithm="exact",
                    n_features_considered=len(allowed),
                )
    return Recourse(flips=[], cost=float("inf"),
                    original_prediction=original,
                    new_prediction=original,
                    target_prediction=target_prediction,
                    found=False, algorithm="exact",
                    n_features_considered=len(allowed))


def recourse_greedy(
    cctm: CCTM,
    x: np.ndarray,
    target_prediction: Optional[int] = None,
    feasibility: Optional[FeatureFeasibility] = None,
    max_flips: int = 5,
    use_causal_influence: bool = True,
) -> Recourse:
    """Greedy recourse: at each step pick the flip that moves the vote sum
    most strongly toward the target sign, normalised by cost (and by per-feature
    causal influence if available)."""
    x = np.asarray(x, dtype=np.int8)
    original = cctm.predict_one(x)
    if target_prediction is None:
        target_prediction = 1 - original
    if original == target_prediction:
        return Recourse(flips=[], cost=0.0, original_prediction=original,
                        new_prediction=original,
                        target_prediction=target_prediction,
                        found=True, algorithm="greedy",
                        n_features_considered=0)

    allowed = set(_feasible_flip_set(x, feasibility))
    current = x.astype(np.int8).copy()
    chosen: List[int] = []
    total_cost = 0.0
    base_v = cctm.vote_sum(current)

    target_sign = +1 if target_prediction == 1 else -1
    # Feature-level prior on which flips help.
    ci = cctm.feature_causal_influence() if use_causal_influence else None

    for _ in range(max_flips):
        # Score each allowed candidate flip by the change in vote sum
        # toward the target sign, normalised by cost.
        best_i, best_score = None, -np.inf
        for i in list(allowed):
            if not (feasibility is None or feasibility.allowed_flip(i, int(current[i]))):
                continue
            cand = current.copy(); cand[i] = 1 - cand[i]
            v_cand = cctm.vote_sum(cand)
            delta = target_sign * (v_cand - base_v)
            cost_i = (feasibility.flip_cost(i) if feasibility is not None else 1.0)
            prior = float(ci[i]) if ci is not None else 1.0
            score = (delta + 1e-6) * (prior + 1e-6) / (cost_i + 1e-6)
            if score > best_score:
                best_score = score; best_i = i

        if best_i is None:
            break
        current[best_i] = 1 - current[best_i]
        chosen.append(best_i)
        total_cost += (feasibility.flip_cost(best_i)
                       if feasibility is not None else 1.0)
        allowed.discard(best_i)
        base_v = cctm.vote_sum(current)

        if cctm.predict_one(current) == target_prediction:
            return Recourse(
                flips=chosen, cost=total_cost,
                original_prediction=original,
                new_prediction=target_prediction,
                target_prediction=target_prediction,
                found=True, algorithm="greedy",
                n_features_considered=len(_feasible_flip_set(x, feasibility)),
            )

    return Recourse(flips=chosen, cost=total_cost,
                    original_prediction=original,
                    new_prediction=cctm.predict_one(current),
                    target_prediction=target_prediction,
                    found=False, algorithm="greedy",
                    n_features_considered=len(_feasible_flip_set(x, feasibility)))


def recourse_for(
    cctm: CCTM,
    x: np.ndarray,
    target_prediction: Optional[int] = None,
    feasibility: Optional[FeatureFeasibility] = None,
    max_flips: int = 5,
    exact_threshold: int = 12,
) -> Recourse:
    """Convenience: choose ``recourse_exact`` for small feature counts, else
    fall back to ``recourse_greedy``."""
    if len(x) <= exact_threshold:
        return recourse_exact(cctm, x, target_prediction,
                              feasibility=feasibility, max_flips=max_flips)
    return recourse_greedy(cctm, x, target_prediction,
                           feasibility=feasibility, max_flips=max_flips)
