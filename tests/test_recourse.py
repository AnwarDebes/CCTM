"""Tests for the counterfactual recourse module."""

import numpy as np
import pytest

from cctm import (
    CCTM, CCTMConfig, FeatureFeasibility,
)
from cctm.recourse import (
    Recourse, recourse_exact, recourse_greedy, recourse_for,
)


def fit_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int8)
    cfg = CCTMConfig(n_clauses=20, n_features=2, threshold=10, s=3.0, seed=42)
    return CCTM(cfg).fit(X, y, epochs=15), X, y


def test_recourse_finds_xor_flip():
    m, X, y = fit_xor()
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], max_flips=2)
        if r.found:
            x_cf = X[i].copy()
            for j in r.flips:
                x_cf[j] = 1 - x_cf[j]
            assert m.predict_one(x_cf) == r.new_prediction
            assert r.new_prediction != r.original_prediction
            assert r.length <= 2


def test_recourse_respects_immutable():
    m, X, y = fit_xor()
    fz = FeatureFeasibility(immutable=[0])
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], feasibility=fz, max_flips=2)
        # If a recourse is found, it MUST not flip feature 0.
        if r.found:
            assert 0 not in r.flips


def test_greedy_matches_exact_on_small_xor():
    m, X, y = fit_xor()
    matches = 0; total = 0
    for i in range(X.shape[0]):
        e = recourse_exact(m, X[i], max_flips=2)
        g = recourse_greedy(m, X[i], max_flips=2)
        if e.found and g.found:
            total += 1
            if g.length == e.length:
                matches += 1
    # Greedy should very often match exact on this simple problem.
    if total > 0:
        assert matches / total >= 0.95, (
            f"Greedy matched {matches}/{total} = {matches/total:.3f} -- "
            "expected >= 0.95 on XOR"
        )


def test_recourse_no_op_when_already_at_target():
    m, X, y = fit_xor()
    pred = m.predict_one(X[0])
    r = recourse_exact(m, X[0], target_prediction=pred)
    assert r.found
    assert r.flips == []
    assert r.length == 0
    assert r.cost == 0.0


def test_recourse_for_dispatches():
    m, X, y = fit_xor()
    r = recourse_for(m, X[0], exact_threshold=12)
    assert isinstance(r, Recourse)
    assert r.algorithm == "exact"


def test_recourse_cost_aware():
    m, X, y = fit_xor()
    # Make feature 0 very expensive, feature 1 cheap.
    fz = FeatureFeasibility(cost=[100.0, 1.0])
    # Find any sample where a recourse exists and exactly one of {0, 1} works.
    found = False
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], feasibility=fz, max_flips=2)
        if r.found and r.length == 1:
            # Should prefer the cheaper flip if BOTH single flips work.
            x = X[i]
            both_work_a = (m.predict_one(np.array([1 - x[0], x[1]], dtype=np.int8))
                            != m.predict_one(x))
            both_work_b = (m.predict_one(np.array([x[0], 1 - x[1]], dtype=np.int8))
                            != m.predict_one(x))
            if both_work_a and both_work_b:
                assert r.flips == [1], f"Expected cheap flip [1], got {r.flips}"
                found = True
                break
    # Don't fail if no test case existed, but emit a soft note.
    if not found:
        pytest.skip("No sample where both single flips would solve recourse.")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
