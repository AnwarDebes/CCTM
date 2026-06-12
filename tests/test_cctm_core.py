"""Tests for CCTM core (causal training, causal score tracking)."""

import numpy as np
import pytest

from cctm import (
    CCTM, CCTMConfig, TMConfig, TsetlinMachine,
    generate_cf_pairs_from_oracle,
)


def make_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int8)
    return X, y


def test_cctm_without_cf_degrades_to_vanilla_tm():
    X, y = make_xor(n=400, seed=0)
    cfg = CCTMConfig(n_clauses=20, n_features=2, threshold=10, s=3.0, seed=42)
    m = CCTM(cfg).fit(X, y, epochs=15)
    acc = m.score(X, y)
    assert acc >= 0.95, f"CCTM without CF pairs should match vanilla TM; got {acc}"
    # No CF pairs seen -> causal_score remains all zero.
    assert m.causal_count == 0
    assert m.causal_score.sum() == 0.0


def test_cctm_with_cf_pairs_learns_causal_score():
    rng = np.random.default_rng(0)
    # Three features; only feature 0 is causal (y = x0).
    n = 600
    X = rng.integers(0, 2, size=(n, 3)).astype(np.int8)
    y = X[:, 0].astype(np.int8)
    # Oracle: ground truth label
    def oracle(x_row): return int(x_row[0])
    X_cf, y_cf = generate_cf_pairs_from_oracle(
        X, oracle, rng=rng, flip_strategy="single",
    )
    cfg = CCTMConfig(n_clauses=20, n_features=3, threshold=10, s=3.0, seed=42)
    m = CCTM(cfg).fit(X, y, X_cf=X_cf, y_cf=y_cf, epochs=10)
    acc = m.score(X, y)
    assert acc >= 0.90, f"CCTM should still learn the label; got {acc}"
    # Feature causal influence should be highest on feature 0.
    ci = m.feature_causal_influence()
    assert ci[0] > ci[1], f"Feature 0 should be most causal; got CI={ci}"
    assert ci[0] > ci[2], f"Feature 0 should be most causal; got CI={ci}"


def test_causal_score_zero_when_no_label_flip_pairs():
    """If every CF pair has y == y_cf, causal_score must NEVER update."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.integers(0, 2, size=(n, 3)).astype(np.int8)
    y = X[:, 0].astype(np.int8)
    X_cf = X.copy()
    y_cf = y.copy()                                 # no label flips
    cfg = CCTMConfig(n_clauses=20, n_features=3, threshold=10, s=3.0, seed=42)
    m = CCTM(cfg).fit(X, y, X_cf=X_cf, y_cf=y_cf, epochs=5)
    assert m.causal_count == 0
    assert m.causal_score.sum() == 0.0


def test_cctm_predict_matches_tm():
    """Sanity: CCTM.predict must equal the underlying TM's predict."""
    X, y = make_xor(n=200, seed=1)
    cfg = CCTMConfig(n_clauses=20, n_features=2, threshold=10, s=3.0, seed=7)
    m = CCTM(cfg).fit(X, y, epochs=5)
    # Predict via CCTM
    preds_cctm = m.predict(X)
    # Predict via underlying TM directly
    preds_tm = m.tm.predict(X)
    assert np.array_equal(preds_cctm, preds_tm)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
