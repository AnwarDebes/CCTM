"""Tests for the CF receipt module."""

import numpy as np
import pytest

from cctm import (
    CCTM, CCTMConfig, FeatureFeasibility,
    recourse_exact, build_cf_receipt, verify_receipt, sign_receipt,
)


def fit_xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int8)
    cfg = CCTMConfig(n_clauses=20, n_features=2, threshold=10, s=3.0, seed=42)
    return CCTM(cfg).fit(X, y, epochs=15), X, y


def test_receipt_round_trip():
    m, X, y = fit_xor()
    for i in range(min(10, X.shape[0])):
        r = recourse_exact(m, X[i], max_flips=2)
        if not r.found:
            continue
        receipt = build_cf_receipt(m, X[i], r)
        ok, info = verify_receipt(receipt)
        assert ok, f"verify failed: {info}"


def test_receipt_tampering_detected():
    m, X, y = fit_xor()
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], max_flips=2)
        if not r.found:
            continue
        receipt = build_cf_receipt(m, X[i], r)
        # Flip a recorded clause output -> verification must fail
        receipt.clauses[0].output_on_x = 1 - receipt.clauses[0].output_on_x
        ok, info = verify_receipt(receipt)
        assert not ok
        return  # one is enough
    pytest.skip("no successful recourse to tamper")


def test_receipt_signing_round_trip():
    m, X, y = fit_xor()
    key = b"test-secret-key-1234567890"
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], max_flips=2)
        if not r.found:
            continue
        receipt = build_cf_receipt(m, X[i], r, key=key)
        ok, info = verify_receipt(receipt, key=key)
        assert ok
        # Wrong key -> signature mismatch.
        ok2, info2 = verify_receipt(receipt, key=b"different-key-asdf")
        assert not ok2
        assert not info2["signature_ok"]
        return
    pytest.skip("no successful recourse to sign")


def test_receipt_feasibility_violation_detected():
    m, X, y = fit_xor()
    # Hand-build a receipt with a flip that violates immutability.
    for i in range(X.shape[0]):
        r = recourse_exact(m, X[i], max_flips=2)
        if not r.found or 0 not in r.flips:
            continue
        # Add immutable=[0] to the receipt after the fact: should flag
        # the recorded flip 0 as a violation.
        fz = FeatureFeasibility(immutable=[0])
        receipt = build_cf_receipt(m, X[i], r, feasibility=fz)
        ok, info = verify_receipt(receipt)
        assert not ok
        assert any("immutab" in v.lower() for v in info["feasibility_violations"])
        return
    pytest.skip("no recourse touching feature 0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
