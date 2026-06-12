"""Synthetic SCM experiment: per-feature ATE recovery.

We generate from a known Structural Causal Model where the ground-truth
population Average Treatment Effect (ATE) of each binary feature is
computable in closed form by intervening on the SCM and averaging.

We train two models on the same data:

    M_vanilla: a vanilla TM (no CF pairs)
    M_cctm:    CCTM with CF pairs generated from the oracle

We compare each model's *implied* per-feature TE (the empirical
``feature_treatment_effect`` of CCTM, and for the vanilla TM we estimate
it from the model's predictions on (x, x') flipped pairs) against the
ground-truth ATE.

The headline metric is Pearson correlation between the model's TE vector
and the true ATE vector, plus the mean absolute error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cctm import CCTM, CCTMConfig, TMConfig, TsetlinMachine
from cctm.causal import generate_cf_pairs_from_oracle
from cctm.data import generate_scm_dataset, split_train_test


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def estimate_te_from_model(model, X, oracle, rng):
    """Estimate per-feature TE from a model by:
       (a) flipping each feature in turn over the test set,
       (b) measuring how often the model's prediction flips with it,
       (c) comparing against the oracle's true label flip rate (this is
           what the model would 'see' as evidence of causation if given a
           CF oracle)."""
    n, d = X.shape
    te = np.zeros(d, dtype=np.float32)
    for j in range(d):
        Xj = X.copy(); Xj[:, j] = 1 - Xj[:, j]
        y_orig = model.predict(X)
        y_flip = model.predict(Xj)
        te[j] = float(np.abs(y_flip - y_orig).mean())
    return te


def run_seed(seed: int, n: int = 1500, epochs: int = 10):
    rng = np.random.default_rng(seed)
    X, y, ATE, oracle = generate_scm_dataset(n=n, seed=seed)
    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)

    # Vanilla TM
    vanilla = TsetlinMachine(TMConfig(n_clauses=40, n_features=5,
                                      threshold=15, s=3.9, seed=seed))
    t0 = time.time()
    vanilla.fit(Xtr, ytr, epochs=epochs)
    t_vanilla = time.time() - t0
    acc_vanilla = vanilla.score(Xte, yte)
    te_vanilla = estimate_te_from_model(vanilla, Xte, oracle, rng)

    # CCTM with CF pairs
    Xcf, ycf = generate_cf_pairs_from_oracle(
        Xtr, oracle, rng=rng, flip_strategy="single",
    )
    cctm = CCTM(CCTMConfig(n_clauses=40, n_features=5,
                            threshold=15, s=3.9, seed=seed))
    t0 = time.time()
    cctm.fit(Xtr, ytr, X_cf=Xcf, y_cf=ycf, epochs=epochs)
    t_cctm = time.time() - t0
    acc_cctm = cctm.score(Xte, yte)
    te_cctm_empirical = cctm.feature_treatment_effect()      # direct from CF stats
    te_cctm_predicted = estimate_te_from_model(cctm, Xte, oracle, rng)

    def metrics(estimate):
        return {
            "pearson": float(np.corrcoef(estimate, np.abs(ATE))[0, 1]),
            "mae":     float(np.mean(np.abs(estimate - np.abs(ATE)))),
            "values":  [float(v) for v in estimate],
        }

    return {
        "seed": seed,
        "ATE_ground_truth": [float(v) for v in np.abs(ATE)],
        "vanilla_tm": {
            "test_acc": acc_vanilla,
            "train_time_s": t_vanilla,
            "te_from_predictions": metrics(te_vanilla),
        },
        "cctm": {
            "test_acc": acc_cctm,
            "train_time_s": t_cctm,
            "te_empirical": metrics(te_cctm_empirical),
            "te_from_predictions": metrics(te_cctm_predicted),
        },
    }


def main(out_path: Path = RESULTS_DIR / "synthetic_scm.json",
         seeds=(0, 1, 2, 3, 4), n=1500, epochs=10):
    all_rows = []
    overall = time.time()
    for seed in seeds:
        row = run_seed(seed, n=n, epochs=epochs)
        all_rows.append(row)
        print(f"seed={seed}  "
              f"vanilla acc={row['vanilla_tm']['test_acc']:.3f}  "
              f"CCTM acc={row['cctm']['test_acc']:.3f}  "
              f"CCTM TE Pearson (empirical)={row['cctm']['te_empirical']['pearson']:.3f}  "
              f"vanilla TE Pearson (pred)={row['vanilla_tm']['te_from_predictions']['pearson']:.3f}")
    print(f"\nTotal: {time.time() - overall:.1f}s")

    # Summary
    summary = {
        "n_seeds": len(seeds),
        "vanilla_tm_test_acc_mean": float(np.mean([r["vanilla_tm"]["test_acc"] for r in all_rows])),
        "cctm_test_acc_mean":       float(np.mean([r["cctm"]["test_acc"] for r in all_rows])),
        "cctm_te_empirical_pearson_mean": float(np.mean([r["cctm"]["te_empirical"]["pearson"] for r in all_rows])),
        "cctm_te_empirical_mae_mean":     float(np.mean([r["cctm"]["te_empirical"]["mae"] for r in all_rows])),
        "vanilla_te_from_pred_pearson_mean": float(np.mean([r["vanilla_tm"]["te_from_predictions"]["pearson"] for r in all_rows])),
        "vanilla_te_from_pred_mae_mean":     float(np.mean([r["vanilla_tm"]["te_from_predictions"]["mae"] for r in all_rows])),
        "cctm_te_from_pred_pearson_mean": float(np.mean([r["cctm"]["te_from_predictions"]["pearson"] for r in all_rows])),
        "cctm_te_from_pred_mae_mean":     float(np.mean([r["cctm"]["te_from_predictions"]["mae"] for r in all_rows])),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    out_path.write_text(json.dumps({"per_seed": all_rows, "summary": summary},
                                   indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
