"""Loan-denial recourse experiment.

Where structural recourse earns its keep.

Setup
-----
We train two models on the bundled synthetic loan-denial dataset:

    M_vanilla: a vanilla TM (no CF pairs)
    M_cctm:    CCTM with CF pairs generated from the oracle

For every test sample that the model predicts as "deny" (label 0), we
attempt to find a counterfactual recourse: the smallest, cheapest set of
feasible feature flips that would turn the decision into "approve" (1).

Feasibility constraints (mirror EU AI Act Article 86 / US ECOA practice):

* ``gender_male`` and ``citizen`` are IMMUTABLE (the law forbids using
  them as recourse).
* ``age_over_25`` is MONOTONE-UP (one cannot un-age).
* ``age_over_45`` is MONOTONE-UP.
* All other features are mutable at unit cost.

We report:

* ``success_rate``: fraction of denied samples for which a recourse
  exists within the budget (max_flips=3).
* ``mean_length``: mean number of flips in successful recourses.
* ``mean_cost``: mean total cost.
* ``mean_latency_ms``: time per recourse search (in milliseconds).

The contribution of CCTM is the *causal-influence-guided* greedy search,
which the experiment isolates by also running an exact baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cctm import (
    CCTM, CCTMConfig, TMConfig, TsetlinMachine, FeatureFeasibility,
    recourse_exact, recourse_greedy, build_cf_receipt, verify_receipt,
)
from cctm.causal import generate_cf_pairs_from_oracle
from cctm.data import generate_loan_dataset, split_train_test


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def loan_feasibility(cols):
    return FeatureFeasibility(
        immutable=[cols.gender_male, cols.citizen],
        monotone_up=[cols.age_over_25, cols.age_over_45],
        cost=[1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 0.0, 0.0],
    )


def evaluate_recourse(model, X, feasibility, label_for_recourse=0,
                      target=1, max_flips=3, algorithm="exact"):
    """Iterate over samples predicted as ``label_for_recourse``; attempt
    to find a recourse that flips them to ``target``.  Returns aggregated
    metrics + raw per-sample records."""
    if algorithm == "exact":
        rfn = recourse_exact
    elif algorithm == "greedy":
        rfn = recourse_greedy
    else:
        raise ValueError(algorithm)
    rows = []
    for i in range(X.shape[0]):
        if model.predict_one(X[i]) != label_for_recourse:
            continue
        t0 = time.time()
        r = rfn(model, X[i], feasibility=feasibility, max_flips=max_flips,
                target_prediction=target)
        elapsed_ms = (time.time() - t0) * 1000.0
        rows.append({
            "found": r.found,
            "length": r.length if r.found else None,
            "cost": r.cost if r.found else None,
            "latency_ms": elapsed_ms,
        })
    if not rows:
        return {"n_attempts": 0}, []
    success_rate = float(np.mean([r["found"] for r in rows]))
    lens = np.array([r["length"] for r in rows if r["found"]], dtype=np.float32)
    costs = np.array([r["cost"] for r in rows if r["found"]], dtype=np.float32)
    lats = np.array([r["latency_ms"] for r in rows], dtype=np.float32)
    return {
        "n_attempts": len(rows),
        "success_rate": success_rate,
        "mean_length": float(lens.mean()) if lens.size else None,
        "mean_cost": float(costs.mean()) if costs.size else None,
        "mean_latency_ms": float(lats.mean()),
        "p95_latency_ms": float(np.percentile(lats, 95)),
    }, rows


def run_seed(seed: int, n: int = 2000, epochs: int = 12):
    rng = np.random.default_rng(seed)
    X, y, cols, oracle = generate_loan_dataset(n=n, seed=seed,
                                               bias_strength=0.0)
    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)
    fz = loan_feasibility(cols)

    # Vanilla TM
    vanilla = TsetlinMachine(TMConfig(n_clauses=80, n_features=cols.n_features,
                                      threshold=20, s=3.9, seed=seed))
    vanilla.fit(Xtr, ytr, epochs=epochs)

    # CCTM with CF pairs (single-flip)
    Xcf, ycf = generate_cf_pairs_from_oracle(
        Xtr, oracle, rng=rng, flip_strategy="single",
    )
    cctm = CCTM(CCTMConfig(n_clauses=80, n_features=cols.n_features,
                            threshold=20, s=3.9, seed=seed))
    cctm.fit(Xtr, ytr, X_cf=Xcf, y_cf=ycf, epochs=epochs)

    acc_vanilla = vanilla.score(Xte, yte)
    acc_cctm = cctm.score(Xte, yte)

    # Recourse evaluation.
    # Vanilla TM uses recourse_exact too (since CCTM's CI guide is
    # unavailable to it).
    class TMShim:
        """Provide CCTM-like interface for vanilla TM so recourse_* works."""
        def __init__(self, tm):
            self.tm = tm
            self.cfg_n_features = tm.cfg.n_features
        def predict_one(self, x):
            lits = np.concatenate([x.astype(np.int8), 1 - x.astype(np.int8)])
            return self.tm.predict_one(lits)
        def vote_sum(self, x):
            lits = np.concatenate([x.astype(np.int8), 1 - x.astype(np.int8)])
            v, _ = self.tm.vote_sum(lits, predict=True)
            return int(v)
        def feature_causal_influence(self):
            # uniform "no causal information"
            return np.ones(self.cfg_n_features, dtype=np.float32)

    vanilla_shim = TMShim(vanilla)

    # Exact recourse on both models
    res_vanilla_exact, _ = evaluate_recourse(
        vanilla_shim, Xte, feasibility=fz, max_flips=3, algorithm="exact",
    )
    res_cctm_exact, cctm_exact_records = evaluate_recourse(
        cctm, Xte, feasibility=fz, max_flips=3, algorithm="exact",
    )
    # Greedy recourse on both -- the CCTM uses CI, the shim doesn't.
    res_vanilla_greedy, _ = evaluate_recourse(
        vanilla_shim, Xte, feasibility=fz, max_flips=3, algorithm="greedy",
    )
    res_cctm_greedy, _ = evaluate_recourse(
        cctm, Xte, feasibility=fz, max_flips=3, algorithm="greedy",
    )

    # Receipt-verification sanity check on the first 20 successful CCTM
    # exact recourses.
    receipts_ok = 0; receipts_tested = 0
    for rec in cctm_exact_records[:50]:
        if rec["found"]:
            i = next(idx for idx in range(Xte.shape[0])
                     if cctm.predict_one(Xte[idx]) == 0
                     and idx >= receipts_tested)
            from cctm.recourse import recourse_exact as _rx
            r = _rx(cctm, Xte[i], feasibility=fz, max_flips=3)
            if r.found:
                receipt = build_cf_receipt(cctm, Xte[i], r, feasibility=fz)
                ok, _info = verify_receipt(receipt)
                if ok:
                    receipts_ok += 1
                receipts_tested += 1
        if receipts_tested >= 20:
            break

    return {
        "seed": seed,
        "vanilla_tm": {"test_acc": acc_vanilla,
                       "exact": res_vanilla_exact,
                       "greedy": res_vanilla_greedy},
        "cctm":       {"test_acc": acc_cctm,
                       "exact": res_cctm_exact,
                       "greedy": res_cctm_greedy,
                       "receipts_ok": receipts_ok,
                       "receipts_tested": receipts_tested},
    }


def main(out_path: Path = RESULTS_DIR / "loan_recourse.json",
         seeds=(0, 1, 2), n=2000, epochs=10):
    rows = []
    overall = time.time()
    for seed in seeds:
        r = run_seed(seed, n=n, epochs=epochs)
        rows.append(r)
        ve = r["vanilla_tm"]["exact"]
        ce = r["cctm"]["exact"]
        vg = r["vanilla_tm"]["greedy"]
        cg = r["cctm"]["greedy"]
        print(f"seed={seed}  "
              f"vanilla_acc={r['vanilla_tm']['test_acc']:.3f}  "
              f"cctm_acc={r['cctm']['test_acc']:.3f}")
        print(f"           vanilla_exact: succ={ve['success_rate']:.3f} "
              f"len={ve['mean_length']} cost={ve['mean_cost']} "
              f"lat={ve['mean_latency_ms']:.2f}ms")
        print(f"           cctm_exact:    succ={ce['success_rate']:.3f} "
              f"len={ce['mean_length']} cost={ce['mean_cost']} "
              f"lat={ce['mean_latency_ms']:.2f}ms")
        print(f"           vanilla_greedy succ={vg['success_rate']:.3f} "
              f"len={vg['mean_length']} cost={vg['mean_cost']} "
              f"lat={vg['mean_latency_ms']:.2f}ms")
        print(f"           cctm_greedy    succ={cg['success_rate']:.3f} "
              f"len={cg['mean_length']} cost={cg['mean_cost']} "
              f"lat={cg['mean_latency_ms']:.2f}ms")
        print(f"           receipts:      {r['cctm']['receipts_ok']}/{r['cctm']['receipts_tested']} verified")

    print(f"\nTotal: {time.time() - overall:.1f}s")

    def aggregate(keypath):
        # keypath like ("vanilla_tm", "exact", "success_rate")
        vals = []
        for r in rows:
            x = r
            for k in keypath:
                x = x.get(k, None)
                if x is None:
                    break
            if isinstance(x, (int, float)) and x is not None:
                vals.append(float(x))
        if not vals:
            return None
        return {"mean": float(np.mean(vals)),
                "std":  float(np.std(vals))}

    summary = {
        "vanilla_test_acc": aggregate(("vanilla_tm", "test_acc")),
        "cctm_test_acc":    aggregate(("cctm", "test_acc")),
        "vanilla_exact_success":   aggregate(("vanilla_tm", "exact", "success_rate")),
        "cctm_exact_success":      aggregate(("cctm",       "exact", "success_rate")),
        "vanilla_exact_length":    aggregate(("vanilla_tm", "exact", "mean_length")),
        "cctm_exact_length":       aggregate(("cctm",       "exact", "mean_length")),
        "vanilla_exact_lat_ms":    aggregate(("vanilla_tm", "exact", "mean_latency_ms")),
        "cctm_exact_lat_ms":       aggregate(("cctm",       "exact", "mean_latency_ms")),
        "vanilla_greedy_success":  aggregate(("vanilla_tm", "greedy", "success_rate")),
        "cctm_greedy_success":     aggregate(("cctm",       "greedy", "success_rate")),
        "vanilla_greedy_length":   aggregate(("vanilla_tm", "greedy", "mean_length")),
        "cctm_greedy_length":      aggregate(("cctm",       "greedy", "mean_length")),
        "vanilla_greedy_lat_ms":   aggregate(("vanilla_tm", "greedy", "mean_latency_ms")),
        "cctm_greedy_lat_ms":      aggregate(("cctm",       "greedy", "mean_latency_ms")),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v['mean']:.4f} +- {v['std']:.4f}")

    out_path.write_text(json.dumps({"per_seed": rows, "summary": summary},
                                   indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
