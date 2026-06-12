"""Fairness-audit experiment.

Uses the bundled loan-denial dataset with a tunable demographic bias toward
``gender_male``.  Asks two questions:

1.  Does CCTM's per-feature treatment-effect estimator correctly flag
    ``gender_male`` as having a non-zero causal effect on the prediction
    (which would be a fairness flag) even though the true *underlying*
    rule does not use gender?
2.  When we MUTE the most biased clauses (those with high causal-score AND
    high gender-inclusion), can we restore demographic parity at small
    accuracy cost?

This is the experiment that singles CCTM out: it provides causal-effect
signal that a vanilla TM does not produce, and it shows that the signal is
actionable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from cctm import (
    CCTM, CCTMConfig, TMConfig, TsetlinMachine,
)
from cctm.causal import generate_cf_pairs_from_oracle
from cctm.data import generate_loan_dataset, split_train_test


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def demographic_parity(model, X, cols) -> float:
    """|P(predict=1 | gender_male=1) - P(predict=1 | gender_male=0)|"""
    male = X[:, cols.gender_male] == 1
    female = X[:, cols.gender_male] == 0
    if not male.any() or not female.any():
        return 0.0
    pm = model.predict(X[male]).mean()
    pf = model.predict(X[female]).mean()
    return float(abs(pm - pf))


def equal_opportunity_gap(model, X, y, cols) -> float:
    """|P(predict=1 | y=1, male) - P(predict=1 | y=1, female)|"""
    male_pos = (X[:, cols.gender_male] == 1) & (y == 1)
    female_pos = (X[:, cols.gender_male] == 0) & (y == 1)
    if not male_pos.any() or not female_pos.any():
        return 0.0
    pm = model.predict(X[male_pos]).mean()
    pf = model.predict(X[female_pos]).mean()
    return float(abs(pm - pf))


def find_biased_clauses(cctm, cols, threshold=0.05) -> np.ndarray:
    """Return a Boolean mask over clauses identifying 'biased' clauses:
    those that include the gender_male literal (raw or negated) AND have
    non-trivial causal score."""
    acts = cctm.tm.bank.actions()                          # (C, 2L)
    L = cctm.cfg.n_features
    inc_g = acts[:, cols.gender_male] | acts[:, cols.gender_male + L]
    return inc_g & (cctm.causal_score > threshold)


def mute_clauses(cctm, mask):
    """Disable clauses identified by ``mask`` by forcing them to all-exclude
    (which makes their output 0 at predict time)."""
    cctm.tm.bank.state[mask, :] = cctm.tm.bank.N           # exclude boundary


def run_seed(seed: int, n: int = 2500, bias_strength: float = 0.30, epochs: int = 12):
    rng = np.random.default_rng(seed)
    X, y, cols, oracle_unbiased = generate_loan_dataset(
        n=n, seed=seed, bias_strength=bias_strength,
    )
    # Oracle ground truth is BIAS-FREE: even though the data we train on is
    # biased, the true decision rule doesn't use gender.
    _, _, _, oracle_clean = generate_loan_dataset(
        n=1, seed=seed, bias_strength=0.0,
    )

    Xtr, ytr, Xte, yte = split_train_test(X, y, seed=seed)
    y_clean_te = np.array([oracle_clean(Xte[i]) for i in range(Xte.shape[0])],
                          dtype=np.int8)

    # Vanilla TM (no CF), trained on biased data
    vanilla = TsetlinMachine(TMConfig(n_clauses=80, n_features=cols.n_features,
                                      threshold=20, s=3.9, seed=seed))
    vanilla.fit(Xtr, ytr, epochs=epochs)
    acc_vanilla = vanilla.score(Xte, y_clean_te)
    dp_vanilla = demographic_parity(vanilla, Xte, cols)
    eo_vanilla = equal_opportunity_gap(vanilla, Xte, y_clean_te, cols)

    # CCTM with CF pairs.  We use the CLEAN oracle for CF labels -- this
    # mirrors the deployment scenario where we know individual features'
    # causal effect on the *ideal* decision (e.g., by querying a
    # human-in-the-loop reviewer or a regulatory specification).
    Xcf, ycf = generate_cf_pairs_from_oracle(
        Xtr, oracle_clean, rng=rng, flip_strategy="single",
    )
    cctm = CCTM(CCTMConfig(n_clauses=80, n_features=cols.n_features,
                            threshold=20, s=3.9, seed=seed))
    cctm.fit(Xtr, ytr, X_cf=Xcf, y_cf=ycf, epochs=epochs)
    acc_cctm = cctm.score(Xte, y_clean_te)
    dp_cctm = demographic_parity(cctm, Xte, cols)
    eo_cctm = equal_opportunity_gap(cctm, Xte, y_clean_te, cols)

    # The TE on gender_male should be near 0 (it is not causal under the
    # clean rule).  The dataset bias may push it UP for the vanilla TM
    # (which has no signal to push back).
    te = cctm.feature_treatment_effect()

    # Audit: identify biased clauses in the CCTM, mute them, re-evaluate.
    n_biased = int(find_biased_clauses(cctm, cols).sum())
    audited = CCTM(CCTMConfig(n_clauses=80, n_features=cols.n_features,
                              threshold=20, s=3.9, seed=seed))
    audited.tm.bank.state = cctm.tm.bank.state.copy()
    audited.causal_score = cctm.causal_score.copy()
    audited.feature_te_num = cctm.feature_te_num.copy()
    audited.feature_te_den = cctm.feature_te_den.copy()
    audited.causal_count = cctm.causal_count
    mute_clauses(audited, find_biased_clauses(audited, cols))
    acc_audited = audited.score(Xte, y_clean_te)
    dp_audited = demographic_parity(audited, Xte, cols)
    eo_audited = equal_opportunity_gap(audited, Xte, y_clean_te, cols)

    return {
        "seed": seed,
        "bias_strength": bias_strength,
        "vanilla":  {"acc": acc_vanilla, "dp": dp_vanilla, "eo": eo_vanilla},
        "cctm":     {"acc": acc_cctm,    "dp": dp_cctm,    "eo": eo_cctm,
                     "te_gender": float(te[cols.gender_male]),
                     "n_biased_clauses": n_biased},
        "audited":  {"acc": acc_audited, "dp": dp_audited, "eo": eo_audited},
    }


def main(out_path: Path = RESULTS_DIR / "fairness_audit.json",
         seeds=(0, 1, 2), bias_strength=0.30, epochs=12):
    rows = []
    overall = time.time()
    for seed in seeds:
        r = run_seed(seed, bias_strength=bias_strength, epochs=epochs)
        rows.append(r)
        v = r["vanilla"]; c = r["cctm"]; a = r["audited"]
        print(f"seed={seed} bias={bias_strength}")
        print(f"  vanilla:  acc={v['acc']:.3f}  DP={v['dp']:.3f}  EO={v['eo']:.3f}")
        print(f"  cctm:     acc={c['acc']:.3f}  DP={c['dp']:.3f}  EO={c['eo']:.3f} "
              f"  TE(gender)={c['te_gender']:.3f}  biased_clauses={c['n_biased_clauses']}")
        print(f"  audited:  acc={a['acc']:.3f}  DP={a['dp']:.3f}  EO={a['eo']:.3f}")

    print(f"\nTotal: {time.time() - overall:.1f}s")

    def avg(key):
        vals = [r[key[0]][key[1]] for r in rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "n_seeds": len(seeds),
        "bias_strength": bias_strength,
        "vanilla_acc": avg(("vanilla", "acc")),
        "vanilla_dp":  avg(("vanilla", "dp")),
        "vanilla_eo":  avg(("vanilla", "eo")),
        "cctm_acc":    avg(("cctm",    "acc")),
        "cctm_dp":     avg(("cctm",    "dp")),
        "cctm_eo":     avg(("cctm",    "eo")),
        "cctm_te_gender": avg(("cctm", "te_gender")),
        "audited_acc": avg(("audited", "acc")),
        "audited_dp":  avg(("audited", "dp")),
        "audited_eo":  avg(("audited", "eo")),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k}: {v['mean']:.4f} +- {v['std']:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path.write_text(json.dumps({"per_seed": rows, "summary": summary},
                                   indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
