"""Counterfactual SAT receipts for CCTM.

Every recourse produced by CCTM is packaged into a self-contained certificate
that an auditor can re-evaluate without access to the model state.  The
receipt records:

* the original feature vector ``x``;
* the prediction ``y`` and the per-clause inclusion mask of the firing
  clauses at the time of prediction;
* the recourse flip set ``S``;
* the new prediction ``y'``;
* (optional) HMAC-SHA-256 signature over the canonical JSON;
* (optional) feasibility metadata identifying which features were
  flagged immutable / monotone, so the auditor can enforce policy
  independently.

Verification simply re-evaluates the per-clause CNF:  for every recorded
firing clause, every included literal in ``x`` must be 1; the AND of the
included literals must equal the recorded output; and the same procedure on
``x`` with the flip set applied must yield the new prediction.

The verification path does NOT touch the CCTM model -- it works on
literal-level inclusion masks only.  This is critical for the EU AI Act
Article 86 use case: the explanation must remain valid even if the
underlying model is later updated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClauseSnapshot:
    """Frozen snapshot of a clause at receipt time."""
    clause_id: int
    polarity: int
    includes: List[int]                 # literal indices the clause includes
    output_on_x: int
    output_on_x_cf: int

    def evaluate_on(self, literals: List[int]) -> int:
        if not self.includes:
            return 0                    # empty clause -> predict-mode = 0
        for i in self.includes:
            if not literals[i]:
                return 0
        return 1


@dataclass
class CFReceipt:
    """Counterfactual receipt.

    ``x`` and ``x_cf`` are stored as flat 0/1 lists.  ``vote_sum`` and
    ``vote_sum_cf`` are the integer vote totals.  ``immutable`` /
    ``monotone_up`` / ``monotone_down`` echo the feasibility metadata so an
    auditor can verify the recourse stayed within policy.
    """
    x: List[int]
    x_cf: List[int]
    n_features: int
    flips: List[int]
    cost: float
    original_prediction: int
    new_prediction: int
    target_prediction: int
    vote_sum_x: int
    vote_sum_xcf: int
    clauses: List[ClauseSnapshot]
    immutable: List[int] = field(default_factory=list)
    monotone_up: List[int] = field(default_factory=list)
    monotone_down: List[int] = field(default_factory=list)
    description: str = ""
    signature: Optional[str] = None

    def canonical_bytes(self) -> bytes:
        d = asdict(self)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def sign_receipt(r: CFReceipt, key: bytes) -> CFReceipt:
    r.signature = hmac.new(key, r.canonical_bytes(), hashlib.sha256).hexdigest()
    return r


def _literals_from_x(x: List[int]) -> List[int]:
    return list(x) + [1 - v for v in x]


def verify_receipt(
    r: CFReceipt,
    key: Optional[bytes] = None,
) -> Tuple[bool, dict]:
    """Verify a CF receipt.  Returns (ok, info).

    Checks:
        * every clause's stored output_on_x matches the AND of its included
          literals on ``x``;
        * every clause's stored output_on_x_cf matches the AND on ``x_cf``;
        * the per-clause polarity-weighted sums match the stored ``vote_sum_*``;
        * the new_prediction is consistent with vote_sum_xcf (positive sum
          -> prediction 1, otherwise 0);
        * the flip set respects the feasibility metadata (immutable /
          monotone_up / monotone_down);
        * (optional) HMAC signature.
    """
    info: dict = {"clause_ok": [], "feasibility_violations": []}

    lits_x = _literals_from_x(r.x)
    lits_xcf = _literals_from_x(r.x_cf)

    # Clause-by-clause replay.
    v_x = 0
    v_xcf = 0
    for cs in r.clauses:
        ox = cs.evaluate_on(lits_x)
        ocf = cs.evaluate_on(lits_xcf)
        ok = (ox == cs.output_on_x) and (ocf == cs.output_on_x_cf)
        info["clause_ok"].append(ok)
        v_x += cs.polarity * ox
        v_xcf += cs.polarity * ocf

    votes_consistent = (v_x == r.vote_sum_x and v_xcf == r.vote_sum_xcf)
    pred_consistent = (r.new_prediction == (1 if r.vote_sum_xcf > 0 else 0)
                       and r.original_prediction == (1 if r.vote_sum_x > 0 else 0))

    # Feasibility check on the flip set.
    for i in r.flips:
        if i in r.immutable:
            info["feasibility_violations"].append(
                f"flip {i} violates immutability")
        if i in r.monotone_up and r.x[i] == 1:
            info["feasibility_violations"].append(
                f"flip {i} violates monotone_up (1 -> 0)")
        if i in r.monotone_down and r.x[i] == 0:
            info["feasibility_violations"].append(
                f"flip {i} violates monotone_down (0 -> 1)")

    # Flip set actually realises x -> x_cf?
    realised = True
    for i in range(r.n_features):
        if i in r.flips:
            if r.x_cf[i] != 1 - r.x[i]:
                realised = False
                break
        else:
            if r.x_cf[i] != r.x[i]:
                realised = False
                break

    sig_ok = True
    if r.signature is not None:
        if key is None:
            sig_ok = False
        else:
            expected = hmac.new(key, r.canonical_bytes(),
                                hashlib.sha256).hexdigest()
            sig_ok = hmac.compare_digest(expected, r.signature)

    ok = (all(info["clause_ok"])
          and votes_consistent
          and pred_consistent
          and realised
          and not info["feasibility_violations"]
          and sig_ok)

    info.update({
        "votes_consistent": votes_consistent,
        "pred_consistent": pred_consistent,
        "flip_set_realises_xcf": realised,
        "signature_ok": sig_ok,
    })
    return ok, info


def build_cf_receipt(
    cctm,
    x: np.ndarray,
    recourse,
    feasibility=None,
    description: str = "",
    key: Optional[bytes] = None,
) -> CFReceipt:
    """Materialise a receipt from a CCTM, a sample, and a Recourse object."""
    x = np.asarray(x, dtype=np.int8)
    x_cf = x.copy()
    for i in recourse.flips:
        x_cf[i] = 1 - x_cf[i]

    lits_x = np.concatenate([x, 1 - x])
    lits_xcf = np.concatenate([x_cf, 1 - x_cf])
    co_x = cctm.tm.bank.clause_outputs(lits_x, predict=True)
    co_xcf = cctm.tm.bank.clause_outputs(lits_xcf, predict=True)
    acts = cctm.tm.bank.actions()

    snaps: List[ClauseSnapshot] = []
    v_x = 0; v_xcf = 0
    for j in range(cctm.tm.cfg.n_clauses):
        includes = [int(i) for i in np.where(acts[j])[0]]
        cs = ClauseSnapshot(
            clause_id=j,
            polarity=int(cctm.tm.polarity[j]),
            includes=includes,
            output_on_x=int(co_x[j]),
            output_on_x_cf=int(co_xcf[j]),
        )
        snaps.append(cs)
        v_x += cs.polarity * cs.output_on_x
        v_xcf += cs.polarity * cs.output_on_x_cf

    receipt = CFReceipt(
        x=[int(v) for v in x],
        x_cf=[int(v) for v in x_cf],
        n_features=int(len(x)),
        flips=list(recourse.flips),
        cost=float(recourse.cost),
        original_prediction=int(recourse.original_prediction),
        new_prediction=int(recourse.new_prediction),
        target_prediction=int(recourse.target_prediction),
        vote_sum_x=int(v_x),
        vote_sum_xcf=int(v_xcf),
        clauses=snaps,
        immutable=list(feasibility.immutable) if feasibility else [],
        monotone_up=list(feasibility.monotone_up) if feasibility else [],
        monotone_down=list(feasibility.monotone_down) if feasibility else [],
        description=description,
    )
    if key is not None:
        sign_receipt(receipt, key)
    return receipt
