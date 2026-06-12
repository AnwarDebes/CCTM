"""CCTM: Counterfactual Causal Tsetlin Machine.

A sixth member of the Tsetlin Machine family.  CCTM adds three primitives:

    * **Do-clauses** -- clauses asserted over intervened inputs.
    * **Type-III contrastive feedback** -- training from factual /
      counterfactual (x, y, x', y') pairs with per-clause causal-score
      tracking.
    * **Structural counterfactual recourse** -- minimal-flip recourse derived
      directly from the trained clause structure, with optional feature
      feasibility constraints and SAT-verifiable receipts.

CCTM targets regulatorily-mandated individual-decision explanations under
EU AI Act Article 86, GDPR Article 22 and US ECOA.

Pure-NumPy reference implementation.  Pairs naturally with HGTM and THGTM
but is independent of both.
"""

from .ta import TABank
from .tm import TsetlinMachine, TMConfig, to_literals
from .causal import (
    CCTM, CCTMConfig, FeatureFeasibility, generate_cf_pairs_from_oracle,
)
from .recourse import (
    Recourse, recourse_exact, recourse_greedy, recourse_for,
)
from .receipts import (
    CFReceipt, ClauseSnapshot, sign_receipt, verify_receipt, build_cf_receipt,
)

__version__ = "0.1.0"

__all__ = [
    "TABank", "TsetlinMachine", "TMConfig", "to_literals",
    "CCTM", "CCTMConfig", "FeatureFeasibility",
    "generate_cf_pairs_from_oracle",
    "Recourse", "recourse_exact", "recourse_greedy", "recourse_for",
    "CFReceipt", "ClauseSnapshot",
    "sign_receipt", "verify_receipt", "build_cf_receipt",
]
