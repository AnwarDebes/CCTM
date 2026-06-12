# Changelog

All notable changes to CCTM are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-12

Reference implementation accompanying the paper.

### Added

- Canonical Tsetlin Automaton bank and binary TM wrapper.
- CCTM core: counterfactual data augmentation against an external CF
  oracle, Type-III contrastive feedback, per-clause causal scores,
  per-feature treatment effects, and causal influence.
- Structural counterfactual recourse: exact and greedy minimal flip
  sets under feasibility constraints.
- Signed counterfactual receipts: DIMACS CNF plus HMAC certificate,
  offline re-verifiable in about 0.1 ms.
- Bundled synthetic SCM and loan-denial datasets.
- Three reproducible experiments (ATE recovery, loan recourse,
  fairness audit), about 60 s total on a single CPU.
- Figure-generation script, LaTeX paper, 14 unit tests.

### Known limitations (v0.1)

- Conditions on an external CF oracle; does not learn a causal DAG.
- Single-feature interventions only.
- Synthetic datasets only; Adult-Income and COMPAS are future work.

[0.1.0]: https://github.com/AnwarDebes/CCTM/releases/tag/v0.1.0
