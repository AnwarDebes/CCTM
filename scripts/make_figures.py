"""Generate every figure used in the CCTM paper from the experiment JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


# ---------------- Figure: synthetic SCM ATE recovery ----------------
def fig_scm():
    data = load("synthetic_scm.json")
    rows = data["per_seed"]
    # Per-feature: vanilla TE_from_pred, cctm TE_empirical, cctm TE_from_pred, ground truth
    n_features = len(rows[0]["ATE_ground_truth"])
    gt = np.mean([r["ATE_ground_truth"] for r in rows], axis=0)
    van = np.mean([r["vanilla_tm"]["te_from_predictions"]["values"]
                   for r in rows], axis=0)
    cctm_emp = np.mean([r["cctm"]["te_empirical"]["values"]
                        for r in rows], axis=0)
    cctm_pred = np.mean([r["cctm"]["te_from_predictions"]["values"]
                         for r in rows], axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.2
    x = np.arange(n_features)
    ax.bar(x - 1.5 * width, gt, width, label="True $|ATE|$ (oracle)", color="#444")
    ax.bar(x - 0.5 * width, van, width, label="Vanilla TM (pred-flip)", color="#9ec5fe")
    ax.bar(x + 0.5 * width, cctm_emp, width, label="CCTM (CF stats)", color="#7caf3a")
    ax.bar(x + 1.5 * width, cctm_pred, width, label="CCTM (pred-flip)", color="#3a7caf")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$x_{i}$" for i in range(n_features)])
    ax.set_ylabel("Estimated $|ATE|$")
    ax.set_ylim(0, max(gt.max(), van.max(), cctm_emp.max(), cctm_pred.max()) * 1.2)
    ax.set_title("Synthetic SCM: per-feature treatment effect recovery")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "scm_ate.png", dpi=200)
    fig.savefig(FIGS_DIR / "scm_ate.pdf")
    plt.close(fig)


# ---------------- Figure: loan recourse ----------------
def fig_loan():
    data = load("loan_recourse.json")
    s = data["summary"]
    keys = ["exact_success", "exact_length", "exact_lat_ms",
            "greedy_success", "greedy_length", "greedy_lat_ms"]
    nice = {
        "exact_success": "Recourse\nsuccess",
        "exact_length":  "Mean\nlength",
        "exact_lat_ms":  "Latency\n(ms, /100)",
        "greedy_success":"Recourse\nsuccess",
        "greedy_length": "Mean\nlength",
        "greedy_lat_ms": "Latency\n(ms, /100)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    for ax, kind in zip(axes, ("exact", "greedy")):
        vmeans, cmeans, vstds, cstds, lbls = [], [], [], [], []
        for k in keys:
            if not k.startswith(kind):
                continue
            v = s[f"vanilla_{k}"]; c = s[f"cctm_{k}"]
            scale = 0.01 if k.endswith("_lat_ms") else 1.0
            vmeans.append(v["mean"] * scale)
            cmeans.append(c["mean"] * scale)
            vstds.append(v["std"] * scale)
            cstds.append(c["std"] * scale)
            lbls.append(nice[k])
        x = np.arange(len(lbls))
        w = 0.4
        ax.bar(x - w / 2, vmeans, width=w, yerr=vstds, capsize=3,
               label="Vanilla TM", color="#9ec5fe")
        ax.bar(x + w / 2, cmeans, width=w, yerr=cstds, capsize=3,
               label="CCTM",       color="#3a7caf")
        ax.set_xticks(x)
        ax.set_xticklabels(lbls)
        ax.set_title(f"Recourse algorithm: {kind}")
        ax.set_ylim(0, max(max(vmeans), max(cmeans)) * 1.2)
        ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Loan-denial recourse: CCTM matches vanilla TM "
                 "(both succeed 100\\%, length ~1.20)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "loan_recourse.png", dpi=200)
    fig.savefig(FIGS_DIR / "loan_recourse.pdf")
    plt.close(fig)


# ---------------- Figure: fairness audit ----------------
def fig_fairness():
    data = load("fairness_audit.json")
    s = data["summary"]
    names = ["vanilla", "cctm", "audited"]
    display = {"vanilla": "Vanilla TM\n(biased data)",
               "cctm":    "CCTM\n(CF training)",
               "audited": "CCTM\n+ clause audit"}
    metrics = ["acc", "dp"]
    metric_names = {"acc": "Accuracy vs CLEAN ground truth",
                    "dp":  "Demographic parity gap"}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    for ax, m in zip(axes, metrics):
        means = [s[f"{n}_{m}"]["mean"] for n in names]
        stds  = [s[f"{n}_{m}"]["std"]  for n in names]
        x = np.arange(len(names))
        colors = ["#cf6679", "#3a7caf", "#7caf3a"]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([display[n] for n in names])
        ax.set_title(metric_names[m])
        ax.set_ylim(0, max(means) * 1.35 if max(means) > 0 else 0.05)
        for i, v in enumerate(means):
            ax.text(i, v + max(means) * 0.04, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Fairness audit: CCTM recovers from biased training data; "
                 "muting flagged clauses further halves DP",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fairness_audit.png", dpi=200)
    fig.savefig(FIGS_DIR / "fairness_audit.pdf")
    plt.close(fig)


# ---------------- Figure: architecture schematic ----------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.axis("off")
    boxes = [
        ("Factual data\n$(x_i, y_i)$",     0.04, 0.78, 0.18, 0.14, "#eee"),
        ("CF oracle\n(generates $x'_i, y'_i$)",
                                            0.04, 0.45, 0.18, 0.14, "#cfe6f7"),
        ("CCTM core\n(TA bank +\nType-I/II/III feedback)",
                                            0.30, 0.55, 0.24, 0.30, "#fde2e1"),
        ("Per-clause\ncausal score",       0.60, 0.78, 0.20, 0.12, "#fff5cc"),
        ("Feature-level\nTE / CI",         0.60, 0.60, 0.20, 0.12, "#fff5cc"),
        ("Structural\ncounterfactual\nrecourse",
                                            0.60, 0.32, 0.20, 0.18, "#dbf3d9"),
        ("Signed CF\nreceipt\n(DIMACS + HMAC)",
                                            0.82, 0.32, 0.16, 0.18, "#e7d3ff"),
    ]
    for txt, x, y, w, h, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="black",
                                   linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    def arr(x0, y0, x1, y1, style="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0,
                                    linestyle=style))
    arr(0.22, 0.84, 0.30, 0.78)
    arr(0.22, 0.51, 0.30, 0.60)
    arr(0.54, 0.78, 0.60, 0.84)
    arr(0.54, 0.66, 0.60, 0.66)
    arr(0.54, 0.55, 0.60, 0.42)
    arr(0.80, 0.40, 0.82, 0.40)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("CCTM architecture", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "architecture.png", dpi=200)
    fig.savefig(FIGS_DIR / "architecture.pdf")
    plt.close(fig)


def main():
    fig_scm()
    fig_loan()
    fig_fairness()
    fig_architecture()
    print(f"Wrote figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()
