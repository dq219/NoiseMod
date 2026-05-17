"""Training-curve figures for the slide deck, all loaded from
``results/batch_size_128/``:

* ``training_accuracy.png`` — test accuracy over epochs, one curve per N,
  colored by N.
* ``eigvals_representative.png`` — first 10 Laplacian eigenvalues over epochs
  for the representative N=4 run (2x5 panel, mirrors run.py:save_plots).
* ``lamN_pair.png`` — λ_N (dashed) and λ_{N+1} (solid) over epochs, one color
  per N. Same panel as plot_eigengap.py's top-left, separated out for slides.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results" / "batch_size_128"


def load_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_p = run_dir / "config.json"
        npz_p = run_dir / "metrics.npz"
        if not (cfg_p.exists() and npz_p.exists()):
            continue
        cfg = json.load(open(cfg_p))
        d = np.load(npz_p)
        runs.append({
            "dir": run_dir, "N": int(cfg["N"]),
            "reg": float(cfg["reg_weight"]),
            "epochs": d["epoch"], "acc": d["test_acc"],
            "eigvals": d["eigvals"],
        })
    runs.sort(key=lambda r: r["N"])
    return runs


def cmap_for_runs(runs: list[dict]) -> dict[int, tuple]:
    cmap = plt.get_cmap("viridis")
    Ns = [r["N"] for r in runs]
    nN = max(1, len(Ns) - 1)
    return {N: cmap(i / nN) for i, N in enumerate(Ns)}


def make_accuracy(runs: list[dict]) -> Path:
    colors = cmap_for_runs(runs)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        valid = ~np.isnan(r["acc"])
        ax.plot(r["epochs"][valid], r["acc"][valid],
                color=colors[r["N"]], lw=1.6, label=f"N={r['N']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (per-frame)")
    ax.set_title("Per-frame test accuracy vs epoch (batch size 128, reg=1.0)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.1, color="k", lw=0.5, ls=":", alpha=0.6)
    ax.text(0.5, 0.105, "chance", fontsize=8, color="#555")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    out = FIGS / "training_accuracy.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_eigvals_panel(runs: list[dict], N_rep: int = 4) -> Path:
    rep = next((r for r in runs if r["N"] == N_rep), None)
    if rep is None:
        rep = runs[len(runs) // 2]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(rep["epochs"], rep["eigvals"][:, i], color="#1f3a68", lw=1.4)
        ax.set_title(rf"$\lambda_{{{i + 1}}}$")
        ax.grid(True, alpha=0.3)
        if i >= 5:
            ax.set_xlabel("epoch")
        if i % 5 == 0:
            ax.set_ylabel("eigenvalue")
        if (i + 1) == rep["N"]:
            ax.set_facecolor("#fff3f0")
        if (i + 1) == rep["N"] + 1:
            ax.set_facecolor("#f0f7ff")
    fig.suptitle(
        f"First 10 Laplacian eigenvalues of fc2 — N = {rep['N']} "
        f"(λ_N panel shaded red, λ_{{N+1}} blue)", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGS / "eigvals_representative.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_lam_pair(runs: list[dict]) -> Path:
    """λ_N and λ_{N+1} overlay across N; N=1 excluded."""
    rs = [r for r in runs if r["N"] >= 2]
    colors = cmap_for_runs(rs)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rs:
        N = r["N"]
        ax.plot(r["epochs"], r["eigvals"][:, N - 1], color=colors[N],
                lw=1.2, ls="--", alpha=0.85)
        ax.plot(r["epochs"], r["eigvals"][:, N], color=colors[N], lw=1.6,
                label=f"N={N}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("eigenvalue")
    ax.set_title(r"$\lambda_N$ (dashed) and $\lambda_{N+1}$ (solid) vs epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    out = FIGS / "lamN_pair.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    runs = load_runs()
    for path in (make_accuracy(runs),
                 make_eigvals_panel(runs),
                 make_lam_pair(runs)):
        print(f"wrote {path}")
