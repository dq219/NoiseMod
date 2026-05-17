"""Single-panel figure: the first 10 Laplacian eigenvalues of fc2 at the
final checkpoint, one curve per N = 1..8.

Reads ``eigvals`` (shape ``(E+1, 10)``) from each run's ``metrics.npz`` and
plots the last row as a function of eigenvalue index. Highlights position
``N`` on each curve so the eigengap at index ``N`` is visible.
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
        cfg_p, npz_p = run_dir / "config.json", run_dir / "metrics.npz"
        if not (cfg_p.exists() and npz_p.exists()):
            continue
        cfg = json.load(open(cfg_p))
        d = np.load(npz_p)
        runs.append({
            "N": int(cfg["N"]),
            "eigvals_final": d["eigvals"][-1],
        })
    runs.sort(key=lambda r: r["N"])
    return runs


def main() -> Path:
    runs = load_runs()
    Ns = [r["N"] for r in runs]
    cmap = plt.get_cmap("viridis")
    colors = {N: cmap(i / max(1, len(Ns) - 1)) for i, N in enumerate(Ns)}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    idx = np.arange(1, 11)
    for r in runs:
        N = r["N"]
        ax.plot(idx, r["eigvals_final"], "o-", color=colors[N], lw=1.6,
                markersize=5, label=f"N={N}")
        # mark position N (the smallest eigenvalue inside the "module" block)
        if 1 <= N <= 10:
            ax.scatter(N, r["eigvals_final"][N - 1], s=90, facecolor="none",
                       edgecolor=colors[N], linewidth=1.8, zorder=5)

    ax.set_xlabel("eigenvalue index $k$")
    ax.set_ylabel(r"$\lambda_k$ at final epoch")
    ax.set_title("Laplacian eigenspectrum of $|W_{fc2}|$ at final epoch "
                 "(batch=128, reg=1.0)")
    ax.set_xticks(idx)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="upper left")

    fig.tight_layout()
    out = FIGS / "final_spectrum.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {main()}")
