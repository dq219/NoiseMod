"""Modularity-result figures for the slide deck:

* ``eigengap_overview.png`` — 1x2 grid:
    [0] eigengap vs epoch, one curve per N (N>=2); each run's identified
        half-rise point  (t_{1/2}, g_{1/2})  is overlaid as a scatter marker.
    [1] half-rise epoch  t_{1/2}  vs  N  (linear scale, labeled markers).
* ``final_modularity_vs_N.png`` — at the final epoch and at initialization:
    λ_N, λ_{N+1}, and the eigengap, vs N.

Both load every run under ``results/batch_size_128/`` (filtered to N >= 2 for
the eigengap figure; N >= 1 is fine for the final-modularity scatter, but the
N=1 gap is included only for reference — see README §"Cross-run analysis").
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
            "dir": run_dir, "N": int(cfg["N"]),
            "reg": float(cfg["reg_weight"]),
            "epochs": d["epoch"], "eigvals": d["eigvals"],
        })
    runs.sort(key=lambda r: r["N"])
    return runs


def half_rise_epoch(epochs: np.ndarray, gap: np.ndarray) -> tuple[float, float]:
    """Return ``(t_half, g_half)`` where the (monotonised) gap first reaches
    ``(g_init + g_final) / 2``, with linear interpolation between samples.
    ``g_half`` is the value used as the threshold, so the returned point is
    always exactly on the half-rise level."""
    gap = np.minimum.accumulate(gap[::-1])[::-1]
    g_init, g_final = float(gap[0]), float(gap[-1])
    g_half = 0.5 * (g_init + g_final)
    above = gap >= g_half if g_final >= g_init else gap <= g_half
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float("nan"), g_half
    i = int(idx[0])
    if i == 0:
        return float(epochs[0]), g_half
    g0, g1 = gap[i - 1], gap[i]
    t0, t1 = epochs[i - 1], epochs[i]
    if g1 == g0:
        return float(t1), g_half
    return float(t0 + (g_half - g0) / (g1 - g0) * (t1 - t0)), g_half


def make_eigengap_overview(runs: list[dict]) -> Path:
    rs = [r for r in runs if r["N"] >= 2]
    Ns = [r["N"] for r in rs]
    cmap = plt.get_cmap("viridis")
    colors = {N: cmap(i / max(1, len(Ns) - 1)) for i, N in enumerate(Ns)}

    gaps = {r["N"]: r["eigvals"][:, r["N"]] - r["eigvals"][:, r["N"] - 1]
            for r in rs}
    halves = {r["N"]: half_rise_epoch(r["epochs"], gaps[r["N"]]) for r in rs}

    fig, (ax_full, ax_half) = plt.subplots(1, 2, figsize=(13, 4.8))

    for r in rs:
        N = r["N"]
        ax_full.plot(r["epochs"], gaps[N], color=colors[N], lw=1.6,
                     label=f"N={N}")
    # half-rise markers overlaid on the gap-vs-epoch curves
    for r in rs:
        N = r["N"]
        t_half, g_half = halves[N]
        if np.isnan(t_half):
            continue
        ax_full.scatter(t_half, g_half, s=70, color=colors[N],
                        edgecolor="black", linewidth=1.0, zorder=5)

    ax_full.set_xlabel("epoch")
    ax_full.set_ylabel(r"$\lambda_{N+1} - \lambda_N$")
    ax_full.set_title("Modularity eigengap vs epoch "
                      r"($\circ$: half-rise point)")
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(fontsize=9, ncol=2, frameon=False)

    Ns_arr = np.array(Ns)
    t_halves = np.array([halves[N][0] for N in Ns])
    ax_half.plot(Ns_arr, t_halves, "o-", color="#1f3a68", lw=1.6, markersize=6)
    for n, t in zip(Ns_arr, t_halves):
        ax_half.annotate(f"{t:.1f}", (n, t), textcoords="offset points",
                          xytext=(6, 6), fontsize=9, color="#1f3a68")
    ax_half.set_xlabel("N (number of frames / modules)")
    ax_half.set_ylabel(r"half-rise epoch $t_{1/2}$")
    ax_half.set_title(r"$t_{1/2}$ vs $N$ — roughly linear "
                       "(N=8 still rising at epoch 100)")
    ax_half.grid(True, alpha=0.3)

    fig.suptitle("Modularity eigengap dynamics across $N$ "
                  "(batch=128, reg=1.0)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / "eigengap_overview.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_final_modularity(runs: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax_eig, ax_gap = axes

    Ns = np.array([r["N"] for r in runs])

    lamN_init = np.array([r["eigvals"][0, r["N"] - 1] for r in runs])
    lamNp1_init = np.array([r["eigvals"][0, r["N"]] for r in runs])
    lamN_fin = np.array([r["eigvals"][-1, r["N"] - 1] for r in runs])
    lamNp1_fin = np.array([r["eigvals"][-1, r["N"]] for r in runs])
    gap_init = lamNp1_init - lamN_init
    gap_fin = lamNp1_fin - lamN_fin

    ax_eig.plot(Ns, lamN_init, "o--", color="#999",
                label=r"$\lambda_N$ init")
    ax_eig.plot(Ns, lamNp1_init, "s--", color="#999",
                label=r"$\lambda_{N+1}$ init")
    ax_eig.plot(Ns, lamN_fin, "o-", color="#c0392b", lw=1.8,
                markersize=7, label=r"$\lambda_N$ final")
    ax_eig.plot(Ns, lamNp1_fin, "s-", color="#1f3a68", lw=1.8,
                markersize=7, label=r"$\lambda_{N+1}$ final")
    ax_eig.set_xlabel("N")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_title(r"$\lambda_N$ stays small; $\lambda_{N+1}$ lifts")
    ax_eig.grid(True, alpha=0.3)
    ax_eig.legend(fontsize=9, frameon=False)

    ax_gap.plot(Ns, gap_init, "o--", color="#999", label="init")
    ax_gap.plot(Ns, gap_fin, "o-", color="#2e7d32", lw=1.8, markersize=7,
                label="final")
    for n, g in zip(Ns, gap_fin):
        ax_gap.annotate(f"{g:.2f}", (n, g), textcoords="offset points",
                         xytext=(6, 6), fontsize=9, color="#2e7d32")
    ax_gap.set_xlabel("N")
    ax_gap.set_ylabel(r"$\lambda_{N+1} - \lambda_N$")
    ax_gap.set_title("Final-epoch eigengap shrinks with N")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(fontsize=10, frameon=False)

    fig.suptitle("Eigengap at initialization vs final epoch "
                  "(batch=128, reg=1.0)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / "final_modularity_vs_N.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    runs = load_runs()
    print(f"wrote {make_eigengap_overview(runs)}")
    print(f"wrote {make_final_modularity(runs)}")
