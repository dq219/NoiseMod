"""Visualize the modularity eigengap dynamics across N for runs under
``<results-root>``.

Five panels in a 2x3 grid, single output PNG:
  1. lambda_N (dashed) and lambda_{N+1} (solid) over epochs.
  2. gap = lambda_{N+1} - lambda_N over epochs.
  3. gap over scaled epoch t/N.
  4. gap zoomed to early epochs (xlim=(0, 10), ylim auto-fit to that window).
  5. half-rise epoch t_{1/2} vs N — where t_{1/2} is the first epoch the gap
     reaches (g_init + g_final) / 2 (linear interpolation between samples).

The N=1 run is excluded — its 'eigengap at position N' is lambda_2 - lambda_1
= lambda_2 (since lambda_1 = 0 for any connected graph) and is not the same
quantity as the modularity gap for N >= 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def discover_runs(results_root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        npz_path = run_dir / "metrics.npz"
        if not (cfg_path.exists() and npz_path.exists()):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        runs.append({
            "dir": run_dir,
            "N": int(cfg["N"]),
            "reg": float(cfg["reg_weight"]),
            "name": run_dir.name,
        })
    return runs


def load_eig(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epochs, lam_N, lam_{N+1}). 1-indexed lambdas, so the columns
    are eigvals[:, N-1] and eigvals[:, N]."""
    d = np.load(run["dir"] / "metrics.npz")
    N = run["N"]
    eigvals = d["eigvals"]  # (E+1, 10)
    if eigvals.shape[1] <= N:
        raise ValueError(
            f"{run['name']}: only {eigvals.shape[1]} eigenvalues tracked, "
            f"need at least N+1 = {N + 1}"
        )
    return d["epoch"], eigvals[:, N - 1], eigvals[:, N]


def half_rise_epoch(epochs: np.ndarray, gap: np.ndarray) -> tuple[float, float, float, float]:
    """Return (t_half, g_init, g_final, g_half) with linear interp."""
    # preprocess: make sure gap is non-decreasing by replacing each value with the max so far.
    gap = np.minimum.accumulate(gap[::-1])[::-1]
    g_init = float(gap[0])
    g_final = float(gap[-1])
    g_half = 0.5 * (g_init + g_final)
    above = gap >= g_half if g_final >= g_init else gap <= g_half
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float("nan"), g_init, g_final, g_half
    i = int(idx[0])
    if i == 0:
        return float(epochs[0]), g_init, g_final, g_half
    g0, g1 = gap[i - 1], gap[i]
    t0, t1 = epochs[i - 1], epochs[i]
    if g1 == g0:
        return float(t1), g_init, g_final, g_half
    frac = (g_half - g0) / (g1 - g0)
    return float(t0 + frac * (t1 - t0)), g_init, g_final, g_half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str,
                        default="./results/batch_size_128")
    parser.add_argument("--out", type=str,
                        default="./results/batch_size_128/modularity_analysis.png")
    parser.add_argument("--reg", type=float, default=None,
                        help="filter to runs at this reg_weight")
    parser.add_argument("--zoom-xmax", type=float, default=20.0,
                        help="upper xlim for the zoom panel")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    runs = discover_runs(results_root)
    if args.reg is not None:
        runs = [r for r in runs if np.isclose(r["reg"], args.reg)]
    runs = [r for r in runs if r["N"] >= 2]
    if not runs:
        raise SystemExit(f"no runs found under {results_root}")
    runs.sort(key=lambda r: r["N"])

    Ns = [r["N"] for r in runs]
    cmap = plt.get_cmap("viridis")
    colors = {r["N"]: cmap(i / max(1, len(runs) - 1)) for i, r in enumerate(runs)}

    # Pre-load every run so we can compute ylim from the zoom window.
    loaded: dict[int, dict] = {}
    for r in runs:
        epochs, lam_N, lam_Np1 = load_eig(r)
        gap = lam_Np1 - lam_N
        t_half, g_init, g_final, g_half = half_rise_epoch(epochs, gap)
        loaded[r["N"]] = {
            "epochs": epochs, "lam_N": lam_N, "lam_Np1": lam_Np1, "gap": gap,
            "t_half": t_half, "g_init": g_init, "g_final": g_final,
            "g_half": g_half,
        }

    # ylim for the zoom panel: max over all curves within xlim.
    zoom_mask_max = 0.0
    for d in loaded.values():
        mask = d["epochs"] <= args.zoom_xmax
        if mask.any():
            zoom_mask_max = max(zoom_mask_max, float(np.nanmax(d["gap"][mask])))
    zoom_ymax = 0.05

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax_pair, ax_gap, ax_scaled = axes[0]
    ax_zoom, ax_half, ax_half2 = axes[1]

    for r in runs:
        d = loaded[r["N"]]
        col = colors[r["N"]]
        ax_pair.plot(d["epochs"], d["lam_N"], color=col, lw=1.2, ls="--",
                     alpha=0.85)
        ax_pair.plot(d["epochs"], d["lam_Np1"], color=col, lw=1.6, ls="-",
                     label=f"N={r['N']}")
        ax_gap.plot(d["epochs"], d["gap"], color=col, lw=1.6,
                    label=f"N={r['N']}")
        ax_scaled.plot(d["epochs"] / r["N"], d["gap"], color=col, lw=1.6,
                       label=f"N={r['N']}")
        ax_zoom.plot(d["epochs"], d["gap"], color=col, lw=1.6, marker="o",
                     markersize=3, label=f"N={r['N']}")

    ax_pair.set_xlabel("epoch")
    ax_pair.set_ylabel("eigenvalue")
    ax_pair.set_title(r"$\lambda_N$ (dashed) and $\lambda_{N+1}$ (solid)")
    ax_pair.grid(True, alpha=0.3)
    ax_pair.legend(loc="best", fontsize=9, ncol=2, frameon=False)

    ax_gap.set_xlabel("epoch")
    ax_gap.set_ylabel(r"$\lambda_{N+1} - \lambda_N$")
    ax_gap.set_title("Modularity eigengap")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_gap.legend(loc="best", fontsize=9, ncol=2, frameon=False)

    ax_scaled.set_xlabel("scaled epoch  (t / N)")
    ax_scaled.set_ylabel(r"$\lambda_{N+1} - \lambda_N$")
    ax_scaled.set_title("Modularity eigengap, time rescaled by N")
    ax_scaled.grid(True, alpha=0.3)
    ax_scaled.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_scaled.legend(loc="best", fontsize=9, ncol=2, frameon=False)

    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel(r"$\lambda_{N+1} - \lambda_N$")
    ax_zoom.set_title(f"Modularity eigengap (zoom: epochs 0–{args.zoom_xmax:g})")
    ax_zoom.set_xlim(0, args.zoom_xmax)
    ax_zoom.set_ylim(0, zoom_ymax)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="best", fontsize=9, ncol=2, frameon=False)

    Ns_arr = np.array(Ns)
    t_halves = np.array([loaded[n]["t_half"] for n in Ns])
    ax_half.plot(Ns_arr, t_halves, marker="o", lw=1.5, color="C0")
    for n, t in zip(Ns_arr, t_halves):
        ax_half.annotate(f"{t:.1f}", (n, t), textcoords="offset points",
                         xytext=(6, 4), fontsize=9, color="C0")
    ax_half.set_xlabel("N (number of frames / modules)")
    ax_half.set_ylabel(r"half-rise epoch  $t_{1/2}$")
    ax_half.set_ylim(0, 60)
    ax_half.set_xlim(0)
    ax_half.set_title(r"Epoch where gap reaches $(g_0 + g_T)/2$")
    ax_half.grid(True, alpha=0.3)

    ax_half2.plot(Ns_arr, t_halves, marker="o", lw=1.5, color="C0")
    for n, t in zip(Ns_arr, t_halves):
        ax_half2.annotate(f"{t:.1f}", (n, t), textcoords="offset points",
                         xytext=(6, 4), fontsize=9, color="C0")
    ax_half2.set_xlabel("N (number of frames / modules)")
    ax_half2.set_ylabel(r"half-rise epoch  $t_{1/2}$")
    ax_half2.set_yscale('log')
    # ax_half2.set_xscale('log')
    ax_half2.set_title(r"Epoch where gap reaches $(g_0 + g_T)/2$")
    ax_half2.grid(True, alpha=0.3)

    reg_str = (f"reg_weight = {runs[0]['reg']:g}"
               if all(r["reg"] == runs[0]["reg"] for r in runs)
               else "mixed reg_weight")
    fig.suptitle(f"fc2 Laplacian eigengap at position N  ({reg_str}, "
                 f"N $\\in$ {{{', '.join(str(n) for n in Ns)}}})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    print(f"wrote {out_path}\n")
    print(f"{'N':>3} {'g_init':>10} {'g_final':>10} {'g_half':>10} {'t_half':>10}")
    for n in Ns:
        d = loaded[n]
        print(f"{n:>3} {d['g_init']:>10.4f} {d['g_final']:>10.4f} "
              f"{d['g_half']:>10.4f} {d['t_half']:>10.3f}")


if __name__ == "__main__":
    main()
