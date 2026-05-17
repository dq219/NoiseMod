"""Schematic figures for the slide deck:

* ``arch_pipeline.png``  — boxes-and-arrows showing input → ResNet18 trunk →
  AdaptiveAvgPool((1,N)) → fc1 → LeakyReLU → fc2 → reshape, with shapes
  annotated. The MLP head is highlighted to mark where the regularizer attaches.
* ``modularity_pipeline.png`` — |fc2.W| → cosine similarity S → normalized
  Laplacian L → eigenspectrum, with the eigengap at position N annotated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def draw_box(ax, xy, w, h, text, *, fc="#dfe9f5", ec="#1f3a68", lw=1.2,
             fontsize=10, text_color="black"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color)
    return (cx, cy, w, h)


def arrow(ax, x0, y0, x1, y1, label=None, lw=1.3):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=lw),
    )
    if label is not None:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label,
                ha="center", va="bottom", fontsize=8, color="#444")


def make_arch_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_axis_off()

    boxes = [
        (0.2, 1.4, 1.7, "input\n$(B, 3, 32, 32N)$", "#f0f0f0"),
        (2.3, 1.4, 1.7, "ResNet-18 trunk\n(CIFAR stem,\nno maxpool)", "#cfe2cf"),
        (4.4, 1.4, 1.7, "AdaptiveAvgPool\n$(1, N)$\n$\\to (B, 512, 1, N)$", "#cfe2cf"),
        (6.5, 1.4, 1.7, "flatten\n$\\to (B, 512N)$", "#e0e0e0"),
        (8.6, 1.4, 1.7, "fc1\n$512N \\to h$\n$+$ LeakyReLU", "#ffe4b5"),
        (10.7, 1.4, 1.7, "fc2\n$h \\to 10N$", "#ffd9d9"),
        (12.8, 1.4, 1.05, "view\n$(B, N, 10)$", "#f0f0f0"),
    ]
    centers = []
    for (x, y, w, txt, fc) in boxes:
        c = draw_box(ax, (x, y), w, 1.4, txt, fc=fc, fontsize=9)
        centers.append(c)

    # arrows between consecutive boxes
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        y = boxes[i][1] + 0.7
        arrow(ax, x0, y, x1, y)

    # highlight MLP head + regularizer site
    head_x0 = boxes[4][0] - 0.1
    head_x1 = boxes[5][0] + boxes[5][2] + 0.1
    ax.add_patch(mpatches.FancyBboxPatch(
        (head_x0, 1.25), head_x1 - head_x0, 1.7,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="none", edgecolor="#c0392b", linestyle="--", linewidth=1.5,
    ))
    ax.text((head_x0 + head_x1) / 2, 3.05, "MLP head",
            ha="center", va="bottom", fontsize=10, color="#c0392b")

    # regularizer annotation
    ax.annotate(
        "weighted-activity regularizer\n"
        r"$\mathbb{E}_B \sum_j (|h|^{\alpha} (W_2^\top)^2)_j$",
        xy=(boxes[4][0] + boxes[4][2], 1.4),
        xytext=(boxes[4][0] + 0.0, 0.05),
        ha="left", va="bottom", fontsize=9, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0),
    )

    # output frame labels
    ax.text(13.32, 1.05, r"row $j$: frame $\lfloor j/10 \rfloor$,"
            r" class $j\,\mathrm{mod}\,10$",
            ha="center", va="top", fontsize=8, color="#444")

    fig.suptitle("Architecture: ResNet-18 trunk + per-frame MLP head",
                  fontsize=13, y=0.98)
    out = FIGS / "arch_pipeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def make_modularity_pipeline() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    rng = np.random.default_rng(0)
    h = 24
    N = 3
    out_dim = 10 * N
    # Construct a synthetic block-structured fc2 for illustration.
    W = 0.05 * rng.standard_normal((out_dim, h))
    block_h = h // N
    for k in range(N):
        rows = slice(10 * k, 10 * (k + 1))
        cols = slice(block_h * k, block_h * (k + 1))
        W[rows, cols] += 0.8 * rng.standard_normal((10, block_h))

    # Panel 1: |W|
    ax = axes[0]
    im = ax.imshow(np.abs(W), aspect="auto", cmap="magma")
    ax.set_title(r"$|W_{fc2}|$" "\nshape $(10N,\\; h)$")
    ax.set_xlabel("hidden unit")
    ax.set_ylabel("output unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: similarity S
    Wabs = np.abs(W) + 1e-7
    col_sq = (Wabs ** 2).sum(axis=0)
    norm = np.sqrt(np.outer(col_sq, col_sq))
    S = (Wabs.T @ Wabs) / norm
    ax = axes[1]
    im = ax.imshow(S, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(r"$S = \frac{|W|^\top |W|}{\|col\|\,\|col\|}$"
                  "\ncosine similarity between hidden units")
    ax.set_xlabel("hidden unit")
    ax.set_ylabel("hidden unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Laplacian L
    d = S.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-7))
    L = np.eye(len(S)) - d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)
    ax = axes[2]
    im = ax.imshow(L, cmap="coolwarm",
                    vmin=-np.abs(L).max(), vmax=np.abs(L).max())
    ax.set_title(r"$L = I - D^{-1/2} S D^{-1/2}$"
                  "\nsymmetric normalized Laplacian")
    ax.set_xlabel("hidden unit")
    ax.set_ylabel("hidden unit")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 4: eigenvalues
    eig = np.linalg.eigvalsh(L)
    ax = axes[3]
    idx = np.arange(1, len(eig) + 1)
    ax.plot(idx[:10], eig[:10], "o-", color="#1f3a68", markersize=6)
    ax.axvspan(N + 0.5, N + 1.5, color="#c0392b", alpha=0.15,
               label=f"eigengap at $N$={N}")
    ax.set_xlabel("eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("Spectrum (first 10)\nN modules $\\Rightarrow$ "
                  r"$\lambda_N \to 0$, $\lambda_{N+1}$ jumps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.suptitle("Modularity readout: $|W_{fc2}|$ "
                  r"$\to$ cosine sim $\to$ Laplacian $\to$ eigenspectrum",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGS / "modularity_pipeline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"wrote {make_arch_pipeline()}")
    print(f"wrote {make_modularity_pipeline()}")
