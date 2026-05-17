"""Heatmaps of |fc2.weight| with hidden units reordered by inferred module.

For each requested N, loads ``results/batch_size_128/N{N}_reg1.0_h{512N}``,
clusters the hidden units (columns of fc2.weight) via spectral k-means on the
normalized Laplacian of the cosine-similarity graph (same construction as
run.py:laplacian_spectrum), and renders a heatmap with:

  * rows grouped by frame (row j -> frame j // 10, naturally contiguous);
  * columns sorted by cluster, with cluster boundaries drawn as vertical lines;
  * cluster ordering chosen so that cluster i's modal output frame is i (or
    as close as possible) — this lines up the block-diagonal visually.

Mirrors the spirit of ``../modularisation_via_noise/utils/graphs.py``'s
``DetectCommunity`` but specialised to k = N and to the row ordering we want
(by frame). The reference DetectCommunity hardcodes a ``range(4)`` loop, so
it is not directly applicable for arbitrary N.

Writes one PNG per requested N to ``summary/figs/fc2_heatmap_N{N}.png`` plus a
combined ``summary/figs/fc2_heatmaps_grid.png`` for the slide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import eigh
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results" / "batch_size_128"

DEFAULT_N_LIST = (2, 4, 8)


def cluster_hidden_units(W: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Return a permutation of column indices that groups hidden units by
    inferred module. ``W`` has shape (out_dim, in_dim) — here (10N, h).

    1. Build the cosine-similarity matrix between columns of |W|.
    2. Symmetric-normalized Laplacian.
    3. KMeans on the first k eigenvectors.
    4. Order clusters by the modal *output frame* they project to most
       strongly, so cluster 0 lines up with frame 0 (where possible).
    """
    eps = 1e-7
    A = np.abs(W) + eps
    col_sq = (A ** 2).sum(axis=0)
    norm = np.sqrt(np.outer(col_sq, col_sq))
    S = (A.T @ A) / norm
    d = S.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    L = np.eye(len(S)) - d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :]
    L = 0.5 * (L + L.T)
    eigvals, eigvecs = eigh(L)
    X = eigvecs[:, :k]
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)

    # Choose a permutation of cluster ids so cluster i ↔ frame i. For each
    # cluster, score its affinity to each frame by summing |W| over the
    # frame's output rows and over the cluster's hidden units. Then greedily
    # match using Hungarian-style: argmax assignments in score order.
    out_dim, h = W.shape
    num_frames = out_dim // 10
    score = np.zeros((k, num_frames))
    for c in range(k):
        cols = (labels == c)
        if not cols.any():
            continue
        for f in range(num_frames):
            rows = slice(10 * f, 10 * (f + 1))
            score[c, f] = np.abs(W[rows, :][:, cols]).sum()

    # Greedy assignment: pick the largest entry of `score`, assign that
    # cluster -> frame, mask the row & column, repeat.
    cluster_to_frame = -np.ones(k, dtype=int)
    available_clusters = list(range(k))
    available_frames = list(range(num_frames))
    while available_clusters and available_frames:
        sub = score[np.ix_(available_clusters, available_frames)]
        ci, fi = np.unravel_index(np.argmax(sub), sub.shape)
        c = available_clusters[ci]
        f = available_frames[fi]
        cluster_to_frame[c] = f
        available_clusters.remove(c)
        available_frames.remove(f)
    # Any remaining clusters (k > num_frames is impossible here, but be safe)
    # get unused indices appended after.
    unused = sorted(set(range(k)) - set(cluster_to_frame[cluster_to_frame >= 0]))
    next_free = max(cluster_to_frame.max() + 1, num_frames) if k > 0 else 0
    for c in range(k):
        if cluster_to_frame[c] < 0:
            cluster_to_frame[c] = next_free
            next_free += 1

    new_label = cluster_to_frame[labels]
    sorted_idx = np.argsort(new_label, kind="stable")
    # boundaries: count of items in each sorted-label bucket
    bounds = [0]
    for i in range(k):
        bounds.append(bounds[-1] + int((new_label == i).sum()))
    return sorted_idx, np.array(bounds)


def load_fc2(N: int) -> np.ndarray:
    run = RESULTS / f"N{N}_reg1.0_h{512 * N}"
    sd = torch.load(run / "weights.pt", map_location="cpu")
    return sd["fc2.weight"].cpu().numpy()


def plot_one(ax, W: np.ndarray, N: int) -> None:
    sorted_idx, bounds = cluster_hidden_units(W, k=N)
    W_perm = W[:, sorted_idx]
    A = np.abs(W_perm)
    vmax = np.percentile(A, 99)
    im = ax.imshow(A, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                    interpolation="nearest")
    # column (cluster) boundaries
    for b in bounds[1:-1]:
        ax.axvline(b - 0.5, color="cyan", lw=0.8, alpha=0.9)
    # row (frame) boundaries
    out_dim = W.shape[0]
    for f in range(1, out_dim // 10):
        ax.axhline(10 * f - 0.5, color="white", lw=0.6, alpha=0.7)
    ax.set_title(f"N = {N}   shape = ({W.shape[0]}, {W.shape[1]})",
                  fontsize=11)
    ax.set_xlabel("hidden unit (permuted)")
    ax.set_ylabel("output unit\n(row = 10·frame + class)")
    return im


def main(Ns: tuple[int, ...] = DEFAULT_N_LIST) -> None:
    # individual figures
    for N in Ns:
        W = load_fc2(N)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = plot_one(ax, W, N)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        out = FIGS / f"fc2_heatmap_N{N}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"wrote {out}")

    # combined grid for the slide
    fig, axes = plt.subplots(len(Ns), 1, figsize=(11, 3.0 * len(Ns)))
    if len(Ns) == 1:
        axes = [axes]
    for ax, N in zip(axes, Ns):
        W = load_fc2(N)
        im = plot_one(ax, W, N)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.suptitle(r"$|W_{fc2}|$ with hidden units reordered by inferred module"
                  "  (cyan: cluster boundary, white: frame boundary)",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "fc2_heatmaps_grid.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
