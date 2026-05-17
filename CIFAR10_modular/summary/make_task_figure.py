"""Render one ConcatCIFAR10 sample each for N = 2, 4, 8 and write a single
figure to ``summary/figs/task_examples.png`` for the slide deck.

Uses the project's own ConcatCIFAR10 (no transforms) so the displayed image is
faithful to what the network sees minus normalization. A fixed seed is used so
the figure is reproducible run-to-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from data import ConcatCIFAR10  # noqa: E402

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def display_transform() -> transforms.Compose:
    # No normalization, no augmentation: we want a human-recognisable preview.
    return transforms.Compose([transforms.ToTensor()])


def render() -> Path:
    torch.manual_seed(7)
    np.random.seed(7)

    Ns = [2, 4, 8]
    samples: list[tuple[int, np.ndarray, list[int]]] = []
    for N in Ns:
        ds = ConcatCIFAR10(
            root=str(REPO / "data"), N=N, train=False,
            transform=display_transform(), download=False,
        )
        x, y = ds[0]  # random concat (seed fixed above)
        img = x.permute(1, 2, 0).cpu().numpy()  # (32, 32*N, 3)
        samples.append((N, img, y.tolist()))

    # One row per N. All images share height 32; widths grow with N. We use
    # gridspec with width_ratios = N to keep per-frame aspect square.
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(
        nrows=len(Ns), ncols=1, hspace=0.85, left=0.06, right=0.98,
        top=0.88, bottom=0.04,
    )
    for row, (N, img, labels) in enumerate(samples):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        # Draw a thin separator at every 32 px boundary + label each frame.
        for k in range(N):
            x0 = 32 * k
            if k > 0:
                ax.axvline(x0 - 0.5, color="white", lw=0.8)
            ax.text(
                x0 + 16, -4, CIFAR10_CLASSES[labels[k]],
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel(f"N = {N}", rotation=0, ha="right", va="center",
                       fontsize=12, labelpad=18)

    fig.suptitle(
        "ConcatCIFAR10: N CIFAR-10 frames concatenated horizontally, one label per frame",
        fontsize=12,
    )
    out = HERE / "figs" / "task_examples.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = render()
    print(f"wrote {path}")
