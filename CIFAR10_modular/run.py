"""Train ModularResNet18 on ConcatCIFAR10 and quantify FC-layer modularity.

The MLP head is ``fc1(512*N -> hidden_dim) -> LeakyReLU -> fc2(hidden_dim -> 10*N)``.
The weighted-activity regularizer is applied only to the hidden activation
(post-LeakyReLU output of fc1) using the squared weights of fc2.

Modularity of the resulting fc2 is measured spectrally, mirroring
``modularisation_via_noise/utils/graphs.py``:

  1. Build a cosine-similarity matrix S from |fc2.weight|, treating each
     hidden unit as a node. Two hidden units are similar when they project
     to overlapping subsets of fc2 outputs.
  2. Compute the symmetric normalized Laplacian L = I - D^{-1/2} S D^{-1/2}
     (same spectrum as the random-walk Laplacian I - D^{-1} S used in
     graphs.py, but symmetric so eigvalsh is exact).
  3. A graph with N disconnected components has its first N Laplacian
     eigenvalues equal to 0, so for an N-module FC layer the N-th
     eigenvalue should be small and the (N+1)-th should jump.

Per-run outputs land in ``<results-root>/<run-name>/``:
  * config.json    — all hyperparameters
  * train.log      — per-epoch stdout (loss, ce, reg, test_acc, eigenvalues)
  * metrics.npz    — test_acc, train_loss/ce/reg, eigvals (epochs+1, 10)
  * weights.pt     — final state_dict
  * accuracy.png   — test accuracy over training
  * eigenvalues.png — first 10 Laplacian eigenvalues, 2 rows x 5 cols
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ConcatCIFAR10
from model import ModularResNet18


N_TRACKED_EIGS = 10


def evaluate(model: ModularResNet18, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def laplacian_spectrum(weight: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Sorted eigenvalues of the symmetric-normalized Laplacian on |W|.

    Nodes are the **input** units of the layer (clustered by their outgoing
    projection pattern). For ``weight`` of shape ``(out_dim, in_dim)`` the
    returned eigenvalues have length ``in_dim`` (ascending).
    """
    W = np.abs(weight) + eps
    col_sq = (W ** 2).sum(axis=0)
    norm = np.sqrt(np.outer(col_sq, col_sq))
    S = (W.T @ W) / norm
    d = S.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    L = np.eye(len(S)) - (d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :])
    L = 0.5 * (L + L.T)
    return np.linalg.eigvalsh(L)


def fc2_eigvals(model: ModularResNet18, k: int = N_TRACKED_EIGS) -> np.ndarray:
    """First ``k`` Laplacian eigenvalues of fc2 (padded with NaN if needed)."""
    eig = laplacian_spectrum(model.fc2.weight.detach().cpu().numpy())
    out = np.full(k, np.nan)
    out[: min(k, len(eig))] = eig[: min(k, len(eig))]
    return out


def default_run_name(N: int, reg_weight: float, hidden_dim: int) -> str:
    return f"N{N}_reg{reg_weight:g}_h{hidden_dim}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def save_plots(run_dir: Path, epochs: np.ndarray, test_acc: np.ndarray,
               eigvals: np.ndarray, run_name: str) -> None:
    """Write accuracy.png and eigenvalues.png into ``run_dir``."""
    # accuracy — skip epoch 0 (init, not evaluated)
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ~np.isnan(test_acc)
    ax.plot(epochs[valid], test_acc[valid], marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (per-frame)")
    ax.set_title(f"{run_name}: test accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy.png", dpi=120)
    plt.close(fig)

    # 10 eigenvalues, 2 rows x 5 cols, one panel per eigenvalue index
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(epochs, eigvals[:, i], marker="o", markersize=2.5)
        ax.set_title(f"$\\lambda_{{{i + 1}}}$")
        ax.grid(True, alpha=0.3)
        if i >= 5:
            ax.set_xlabel("epoch")
        if i % 5 == 0:
            ax.set_ylabel("eigenvalue")
    fig.suptitle(f"{run_name}: first {N_TRACKED_EIGS} Laplacian eigenvalues of fc2",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(run_dir / "eigenvalues.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2, help="frames per concatenated sample")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="exponent for the weighted activity regularizer")
    parser.add_argument("--reg-weight", type=float, default=1.0,
                        help="scalar multiplier on the weighted activity regularizer")
    parser.add_argument("--hidden-dim", type=int, default=-1,
                        help="hidden width of the MLP head; -1 (default) means 512*N")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--results-root", type=str, default="./results")
    parser.add_argument("--run-name", type=str, default="",
                        help="subfolder under results-root; defaults to N{N}_reg{reg}_h{hidden}")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    hidden_dim = 512 * args.N if args.hidden_dim < 0 else args.hidden_dim

    run_name = args.run_name or default_run_name(args.N, args.reg_weight, hidden_dim)
    run_dir = Path(args.results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {**vars(args), "hidden_dim_resolved": hidden_dim, "run_name": run_name}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    log_file = open(run_dir / "train.log", "w", buffering=1)  # line-buffered

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")

    train_set = ConcatCIFAR10(root=args.data_root, N=args.N, train=True)
    test_set = ConcatCIFAR10(root=args.data_root, N=args.N, train=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = ModularResNet18(N=args.N, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Per-epoch buffers (epoch 0 = init; epochs 1..E = post-train)
    n_rows = args.epochs + 1
    epochs_arr = np.arange(n_rows)
    test_acc_arr = np.full(n_rows, np.nan)
    train_loss_arr = np.full(n_rows, np.nan)
    train_ce_arr = np.full(n_rows, np.nan)
    train_reg_arr = np.full(n_rows, np.nan)
    eigvals_arr = np.full((n_rows, N_TRACKED_EIGS), np.nan)

    eigvals_arr[0] = fc2_eigvals(model)
    log(f"run_dir={run_dir} device={device} N={args.N} hidden_dim={hidden_dim} "
        f"reg_weight={args.reg_weight} alpha={args.alpha} epochs={args.epochs}")
    log(f"epoch  0 (init)            | first {N_TRACKED_EIGS} eigs "
        f"= [{', '.join(f'{v:.4f}' for v in eigvals_arr[0])}]")

    epoch_times = np.full(args.epochs, np.nan)
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "reg": 0.0}
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stats = model.train_step(
                x, y, optimizer, alpha=args.alpha, reg_weight=args.reg_weight,
            )
            for k, v in stats.items():
                running[k] += v
            n_batches += 1
        scheduler.step()

        avg = {k: v / n_batches for k, v in running.items()}
        test_acc = evaluate(model, test_loader, device)
        eigs = fc2_eigvals(model)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times[epoch - 1] = epoch_time

        train_loss_arr[epoch] = avg["loss"]
        train_ce_arr[epoch] = avg["ce"]
        train_reg_arr[epoch] = avg["reg"]
        test_acc_arr[epoch] = test_acc
        eigvals_arr[epoch] = eigs

        mean_time = float(np.nanmean(epoch_times[:epoch]))
        eta = mean_time * (args.epochs - epoch)
        total_est = mean_time * args.epochs
        log(
            f"epoch {epoch:>3}/{args.epochs} "
            f"| loss {avg['loss']:.4f} ce {avg['ce']:.4f} reg {avg['reg']:.4e} "
            f"| test_acc {test_acc:.4f} "
            f"| lam_1 {eigs[0]:.4f} lam_N {eigs[args.N - 1]:.4f} "
            f"lam_N+1 {eigs[args.N]:.4f} "
            f"gap {eigs[args.N] - eigs[args.N - 1]:.4f} "
            f"| t_epoch {epoch_time:5.1f}s "
            f"eta {format_duration(eta)} (total ~{format_duration(total_est)})"
        )
    total_train_time = time.perf_counter() - train_start
    log(f"training complete in {format_duration(total_train_time)} "
        f"(mean {float(np.nanmean(epoch_times)):.1f}s/epoch over {args.epochs} epochs)")

    np.savez(
        run_dir / "metrics.npz",
        epoch=epochs_arr,
        test_acc=test_acc_arr,
        train_loss=train_loss_arr,
        train_ce=train_ce_arr,
        train_reg=train_reg_arr,
        eigvals=eigvals_arr,
        epoch_time_seconds=epoch_times,
        N=args.N,
        hidden_dim=hidden_dim,
        reg_weight=args.reg_weight,
    )
    torch.save(model.state_dict(), run_dir / "weights.pt")
    save_plots(run_dir, epochs_arr, test_acc_arr, eigvals_arr, run_name)

    log(f"saved metrics + weights + plots to {run_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
