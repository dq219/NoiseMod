import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import ModularResNet18
from pathlib import Path
import tqdm as tqdm
import time

# get similarity matrix from the weight matrix
def getS(mat):
    x = np.abs(mat) + 1e-7
    norm = np.sqrt(np.sum(x ** 2, axis = 0)[:, np.newaxis] @  np.sum(x ** 2, axis = 0)[np.newaxis, :])
    return x.T @ x / norm

# get random walk Laplacian
def getL(S):
    D = np.diag(S.sum(axis=1))
    D_inv = np.diag(1.0 / np.diag(D))
    res = D_inv @ S
    return np.eye(len(res)) - res

def laplacian_spectrum(weight: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    L = getL(getS(weight))
    return np.linalg.eigvalsh(L)


def fc2_eigvals(model: ModularResNet18, k: int = 10) -> np.ndarray:
    """First ``k`` Laplacian eigenvalues of fc2 (padded with NaN if needed)."""
    eig = laplacian_spectrum(model.fc2.weight.detach().cpu().numpy())
    out = np.full(k, np.nan)
    out[: min(k, len(eig))] = eig[: min(k, len(eig))]
    return out

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

def run_one(reg_weight, train_loader, test_loader, device, alpha: float = 2.0, D: int = 4, epochs: int = 500, N_TRACKED_EIGS: int = 10):
    run_name = f"run_D{D}_reg{reg_weight}"
    print(f"Running {run_name} on device {device}...")
    hidden_dim = 512 * D
    run_dir = Path('results') / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ModularResNet18(D=D, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epochs_arr = np.arange(epochs)
    test_acc_arr = np.full(epochs, np.nan)
    train_loss_arr = np.full(epochs, np.nan)
    train_ce_arr = np.full(epochs, np.nan)
    train_reg_arr = np.full(epochs, np.nan)
    eigvals_arr = np.full((epochs, N_TRACKED_EIGS), np.nan)
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "reg": 0.0}
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            stats = model.train_step(x, y, optimizer, alpha=alpha, reg_weight=reg_weight)
            for k, v in stats.items():
                running[k] += v
            n_batches += 1
        scheduler.step()

        avg = {k: v / n_batches for k, v in running.items()}
        test_acc = evaluate(model, test_loader, device)
        eigs = fc2_eigvals(model)

        train_loss_arr[epoch] = avg["loss"]
        train_ce_arr[epoch] = avg["ce"]
        train_reg_arr[epoch] = avg["reg"]
        test_acc_arr[epoch] = test_acc
        eigvals_arr[epoch] = eigs
        print(f"Epoch {epoch}: loss={avg['loss']:.4f}, ce={avg['ce']:.4f}, reg={avg['reg']:.4f}, test_acc={test_acc:.4f}, time={time.time() - t0:.2f}s")

    np.savez(
        run_dir / "metrics.npz",
        epoch=epochs_arr,
        test_acc=test_acc_arr,
        train_loss=train_loss_arr,
        train_ce=train_ce_arr,
        train_reg=train_reg_arr,
        eigvals=eigvals_arr,
        D=D,
        hidden_dim=hidden_dim,
        reg_weight=reg_weight,
    )
    torch.save(model.state_dict(), run_dir / "weights.pt")
    plt.figure()
    plt.plot(epochs_arr, test_acc_arr, label="Test Accuracy")
    plt.savefig(run_dir / "test_acc.png")
    plt.close()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(min(N_TRACKED_EIGS, axes.shape[0])):
        axes[i].plot(epochs_arr, eigvals_arr[:, i])
        axes[i].set_title(f"Eigenvalue {i}")
    plt.savefig(run_dir / "eigvals.png")
    plt.close()