"""One training loop shared by every network.

Batches are dicts so later networks can take extra inputs without a new loop:
    {"x": trial, "y": label, "rest": resting signal, "subject": id}
A model receives the batch and returns logits. Plain baselines only read "x".
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TrialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, subject: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.subject = torch.from_numpy(subject)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i], "subject": self.subject[i]}


def forward(model: nn.Module, batch: dict) -> torch.Tensor:
    """Baselines take a tensor; later networks can opt into the whole batch."""
    if getattr(model, "takes_batch", False):
        return model(batch)
    return model(batch["x"])


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    seed: int = 0


def _to(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, subs = [], [], []
    for batch in loader:
        batch = _to(batch, device)
        preds.append(forward(model, batch).argmax(1).cpu())
        ys.append(batch["y"].cpu())
        subs.append(batch["subject"].cpu())
    return torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(subs).numpy()


def fit(model: nn.Module, train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig, device=None,
        log_every: int = 10) -> tuple[nn.Module, dict]:
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss()

    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    best = {"val_acc": -1.0, "epoch": -1}
    best_state = None
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_dl:
            batch = _to(batch, device)
            loss = loss_fn(forward(model, batch), batch["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch["y"])
            n += len(batch["y"])
        sched.step()

        p, y, _ = predict(model, val_dl, device)
        val_acc = float((p == y).mean())
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": val_acc})
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  loss {total / n:.4f}  val_acc {val_acc:.3f}")
        if epoch - best["epoch"] >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return model, {"best": best, "history": history}
