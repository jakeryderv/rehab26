"""Batching, augmentation, and the training / evaluation loop shared by all experiments."""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .data import EX_INDEX
from .models import build_model


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    model_kwargs: dict = field(default_factory=dict)


class RepDataset(Dataset):
    """Unstandardized sequences (F, T), a one-hot exercise code, and the label."""

    def __init__(self, seq, exercise_ids, y):
        self.x = torch.as_tensor(np.ascontiguousarray(seq.transpose(0, 2, 1)), dtype=torch.float32)
        self.ex = nn.functional.one_hot(torch.as_tensor([EX_INDEX[e] for e in exercise_ids]), 3).float()
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.ex[i], self.y[i]


def make_loaders(seq, meta, train_idx, val_idx, seed, batch_size=64):
    ex, y = meta.exercise_id.values, meta.y.values
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(RepDataset(seq[train_idx], ex[train_idx], y[train_idx]), batch_size=batch_size,
                       shuffle=True, generator=g)
    val = DataLoader(RepDataset(seq[val_idx], ex[val_idx], y[val_idx]), batch_size=512, shuffle=False)
    return train, val


class Augment:
    """Random training-time perturbations of a batch of unstandardized sequences (B, F, T).

    time_warp: strength of a smooth random speed-up / slow-down within the repetition (0 disables).
    jitter:    Gaussian noise, as a fraction of each feature's standard deviation.
    rotate:    maximum rotation about the vertical axis in degrees. Only valid for coordinate features,
               which are laid out as joint-major (x, y, z) triples.
    scale:     maximum relative change in overall size. Coordinate features only.
    """

    def __init__(self, time_warp=0.0, jitter=0.0, rotate=0.0, scale=0.0, feature_std=None):
        self.time_warp, self.jitter, self.rotate, self.scale = time_warp, jitter, rotate, scale
        self.feature_std = feature_std

    def __call__(self, x):
        b, f, t = x.shape
        if self.rotate or self.scale:
            p = x.view(b, f // 3, 3, t)
            theta = (torch.rand(b, 1, 1, device=x.device) * 2 - 1) * math.radians(self.rotate)
            s = 1 + (torch.rand(b, 1, 1, device=x.device) * 2 - 1) * self.scale
            c, sn = torch.cos(theta), torch.sin(theta)
            px, py, pz = p[:, :, 0], p[:, :, 1], p[:, :, 2]
            x = torch.stack([(c * px + sn * pz) * s, py * s, (-sn * px + c * pz) * s], dim=2).reshape(b, f, t)
        if self.time_warp:
            grid = torch.linspace(0, 1, t, device=x.device).expand(b, t)
            amp = torch.rand(b, 1, device=x.device) * self.time_warp
            phase = torch.rand(b, 1, device=x.device)
            # monotonic for amp < 1: derivative is 1 + amp * cos(...)
            pos = (grid + amp * torch.sin(2 * math.pi * (grid + phase)) / (2 * math.pi)
                   - amp * torch.sin(2 * math.pi * phase) / (2 * math.pi)).clamp(0, 1) * (t - 1)
            lo = pos.floor().long().clamp(max=t - 1)
            hi = (lo + 1).clamp(max=t - 1)
            w = (pos - lo).unsqueeze(1)
            idx_lo, idx_hi = lo.unsqueeze(1).expand(b, f, t), hi.unsqueeze(1).expand(b, f, t)
            x = x.gather(2, idx_lo) * (1 - w) + x.gather(2, idx_hi) * w
        if self.jitter:
            x = x + torch.randn_like(x) * self.jitter * self.feature_std[None, :, None]
        return x


def train_model(kind, seq, meta, train_idx, val_idx, seed, config=None, augment=None, device=None):
    """Train one model on one fold. Returns (per-epoch history, final validation probabilities, model)."""
    config = config or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader = make_loaders(seq, meta, train_idx, val_idx, seed, config.batch_size)
    model = build_model(kind, seq.shape[2], **config.model_kwargs).to(device)
    model.fit_scaler(train_loader.dataset.x.to(device))
    if augment is not None and augment.jitter:
        augment.feature_std = train_loader.dataset.x.std(dim=(0, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    y_val = meta.y.values[val_idx]
    history = []
    for epoch in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for x, ex, y in train_loader:
            x, ex, y = x.to(device), ex.to(device), y.to(device)
            if augment is not None:
                x = augment(x)
            opt.zero_grad()
            loss = loss_fn(model(x, ex), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(x.to(device), ex.to(device)) for x, ex, _ in val_loader]).cpu()
        prob = torch.sigmoid(logits).numpy()
        history.append(dict(epoch=epoch + 1, train_loss=total / count,
                            val_loss=loss_fn(logits, torch.as_tensor(y_val, dtype=torch.float32)).item(),
                            val_acc=accuracy_score(y_val, prob > 0.5)))
    return pd.DataFrame(history), prob, model


def run_condition(name, kind, seq, meta, folds, seeds=(0, 1, 2), config=None, augment=None, verbose=True):
    """Train one (model, features) condition on every fold and seed.

    Returns (scores, predictions, histories) as DataFrames tagged with the condition name.
    """
    t0 = time.time()
    scores, preds, hists = [], [], []
    for seed in seeds:
        for train_idx, val_idx, val_person in folds:
            hist, prob, _ = train_model(kind, seq, meta, train_idx, val_idx, seed, config, augment)
            y = meta.y.values[val_idx]
            pred = (prob > 0.5).astype(int)
            scores.append(dict(condition=name, seed=seed, val_person=val_person,
                               acc=accuracy_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, prob),
                               recall=pred[y == 1].mean(), specificity=1 - pred[y == 0].mean(),
                               train_loss=hist.train_loss.iloc[-1], val_loss=hist.val_loss.iloc[-1],
                               best_epoch_acc=hist.val_acc.max()))
            preds.append(meta.iloc[val_idx].assign(condition=name, seed=seed, prob=prob, pred=pred))
            hists.append(hist.assign(condition=name, seed=seed, val_person=val_person))
    scores = pd.DataFrame(scores)
    if verbose:
        print(f"{name:40s} val acc {scores.acc.mean():.3f}  ({time.time() - t0:.0f}s)")
    return scores, pd.concat(preds), pd.concat(hists)


def summarize_scores(scores, order=None):
    """One row per condition: means over folds and seeds, plus spread across participants and seeds."""
    g = scores.groupby("condition")
    out = g[["acc", "f1", "auc", "recall", "specificity", "train_loss", "val_loss", "best_epoch_acc"]].mean()
    out.insert(1, "acc_std_folds", scores.groupby(["condition", "val_person"]).acc.mean().unstack().std(axis=1))
    out.insert(2, "acc_std_seeds", scores.groupby(["condition", "seed"]).acc.mean().unstack().std(axis=1))
    return out.reindex(order) if order is not None else out
