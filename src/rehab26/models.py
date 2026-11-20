"""Models. Every model takes a sequence batch x of shape (B, F, T) and a one-hot exercise code (B, 3).

Each model standardizes its own input with statistics fitted on the training fold (`fit_scaler`), so
augmentation can be applied to unstandardized sequences before the model sees them.
"""
import torch
import torch.nn as nn


def summarize(x):
    """(B, F, T) -> (B, 4F): mean, min, max, range of every feature over time."""
    mn, mx = x.amin(dim=2), x.amax(dim=2)
    return torch.cat([x.mean(dim=2), mn, mx, mx - mn], dim=1)


class _Scaled(nn.Module):
    def __init__(self, n_scaled):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_scaled))
        self.register_buffer("sd", torch.ones(n_scaled))

    def _scaler_input(self, x):
        """What gets standardized, shaped (..., n_scaled) on the last axis."""
        return x.transpose(1, 2)

    @torch.no_grad()
    def fit_scaler(self, x_train):
        z = self._scaler_input(x_train).reshape(-1, self.mu.numel())
        self.mu.copy_(z.mean(0))
        self.sd.copy_(z.std(0, unbiased=False) + 1e-6)


class SummaryMLP(_Scaled):
    """MLP on per-repetition summary statistics. Sees how far things moved, not in what order."""

    def __init__(self, n_feat, n_ex=3, hidden=(128, 64), dropout=0.3):
        super().__init__(4 * n_feat)
        layers, d = [], 4 * n_feat + n_ex
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def _scaler_input(self, x):
        return summarize(x)

    def forward(self, x, ex):
        z = (summarize(x) - self.mu) / self.sd
        return self.net(torch.cat([z, ex], dim=1)).squeeze(1)


class CNN1D(_Scaled):
    """Three 1D convolution blocks over time, then global average and max pooling."""

    def __init__(self, n_feat, n_ex=3, widths=(64, 128, 128), kernel=5, dropout=0.3):
        super().__init__(n_feat)
        blocks, d = [], n_feat
        for i, w in enumerate(widths):
            blocks += [nn.Conv1d(d, w, kernel, padding=kernel // 2), nn.BatchNorm1d(w), nn.ReLU()]
            if i < len(widths) - 1:
                blocks.append(nn.MaxPool1d(2))
            d = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d + n_ex, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, ex):
        z = (x - self.mu[:, None]) / self.sd[:, None]
        h = self.features(z)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2), ex], dim=1)
        return self.head(h).squeeze(1)


class BiGRUAttention(_Scaled):
    """Bidirectional GRU with attention pooling over time."""

    def __init__(self, n_feat, n_ex=3, hidden=64, layers=2, dropout=0.3):
        super().__init__(n_feat)
        self.proj = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden + n_ex, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, ex, return_attention=False):
        z = (x.transpose(1, 2) - self.mu) / self.sd
        h, _ = self.gru(self.proj(z))
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        out = self.head(torch.cat([pooled, ex], dim=1)).squeeze(1)
        return (out, w) if return_attention else out


MODELS = {"mlp": SummaryMLP, "cnn": CNN1D, "bigru": BiGRUAttention}


def build_model(kind, n_feat, **kwargs):
    return MODELS[kind](n_feat, **kwargs)
