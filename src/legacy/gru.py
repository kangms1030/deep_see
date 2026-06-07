# -*- coding: utf-8 -*-
"""레거시 GRU 예측모델 PyTorch 재구현.

레거시 models.py GRUModel: GRU(256, ret_seq=True) → GRU(256) → Dropout(0.5)
→ Dense(OUT_STEPS*1) → Reshape. (recurrent_dropout=0.5는 PyTorch GRU의 층간 dropout으로 근사)
손실 MSE, Adam. 입력 240h → 출력 120h 단일 타깃.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.progress import log
from src.utils import gpu


class GRUForecaster(nn.Module):
    def __init__(self, n_features, out_steps=120, hidden=256, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(hidden, out_steps)
        self.out_steps = out_steps

    def forward(self, x):                       # x: [B, T, F]
        out, _ = self.gru(x)
        last = out[:, -1, :]                    # [B, H]
        return self.head(self.drop(last))       # [B, out_steps]


def train_gru(model, Xtr, Ytr, Xva, Yva, epochs=30, batch_size=128, lr=1e-3,
              patience=6, device="cuda", task="legacy", tag=""):
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    Xva_t = torch.tensor(Xva, device=device); Yva_t = torch.tensor(Yva, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    best = float("inf"); best_state = None; bad = 0
    gpu.reset_peak(); n_batches = len(dl)
    log(f"GRU 학습 {tag}: epochs={epochs} batches/epoch={n_batches} dev={device} "
        f"Xtr={Xtr.shape}", task)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for bi, (xb, yb) in enumerate(dl, 1):
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossfn(pred, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
            if bi % max(1, n_batches // 3) == 0:
                gpu.check(where=f"GRU {tag}")
        model.eval()
        with torch.no_grad():
            vloss = lossfn(model(Xva_t), Yva_t).item() if len(Xva) else float("nan")
        tr = tot / max(1, len(ds))
        pct = int(ep * 100 / epochs)
        log(f"GRU {tag} {pct:3d}% ep={ep}/{epochs} train_mse={tr:.4f} val_mse={vloss:.4f} "
            f"| {gpu.fmt()}", task)
        if vloss < best - 1e-6:
            best = vloss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"GRU {tag} early stop @ep{ep} (best val_mse={best:.4f})", task)
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, device="cuda", batch_size=256):
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], device=device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0, model.out_steps))
