# -*- coding: utf-8 -*-
"""레거시 GAIN(결측 보간) — Yoon et al. 2018 정식 알고리즘의 PyTorch 충실 재구현.

레거시 repo의 core/util.py에 있는 binary_sampler/uniform_sampler/sample_batch_index가
Yoon의 reference 구현과 동일 → 본 모듈은 그 row-wise GAIN을 PyTorch로 옮긴 것.
하이퍼파라미터는 사용설명서 기준(miss_rate=0.15 학습용 인위결측, batch=128, alpha=100, hint=0.9).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from src.utils.progress import log
from src.utils import gpu


def _mlp(din, dh, dout):
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(),
                         nn.Linear(dh, dh), nn.ReLU(),
                         nn.Linear(dh, dout), nn.Sigmoid())


class GAINImputer:
    def __init__(self, dim, alpha=100.0, hint_rate=0.9, batch_size=128,
                 iterations=5000, device="cuda", task="legacy"):
        self.dim = dim
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.bs = batch_size
        self.iters = iterations
        self.device = device if torch.cuda.is_available() else "cpu"
        self.task = task
        h = dim
        self.G = _mlp(dim * 2, h, dim).to(self.device)
        self.D = _mlp(dim * 2, h, dim).to(self.device)

    def fit(self, X: np.ndarray) -> "GAINImputer":
        """train 구간으로 컬럼 정규화 통계 학습 + G/D 적합(누수 방지)."""
        X = np.asarray(X, dtype=np.float64)
        n, dim = X.shape
        M = (~np.isnan(X)).astype(np.float64)               # 1=관측
        cmin = np.nanmin(np.where(M == 1, X, np.nan), axis=0)
        cmax = np.nanmax(np.where(M == 1, X, np.nan), axis=0)
        cmin = np.nan_to_num(cmin, nan=0.0)
        rng = np.where((cmax - cmin) == 0, 1.0, cmax - cmin)
        rng = np.nan_to_num(rng, nan=1.0)
        self.cmin, self.rng = cmin, rng
        Xn = (np.nan_to_num(X, nan=0.0) - cmin) / rng
        Xn = Xn * M

        dev = self.device
        Xt = torch.tensor(Xn, dtype=torch.float32, device=dev)
        Mt = torch.tensor(M, dtype=torch.float32, device=dev)
        optG = torch.optim.Adam(self.G.parameters())
        optD = torch.optim.Adam(self.D.parameters())
        eps = 1e-8

        log(f"GAIN 학습 시작 n={n} dim={dim} iters={self.iters} dev={dev}", self.task)
        last = -5
        for it in range(self.iters):
            idx = torch.randint(0, n, (min(self.bs, n),), device=dev)
            x = Xt[idx]; m = Mt[idx]
            z = torch.rand_like(x) * 0.01
            xin = m * x + (1 - m) * z
            # hint
            b = (torch.rand_like(x) < self.hint_rate).float()
            h = m * b + 0.5 * (1 - b)

            # --- D step ---
            g = self.G(torch.cat([xin, m], dim=1))
            xhat = xin * m + g.detach() * (1 - m)
            d = self.D(torch.cat([xhat, h], dim=1))
            d_loss = -torch.mean(m * torch.log(d + eps) + (1 - m) * torch.log(1 - d + eps))
            optD.zero_grad(); d_loss.backward(); optD.step()

            # --- G step ---
            g = self.G(torch.cat([xin, m], dim=1))
            xhat = xin * m + g * (1 - m)
            d = self.D(torch.cat([xhat, h], dim=1))
            g_adv = -torch.mean((1 - m) * torch.log(d + eps))
            g_mse = torch.sum((m * x - m * g) ** 2) / (torch.sum(m) + eps)
            g_loss = g_adv + self.alpha * g_mse
            optG.zero_grad(); g_loss.backward(); optG.step()

            pct = int((it + 1) * 100 / self.iters)
            if pct >= last + 10:
                last = pct
                log(f"GAIN {pct:3d}% it={it+1}/{self.iters} D={d_loss.item():.3f} "
                    f"G_mse={g_mse.item():.4f} | {gpu.fmt()}", self.task)
                gpu.check(where="GAIN")
        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray) -> np.ndarray:
        """fit에서 학습한 G/통계로 X 보간(관측은 원값 유지). 재학습 없음(누수 방지)."""
        X = np.asarray(X, dtype=np.float64)
        M = (~np.isnan(X)).astype(np.float64)
        Xn = ((np.nan_to_num(X, nan=0.0) - self.cmin) / self.rng) * M
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        Mt = torch.tensor(M, dtype=torch.float32, device=self.device)
        z = torch.rand_like(Xt) * 0.01
        xin = Mt * Xt + (1 - Mt) * z
        g = self.G(torch.cat([xin, Mt], dim=1))
        imp = (Mt * Xt + (1 - Mt) * g).cpu().numpy()
        imp = imp * self.rng + self.cmin
        return np.where(M == 1, X, imp)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
