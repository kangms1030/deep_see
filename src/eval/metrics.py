# -*- coding: utf-8 -*-
"""공통 평가지표: 점추정(NSE/RMSE/MAE/R²/PBIAS) + 확률(CRPS/pinball/coverage/calibration).

레거시 사용설명서와 동일한 NSE 정의를 사용(5일차 일평균 기준은 호출부에서 처리).
관측 결측(NaN)은 모든 지표에서 자동 제외(observed-only 채점).
"""
from __future__ import annotations
import numpy as np


def _valid(obs, sim):
    obs = np.asarray(obs, dtype=float); sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim):
    obs, sim = _valid(obs, sim)
    if obs.size < 2:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def rmse(obs, sim):
    obs, sim = _valid(obs, sim)
    return float(np.sqrt(np.mean((obs - sim) ** 2))) if obs.size else np.nan


def mae(obs, sim):
    obs, sim = _valid(obs, sim)
    return float(np.mean(np.abs(obs - sim))) if obs.size else np.nan


def pbias(obs, sim):
    obs, sim = _valid(obs, sim)
    s = obs.sum()
    return float((obs - sim).sum() / s * 100.0) if s != 0 else np.nan


def r2_corr(obs, sim):
    """결정계수(피어슨 상관 제곱)."""
    obs, sim = _valid(obs, sim)
    if obs.size < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return float(r ** 2)


def point_metrics(obs, sim) -> dict:
    return {"nse": nse(obs, sim), "rmse": rmse(obs, sim), "mae": mae(obs, sim),
            "r2": r2_corr(obs, sim), "pbias": pbias(obs, sim),
            "n": int(np.isfinite(np.asarray(obs, float) + np.asarray(sim, float)).sum())}


# ---------- 확률 지표 (Chronos 분위 예측용) ----------
def pinball_loss(obs, q_pred: dict) -> float:
    """q_pred: {quantile_level: array}. 평균 pinball(quantile) loss."""
    obs = np.asarray(obs, float)
    losses = []
    for ql, pred in q_pred.items():
        pred = np.asarray(pred, float)
        m = np.isfinite(obs) & np.isfinite(pred)
        if not m.any():
            continue
        e = obs[m] - pred[m]
        losses.append(np.mean(np.maximum(ql * e, (ql - 1) * e)))
    return float(np.mean(losses)) if losses else np.nan


def crps_from_quantiles(obs, q_levels, q_matrix) -> float:
    """분위 근사 CRPS = 2 * 평균 pinball loss(분위 격자에 대해).

    q_matrix: shape [n, n_quantiles] (각 분위 예측). q_levels: 1D.
    """
    obs = np.asarray(obs, float)
    q_levels = np.asarray(q_levels, float)
    q_matrix = np.asarray(q_matrix, float)
    vals = []
    for j, ql in enumerate(q_levels):
        pred = q_matrix[:, j]
        m = np.isfinite(obs) & np.isfinite(pred)
        if not m.any():
            continue
        e = obs[m] - pred[m]
        vals.append(np.mean(np.maximum(ql * e, (ql - 1) * e)))
    return float(2.0 * np.mean(vals)) if vals else np.nan


def coverage(obs, lower, upper) -> float:
    obs = np.asarray(obs, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    m = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    if not m.any():
        return np.nan
    return float(np.mean((obs[m] >= lower[m]) & (obs[m] <= upper[m])))


def calibration_error(obs, q_levels, q_matrix) -> float:
    """각 분위에서 (실제 누적비율 - 명목분위) 절대오차 평균."""
    obs = np.asarray(obs, float)
    q_levels = np.asarray(q_levels, float)
    q_matrix = np.asarray(q_matrix, float)
    errs = []
    for j, ql in enumerate(q_levels):
        pred = q_matrix[:, j]
        m = np.isfinite(obs) & np.isfinite(pred)
        if not m.any():
            continue
        emp = np.mean(obs[m] <= pred[m])
        errs.append(abs(emp - ql))
    return float(np.mean(errs)) if errs else np.nan
