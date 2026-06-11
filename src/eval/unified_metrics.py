# -*- coding: utf-8 -*-
"""확장 평가지표: 기존 metrics.py 위에 확률·경보·진단 지표 추가.

추가 지표:
- Winkler Score: 예측 구간 품질 (구간 너비 + 커버리지 위반 패널티)
- PIT (Probability Integral Transform): 분위 보정 진단
- Sharpness: 예측 구간 평균 너비
- Energy Score: 다변량 확률 예보 스킬
- 오염 이벤트 탐지: POD, FAR, CSI, ETS, HSS
"""
from __future__ import annotations
import numpy as np


# ---------- 기존 metrics.py 재수출 ----------
from src.eval.metrics import (  # noqa: F401
    nse, rmse, mae, pbias, r2_corr, point_metrics,
    pinball_loss, crps_from_quantiles, coverage, calibration_error,
)


# ---------- RSR (RMSE–Standard deviation Ratio) ----------
def rsr(obs, sim):
    """RSR = RMSE / σ_obs = √(1 − NSE). 0=완벽, >0.7=미흡."""
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim); obs, sim = obs[m], sim[m]
    if len(obs) < 2:
        return np.nan
    sd = obs.std()
    return float(np.sqrt(np.mean((obs - sim) ** 2)) / sd) if sd > 0 else np.nan


# ---------- Winkler Score ----------
def winkler_score(obs, lower, upper, alpha: float = 0.2) -> float:
    """Winkler Score: 구간 너비 + 커버리지 위반 시 2/α 패널티.
    
    낮을수록 우수. alpha=0.2이면 80% 예측 구간.
    """
    obs = np.asarray(obs, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    m = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    if not m.any():
        return np.nan
    obs, lo, hi = obs[m], lower[m], upper[m]
    width = hi - lo
    penalty = np.zeros_like(width)
    below = obs < lo
    above = obs > hi
    penalty[below] = (2.0 / alpha) * (lo[below] - obs[below])
    penalty[above] = (2.0 / alpha) * (obs[above] - hi[above])
    return float(np.mean(width + penalty))


# ---------- Sharpness (평균 구간 너비) ----------
def sharpness(lower, upper) -> float:
    """예측 구간의 평균 너비. 좁을수록 우수(coverage가 동일할 때)."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    m = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean(upper[m] - lower[m])) if m.any() else np.nan


# ---------- PIT (Probability Integral Transform) ----------
def pit_values(obs, q_levels, q_matrix) -> np.ndarray:
    """각 관측에 대한 PIT 값 계산. 보정 완벽 → Uniform(0,1).
    
    q_matrix: [n, n_quantiles], q_levels: 1D quantile levels.
    PIT = F̂(y) ≈ 선형보간으로 관측이 어떤 분위에 해당하는지 추정.
    """
    obs = np.asarray(obs, float)
    q_levels = np.asarray(q_levels, float)
    q_matrix = np.asarray(q_matrix, float)
    n = len(obs)
    pit = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(obs[i]) or not np.isfinite(q_matrix[i]).all():
            continue
        y = obs[i]
        qs = q_matrix[i]
        if y <= qs[0]:
            pit[i] = q_levels[0] * (y / qs[0]) if qs[0] != 0 else 0.0
        elif y >= qs[-1]:
            pit[i] = q_levels[-1] + (1 - q_levels[-1]) * min(1.0, (y - qs[-1]) / max(abs(qs[-1]), 1e-8))
        else:
            # 선형보간
            j = np.searchsorted(qs, y) - 1
            j = max(0, min(j, len(qs) - 2))
            frac = (y - qs[j]) / (qs[j + 1] - qs[j]) if qs[j + 1] != qs[j] else 0.5
            pit[i] = q_levels[j] + frac * (q_levels[j + 1] - q_levels[j])
    return pit


def pit_reliability(pit_values: np.ndarray, n_bins: int = 10) -> dict:
    """PIT 히스토그램 기반 신뢰성 진단.
    
    Returns dict with bin_edges, counts, deviation (0=완벽 균일).
    """
    vals = pit_values[np.isfinite(pit_values)]
    if len(vals) < 10:
        return {"deviation": np.nan}
    counts, edges = np.histogram(vals, bins=n_bins, range=(0, 1))
    expected = len(vals) / n_bins
    deviation = float(np.mean(np.abs(counts - expected)) / expected)  # 균일편차 비율
    return {"bin_edges": edges.tolist(), "counts": counts.tolist(),
            "deviation": deviation, "n": len(vals)}


# ---------- 오염 이벤트 탐지 지표 ----------
def _contingency(y_true, y_pred):
    """2x2 분할표. y_true, y_pred: binary (0/1)."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def pod(y_true, y_pred) -> float:
    """Probability of Detection (Hit Rate) = TP / (TP + FN)."""
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def far(y_true, y_pred) -> float:
    """False Alarm Ratio = FP / (TP + FP)."""
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    return fp / (tp + fp) if (tp + fp) > 0 else np.nan


def csi(y_true, y_pred) -> float:
    """Critical Success Index (Threat Score) = TP / (TP + FP + FN)."""
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan


def ets(y_true, y_pred) -> float:
    """Equitable Threat Score = (TP - TP_random) / (TP + FP + FN - TP_random).
    
    TP_random = (TP+FP)(TP+FN) / N. 우연보다 좋은 정도. >0이면 스킬 있음.
    """
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    n = tp + fp + fn + tn
    if n == 0:
        return np.nan
    tp_rand = (tp + fp) * (tp + fn) / n
    denom = tp + fp + fn - tp_rand
    return float((tp - tp_rand) / denom) if denom != 0 else np.nan


def hss(y_true, y_pred) -> float:
    """Heidke Skill Score = 2(TP·TN - FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN)).
    
    >0이면 우연보다 우수, 1=완벽. 불균형 데이터에서도 안정적.
    """
    tp, fp, fn, tn = _contingency(y_true, y_pred)
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return float(num / den) if den != 0 else np.nan


def event_detection_metrics(y_true, y_pred) -> dict:
    """오염 이벤트 탐지 종합 지표."""
    return {
        "pod": pod(y_true, y_pred),
        "far": far(y_true, y_pred),
        "csi": csi(y_true, y_pred),
        "ets": ets(y_true, y_pred),
        "hss": hss(y_true, y_pred),
    }


# ---------- CRPS Skill Score ----------
def crps_skill_score(crps_model: float, crps_ref: float) -> float:
    """CRPS-skill = 1 - CRPS_model / CRPS_ref. >0이면 참조보다 우수."""
    if crps_ref is None or np.isnan(crps_ref) or crps_ref <= 0:
        return np.nan
    return float(1.0 - crps_model / crps_ref)


# ---------- 종합 점추정+확률 지표 ----------
def comprehensive_metrics(obs, sim, q_levels=None, q_matrix=None,
                          lower80=None, upper80=None,
                          lower90=None, upper90=None) -> dict:
    """점추정 + 확률(분위 있으면) + 구간(있으면) 종합 지표 사전 반환."""
    m = {}
    m.update(point_metrics(obs, sim))
    m["rsr"] = rsr(obs, sim)

    if q_levels is not None and q_matrix is not None:
        m["crps"] = crps_from_quantiles(obs, q_levels, q_matrix)
        m["calib_err"] = calibration_error(obs, q_levels, q_matrix)
        pv = pit_values(obs, q_levels, q_matrix)
        pr = pit_reliability(pv)
        m["pit_deviation"] = pr["deviation"]

    if lower80 is not None and upper80 is not None:
        m["cov80"] = coverage(obs, lower80, upper80)
        m["winkler80"] = winkler_score(obs, lower80, upper80, alpha=0.2)
        m["sharpness80"] = sharpness(lower80, upper80)

    if lower90 is not None and upper90 is not None:
        m["cov90"] = coverage(obs, lower90, upper90)
        m["winkler90"] = winkler_score(obs, lower90, upper90, alpha=0.1)
        m["sharpness90"] = sharpness(lower90, upper90)

    return m
