# -*- coding: utf-8 -*-
"""수질 임계값 정의(하천 생활환경 기준 등급 경계 + 조류 우려 기준).

direction:
- 'high' : 값이 임계 초과 시 위험(오염↑). TOC/T-N/T-P/Chl-a.
- 'low'  : 값이 임계 미만 시 위험(저산소). DO.
data-driven 대안: 지점별 관측 분위(상위 90% 또는 DO는 하위 10%).
"""
from __future__ import annotations
import numpy as np

THRESHOLDS = {
    "do":    {"value": 5.0,  "direction": "low",  "unit": "mg/L",
              "desc": "DO < 5.0 mg/L (생활환경 '보통' 경계·저산소 위험)"},
    "toc":   {"value": 6.0,  "direction": "high", "unit": "mg/L",
              "desc": "TOC > 6.0 mg/L ('나쁨' 등급 경계)"},
    "tn":    {"value": 3.0,  "direction": "high", "unit": "mg/L",
              "desc": "T-N > 3.0 mg/L (부영양 위험)"},
    "tp":    {"value": 0.1,  "direction": "high", "unit": "mg/L",
              "desc": "T-P > 0.1 mg/L (III등급 경계·부영양)"},
    "chl-a": {"value": 20.0, "direction": "high", "unit": "mg/m^3",
              "desc": "Chl-a > 20 µg/L (조류 우려)"},
}


def get_threshold(target: str, obs_series: np.ndarray | None = None, mode: str = "standard"):
    """mode='standard'(고정 기준) 또는 'percentile'(지점 관측 분위)."""
    base = THRESHOLDS[target]
    if mode == "percentile" and obs_series is not None:
        v = obs_series[np.isfinite(obs_series)]
        if v.size > 20:
            pct = 10 if base["direction"] == "low" else 90
            return {**base, "value": float(np.percentile(v, pct)), "mode": "percentile"}
    return {**base, "mode": "standard"}


def exceed_prob_from_quantiles(qvals, qlevels, thr, direction) -> float:
    """분위값(qvals)·분위수(qlevels)로부터 P(위험) 추정.

    cdf(thr)=P(X<=thr)를 분위격자 선형보간으로 구함.
    high: P(X>thr)=1-cdf ; low: P(X<thr)=cdf.
    """
    qvals = np.asarray(qvals, float); qlevels = np.asarray(qlevels, float)
    m = np.isfinite(qvals)
    if m.sum() < 2:
        return np.nan
    order = np.argsort(qvals[m])
    v = qvals[m][order]; q = qlevels[m][order]
    cdf = float(np.interp(thr, v, q, left=0.0, right=1.0))
    return cdf if direction == "low" else (1.0 - cdf)


def is_event(value, thr, direction) -> bool:
    if not np.isfinite(value):
        return False
    return (value < thr) if direction == "low" else (value > thr)
