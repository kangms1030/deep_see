# -*- coding: utf-8 -*-
"""경보 엔진: 보정분위 → 초과확률 → 다horizon·리드타임·등급·비용가중 α·가드레일.

- 초과확률: thresholds.exceed_prob_from_quantiles(보정 분위 곡선 사용).
- 비용가중 α: val 표본에서 기대비용(c_FN·FN + c_FP·FP) 최소화로 타깃별 α* 결정.
- 등급: config.ALERT_LEVELS(관심/주의/경계). 리드타임: 가장 이른 초과예상일.
- 가드레일: 모델이 persistence 미달인 (지점,타깃)은 low_confidence(상위 replay/scorecard에서 판정).
"""
from __future__ import annotations
import numpy as np

from src.system import config as C
from src.alert import thresholds as TH


def calibrated_curve(raw_q: dict, cal: dict) -> np.ndarray:
    """보정 분위 곡선(7레벨): 외곽(0.05/0.1/0.9/0.95)은 보정, 내부(0.25/0.5/0.75)는 원분위."""
    m = {0.05: cal.get("q0.05_cal"), 0.1: cal.get("q0.1_cal"),
         0.9: cal.get("q0.9_cal"), 0.95: cal.get("q0.95_cal")}
    return np.array([m[q] if (q in m and m[q] is not None and np.isfinite(m[q])) else raw_q[q]
                     for q in C.Q], float)


def exceed_prob(qvals: np.ndarray, thr: float, direction: str) -> float:
    return TH.exceed_prob_from_quantiles(qvals, C.Q, thr, direction)


def cost_optimal_alpha(p: np.ndarray, y: np.ndarray, target: str) -> float:
    """기대비용 최소 α. p:초과확률, y:실제 이벤트(0/1). 단일클래스면 기본값."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m].astype(int)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return C.ALPHA_DEFAULT
    c = C.cost_ratio(target)          # c_FN/c_FP, c_FP=1
    best_a, best_cost = C.ALPHA_DEFAULT, np.inf
    for a in C.ALPHA_GRID:
        pred = (p >= a).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = c * fn + fp
        if cost < best_cost:
            best_cost, best_a = cost, float(a)
    return best_a


def origin_summary(p_by_day: dict, alpha: float):
    """origin-타깃 요약: (fired_any, lead_day, max_level).

    lead_day = p>=alpha가 처음 잡히는 가장 이른 horizon(=리드타임).
    max_level = 전 horizon 중 최고 경보 등급.
    """
    fired_days = [d for d in sorted(p_by_day) if np.isfinite(p_by_day[d]) and p_by_day[d] >= alpha]
    lead = min(fired_days) if fired_days else np.nan
    levels = [C.alert_level(p_by_day[d]) for d in p_by_day if np.isfinite(p_by_day[d])]
    order = {name: i for i, (name, _) in enumerate(C.ALERT_LEVELS)}     # 0=경계(최고)
    present = [lv for lv in levels if lv is not None]
    max_level = min(present, key=lambda lv: order[lv]) if present else None
    return (len(fired_days) > 0), lead, max_level
