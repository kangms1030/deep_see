# -*- coding: utf-8 -*-
"""온라인 conformal(CQR) 보정: 분위 과신을 목표 coverage로 정렬.

연구(final_eval.cqr)에서 검증된 적합도점수 E=max(lo-y, y-hi) 기반.
- conformal_delta: 검증표본으로 보정폭 Q 산출.
- calibrate_series: 각 origin에서 '이미 실현된 최근 K origin'만 사용(인과) → 밴드 조정.
  라벨은 horizon d일차가 origin+d*24h에 확정되므로 그 시점 이후에만 보정에 포함.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.system import config as C


def conformal_delta(cal_lo, cal_hi, cal_y, alpha: float) -> float:
    """검증 (예측구간, 실관측)으로 보정폭 Q 계산(분포무관, 유한표본 보정)."""
    E = np.maximum(np.asarray(cal_lo) - cal_y, np.asarray(cal_y) - np.asarray(cal_hi))
    E = E[np.isfinite(E)]
    if len(E) < C.CONFORMAL_MIN:
        return np.nan
    n = len(E)
    lv = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(E, lv, method="higher"))


def calibrate_series(origin_times, lo, hi, y, label_lag_h: int, alpha: float,
                     window: int = C.CONFORMAL_WINDOW, min_samples: int = C.CONFORMAL_MIN,
                     pooled_delta: float | None = None):
    """origin 시계열을 인과적으로 보정. 반환 (lo_cal, hi_cal, delta_used[per origin]).

    origin_times: 오름차순 Timestamp 배열. lo/hi: 예측 밴드 경계. y: 실관측(NaN 허용).
    label_lag_h: 해당 horizon 라벨 확정까지의 시차(h). pooled_delta: 표본부족 시 폴백폭.
    """
    ot = pd.to_datetime(np.asarray(origin_times))
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); y = np.asarray(y, float)
    n = len(lo)
    realize = ot + pd.Timedelta(hours=label_lag_h)
    E = np.maximum(lo - y, y - hi)             # NaN where y NaN
    lo_c, hi_c, used = lo.copy(), hi.copy(), np.full(n, np.nan)
    for i in range(n):
        avail = (realize.values <= ot.values[i]) & np.isfinite(E)
        idx = np.where(avail)[0]
        if len(idx) >= min_samples:
            Ei = E[idx[-window:]]
            m = len(Ei); lv = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
            Qv = float(np.quantile(Ei, lv, method="higher"))
        elif pooled_delta is not None and np.isfinite(pooled_delta):
            Qv = float(pooled_delta)
        else:
            Qv = 0.0                            # 표본 없음 → 원분위 유지
        used[i] = Qv
        lo_c[i] = lo[i] - Qv; hi_c[i] = hi[i] + Qv
    return lo_c, hi_c, used


def pooled_delta_from(cal_lo, cal_hi, cal_y, alpha: float) -> float:
    """풀링 폴백용 보정폭(여러 지점/origin을 합친 검증표본)."""
    return conformal_delta(cal_lo, cal_hi, cal_y, alpha)
