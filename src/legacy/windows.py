# -*- coding: utf-8 -*-
"""슬라이딩 윈도우 + 일평균 집계(레거시 compa 프로토콜 재현).

- 입력 240h(10일) → 출력 120h(5일). 평가 시 24h 평균으로 5개 일평균 산출.
- NSE 등은 5일차(predict_day index=4) 일평균 기준(레거시 표와 동일).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IN_W = 240
OUT_W = 120


def add_calendar(df: pd.DataFrame, time_col="time") -> pd.DataFrame:
    """레거시 file_open.py 와 동일한 Day/Year sin·cos 피처 추가."""
    ts = pd.to_datetime(df[time_col]).map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    out = df.copy()
    out["Day_sin"] = np.sin(ts * (2 * np.pi / day))
    out["Day_cos"] = np.cos(ts * (2 * np.pi / day))
    out["Year_sin"] = np.sin(ts * (2 * np.pi / year))
    out["Year_cos"] = np.cos(ts * (2 * np.pi / year))
    return out


def make_windows(feat: np.ndarray, target_idx: int, in_w=IN_W, out_w=OUT_W, stride=24,
                 label_source: np.ndarray | None = None):
    """feat: [T, F] (학습용=보간완료). label_source: 라벨 추출용 배열(없으면 feat).

    반환 inputs [N, in_w, F], labels [N, out_w], origin_idx [N] (라벨 시작 시점).
    """
    if label_source is None:
        label_source = feat
    T = feat.shape[0]
    width = in_w + out_w
    X, Y, origins = [], [], []
    for i in range(0, T - width + 1, stride):
        X.append(feat[i:i + in_w])
        Y.append(label_source[i + in_w:i + width, target_idx])
        origins.append(i + in_w)
    if not X:
        return (np.empty((0, in_w, feat.shape[1])), np.empty((0, out_w)), np.array([]))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(origins)


def hour_to_day_mean(arr: np.ndarray, observed: bool = False) -> np.ndarray:
    """[N, 120] → [N, 5] 일평균. observed=True면 NaN 무시(nanmean)."""
    n, w = arr.shape
    days = w // 24
    r = arr[:, :days * 24].reshape(n, days, 24)
    if observed:
        with np.errstate(invalid="ignore"):
            return np.nanmean(r, axis=2)
    return r.mean(axis=2)
