# -*- coding: utf-8 -*-
"""as-of 컨텍스트 빌더: 지점 시계열을 캐시하고, 특정 시점까지의 512h 버퍼를 만든다.

- to_chronos.station_series로 (times, raw, filled) 구성(경량보간 filled, 평가라벨은 raw).
- origin(=예보 기준 시점 인덱스)은 컨텍스트 [o-CTX, o), 라벨 [o, o+OUT_W).
- replay/serve 공용. 미래 미사용(인과) 보장은 호출부(origin 선택)에서 강제.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.system import config as C
from src.chronos import to_chronos as TC


class StationData:
    """단일 지점 시계열 컨테이너(캐시)."""

    def __init__(self, wide: pd.DataFrame, station: str):
        self.station = station
        self.times, self.raw, self.filled, _ = TC.station_series(wide, station)
        self.n = len(self.filled["do"])

    def origin_of(self, asof) -> int:
        """asof(timestamp) 이하 마지막 관측 시점 다음 인덱스 = 컨텍스트 끝(o)."""
        return int(np.searchsorted(self.times.values, np.datetime64(pd.Timestamp(asof)), side="right"))

    def has_context(self, o: int, ctx: int = C.CTX) -> bool:
        return o - ctx >= 0 and o <= self.n

    def test_origins(self, i_va: int, ctx: int = C.CTX, stride: int = C.STRIDE) -> list[int]:
        return TC.test_origins(self.n, i_va, ctx, C.OUT_W, stride)

    def val_origins(self, i_tr: int, i_va: int, ctx: int = C.CTX, stride: int = C.STRIDE) -> list[int]:
        """train_end~val_end 구간 origin(컨텍스트가 train_end 이후, 누수 없음)."""
        seg = i_va - i_tr
        width = ctx + C.OUT_W
        return [i_tr + i + ctx for i in range(0, seg - width + 1, stride)]

    def obs_daily(self, target: str, origins: list[int]) -> np.ndarray:
        """origins × 5일 일평균 관측(raw, 결측 NaN 유지)."""
        out = np.full((len(origins), C.HORIZON_DAYS), np.nan)
        rt = self.raw[target]
        for k, o in enumerate(origins):
            lab = rt[o:o + C.OUT_W][:C.OUT_W]
            if len(lab) < C.OUT_W:
                continue
            with np.errstate(invalid="ignore"):
                out[k] = np.nanmean(lab.reshape(C.HORIZON_DAYS, 24), axis=1)
        return out

    def persistence(self, target: str, origins: list[int]) -> np.ndarray:
        """직전 24h 평균을 5일 내내 평탄 유지(강한 베이스라인)."""
        rt = self.raw[target]
        return np.array([np.nanmean(rt[max(0, o - 24):o]) for o in origins])

    def climatology(self, target: str, origins: list[int], i_tr: int) -> np.ndarray:
        """train 구간 월별 기후값(없으면 전체평균)으로 5일차 예측."""
        rt = self.raw[target]
        tr_months = pd.DatetimeIndex(self.times.iloc[:i_tr].values).month
        clim = pd.Series(rt[:i_tr]).groupby(tr_months.to_numpy()).mean()
        gmean = np.nanmean(rt[:i_tr])
        months = pd.to_datetime(
            self.times.iloc[[min(o + 4 * 24, self.n - 1) for o in origins]].values).month
        return np.array([clim.get(int(m), gmean) for m in months])

    def origin_times(self, origins: list[int]) -> np.ndarray:
        return pd.to_datetime(self.times.iloc[[min(o, self.n - 1) for o in origins]].values)


def split_indices(times: pd.Series, split: dict) -> tuple[int, int]:
    """splits.json 항목 → (i_train_end, i_val_end) 인덱스."""
    t = times.values
    i_tr = int(np.searchsorted(t, np.datetime64(pd.Timestamp(split["train_end"]))))
    i_va = int(np.searchsorted(t, np.datetime64(pd.Timestamp(split["val_end"]))))
    return i_tr, i_va
