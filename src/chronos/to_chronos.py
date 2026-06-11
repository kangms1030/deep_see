# -*- coding: utf-8 -*-
"""Phase 1/6 데이터 → Chronos-2 입력(dict) 변환 (+수문/기상 공변량 선택적 결합).

Chronos2Pipeline.predict/fit 의 inputs = list[dict]:
  {"target": 1D array, "past_covariates": {name: 1D array}}
- 기본: target=해당 채널, past_covariates=나머지 수질 채널.
- use-cov: past_covariates에 수문 공변량(유량/수위/댐 방류·유입·저수위) 추가(과거값만, 미래미지).
- use-weather: past_covariates에 기상 공변량(기온/강수/풍속/습도/일사) 추가.
- 컨텍스트 결측은 경량 선형보간+ffill/bfill. 평가 라벨은 항상 raw 관측(NaN 유지).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data import sources as S

COV_CHANNELS = ["cov_flow", "cov_level", "cov_dam_discharge", "cov_dam_inflow", "cov_dam_level"]
WEATHER_CHANNELS = ["cov_air_temp", "cov_rainfall", "cov_wind_speed", "cov_humidity", "cov_solar_rad"]


def light_fill(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a, dtype="float64").interpolate(limit_direction="both")
    return s.ffill().bfill().fillna(0.0).to_numpy()


def station_series(wide: pd.DataFrame, station: str, cov_wide: pd.DataFrame | None = None,
                   weather_wide: pd.DataFrame | None = None):
    """(times, raw[wq], filled[wq], cov[filled]) 반환. cov_wide/weather_wide 주면 공변량 결합."""
    sdf = wide[wide["station"] == station].sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(sdf["time"])
    raw = {c: sdf[c].to_numpy(np.float64) for c in S.CHANNEL_ORDER}
    filled = {c: light_fill(raw[c]) for c in S.CHANNEL_ORDER}
    cov = {}
    # 수문 공변량
    if cov_wide is not None:
        cdf = cov_wide[cov_wide["station"] == station].sort_values("time")
        cdf = cdf.set_index("time").reindex(times.values)
        for c in COV_CHANNELS:
            if c in cdf.columns:
                arr = cdf[c].to_numpy(np.float64)
                if np.isfinite(arr).any():           # 전부 결측인 공변량(예: 댐 없음)은 제외
                    cov[c] = light_fill(arr)
    # 기상 공변량
    if weather_wide is not None:
        wdf = weather_wide[weather_wide["station"] == station].sort_values("time")
        wdf = wdf.set_index("time").reindex(times.values)
        for c in WEATHER_CHANNELS:
            if c in wdf.columns:
                arr = wdf[c].to_numpy(np.float64)
                if np.isfinite(arr).any():
                    cov[c] = light_fill(arr)
    return times, raw, filled, cov


def test_origins(n: int, i_va: int, in_w: int, out_w: int, stride: int = 24):
    """레거시 windows_in과 동일한 롤링 origin(컨텍스트 전부 test 구간 내, 누수 없음)."""
    width = in_w + out_w
    seg = n - i_va
    return [i_va + i + in_w for i in range(0, seg - width + 1, stride)]


def build_predict_inputs(filled, target, origins, context_length, cov=None):
    covs = [c for c in S.CHANNEL_ORDER if c != target]
    inputs = []
    for o in origins:
        lo = o - context_length
        pc = {c: filled[c][lo:o].astype(np.float32) for c in covs}
        if cov:
            for c in cov:
                pc[c] = cov[c][lo:o].astype(np.float32)
        inputs.append({"target": filled[target][lo:o].astype(np.float32), "past_covariates": pc})
    return inputs


def build_finetune_inputs(wide, stations, target, splits, cov_wide=None,
                          weather_wide=None, min_len=600):
    """LoRA 학습용: 각 지점 train 구간 전체를 1 item(+공변량)."""
    covs = [c for c in S.CHANNEL_ORDER if c != target]
    items, val_items = [], []
    times_all = {}
    for st in stations:
        _, raw, filled, cov = station_series(wide, st, cov_wide, weather_wide=weather_wide)
        t = pd.to_datetime(wide[wide.station == st].sort_values("time")["time"]).values
        i_va = int(np.searchsorted(t, np.datetime64(pd.Timestamp(splits[st]["train_end"]))))
        i_ve = int(np.searchsorted(t, np.datetime64(pd.Timestamp(splits[st]["val_end"]))))
        if i_va < min_len:
            continue

        def mk(hi):
            pc = {c: filled[c][:hi].astype(np.float32) for c in covs}
            for c in cov:
                pc[c] = cov[c][:hi].astype(np.float32)
            return {"target": filled[target][:hi].astype(np.float32), "past_covariates": pc}

        items.append(mk(i_va))
        if i_ve > min_len:
            val_items.append(mk(i_ve))
    return items, val_items
