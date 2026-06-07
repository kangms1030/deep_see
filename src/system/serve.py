# -*- coding: utf-8 -*-
"""호출형 예보 API: forecast_asof(station, asof) → 프론트 소비용 JSON.

- 기본(store): replay 저장소(system_out/forecasts + alert_log)에서 해당 시점 예보·경보 조회.
- live=True: 어댑터로 그 시점 즉시 추론(+저장소 잔차로 pooled conformal 보정) → 동일 스키마.
웹서버는 없음. schemas.forecast_json으로 키를 고정해 추후 프론트 연동을 보장.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from src.system import config as C
from src.system import schemas as SC
from src.alert import thresholds as TH


def _river_name(station):
    sidx = C.load_station_index()
    row = sidx[sidx.station == str(station)]
    if row.empty:
        raise ValueError(f"알 수 없는 지점: {station}")
    nm = row["name"].iloc[0] if "name" in row.columns else station
    return row["river"].iloc[0], nm


def _pick_asof(df, asof):
    times = pd.to_datetime(df["asof"]).sort_values().unique()
    if len(times) == 0:
        return None
    if asof is None:
        return pd.Timestamp(times[-1]).isoformat()
    t = pd.Timestamp(asof)
    le = times[times <= np.datetime64(t)]
    return pd.Timestamp((le[-1] if len(le) else times[0])).isoformat()


def forecast_asof(station: str, asof=None, targets=None) -> dict:
    """저장소 기반 예보·경보 조회 → JSON(dict)."""
    station = str(station)
    river, name = _river_name(station)
    fp = os.path.join(C.FORECASTS, f"{river}_{station}.parquet")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"예보 저장소 없음: {fp} (먼저 replay 실행)")
    fc = pd.read_parquet(fp)
    al_path = os.path.join(C.ALERTS, "alert_log.parquet")
    al = pd.read_parquet(al_path) if os.path.exists(al_path) else pd.DataFrame()
    if len(al):
        al = al[al.station == station]
    asof_sel = _pick_asof(fc, asof)
    cur = fc[fc["asof"] == asof_sel]
    tgs = targets or [t for t in C.TARGETS if t in set(cur["target"])]

    per_target = {}
    for tg in tgs:
        g = cur[cur.target == tg].sort_values("horizon_day")
        if g.empty:
            continue
        ag = al[(al["target"] == tg) & (al["asof"] == asof_sel)] if len(al) else pd.DataFrame()
        pmap = dict(zip(ag["horizon_day"], ag["p_exceed"])) if len(ag) else {}
        th = TH.THRESHOLDS[tg]
        horizons = []
        for _, r in g.iterrows():
            day = int(r["horizon_day"]); pe = pmap.get(day, np.nan)
            horizons.append({"day": day, "median": r["median"],
                             "q0.1": r["q0.1"], "q0.9": r["q0.9"],
                             "q0.1_cal": r["q0.1_cal"], "q0.9_cal": r["q0.9_cal"],
                             "q0.05_cal": r["q0.05_cal"], "q0.95_cal": r["q0.95_cal"],
                             "p_exceed": pe, "level": C.alert_level(pe)})
        meta = ag.iloc[0] if len(ag) else None
        mx = meta["max_level"] if meta is not None else None
        ld = meta["lead_day"] if meta is not None else None
        per_target[tg] = {
            "threshold": float(meta["threshold"]) if meta is not None else th["value"],
            "unit": th["unit"], "direction": th["direction"],
            "alpha": float(meta["alpha"]) if meta is not None else None,
            "low_confidence": bool(meta["low_confidence"]) if meta is not None else False,
            "lead_day": (float(ld) if pd.notna(ld) else None),
            "max_level": (mx if isinstance(mx, str) else None),
            "baseline_persistence": float(g["persistence"].iloc[0]),
            "horizons": horizons}
    return SC.forecast_json(station, river, name, asof_sel, per_target)


def dump_samples(stations=None, asof=None, n_default=4) -> list[str]:
    """프론트 계약 데모용 JSON 샘플 저장."""
    C.ensure_dirs()
    if stations is None:
        sidx = C.load_station_index()
        stations = sidx[sidx.is_representative]["station"].astype(str).tolist()[:n_default]
    out = []
    for st in stations:
        try:
            js = forecast_asof(st, asof)
        except FileNotFoundError:
            continue
        path = os.path.join(C.API_SAMPLES, f"forecast_{js['river']}_{st}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)
        out.append(path)
    return out
