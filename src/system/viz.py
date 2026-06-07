# -*- coding: utf-8 -*-
"""경보 타임라인 시각화(시연 자산, 프론트 시안). system_out 저장소를 읽어 PNG 생성.

상단: 관측·중앙값·80/90% 보정밴드·임계선. 하단: P(임계 초과)·발령마커.
src/alert/alert.plot_timeline 스타일을 시스템 저장소 포맷(long)으로 일반화.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

from src.system import config as C
from src.alert import thresholds as TH
from src.utils.progress import log

TASK = "sys_viz"


def plot_timeline(river, station, target, day=5):
    fp = os.path.join(C.FORECASTS, f"{river}_{station}.parquet")
    if not os.path.exists(fp):
        return None
    fc = pd.read_parquet(fp)
    g = fc[(fc.target == target) & (fc.horizon_day == day)].copy()
    if g.empty:
        return None
    g["t"] = pd.to_datetime(g["asof"]); g = g.sort_values("t")
    al_path = os.path.join(C.ALERTS, "alert_log.parquet")
    p = None; alpha = C.ALPHA_DEFAULT
    if os.path.exists(al_path):
        al = pd.read_parquet(al_path)
        al = al[(al.station == station) & (al.target == target) & (al.horizon_day == day)].copy()
        if len(al):
            al["t"] = pd.to_datetime(al["asof"])
            p = g.merge(al[["t", "p_exceed", "alpha"]], on="t", how="left")["p_exceed"].to_numpy()
            alpha = float(al["alpha"].iloc[0])
    th = TH.THRESHOLDS[target]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.fill_between(g["t"], g["q0.05_cal"], g["q0.95_cal"], alpha=0.15, label="90% PI(보정)")
    ax1.fill_between(g["t"], g["q0.1_cal"], g["q0.9_cal"], alpha=0.25, label="80% PI(보정)")
    ax1.plot(g["t"], g["median"], lw=1, label="중앙값(예측)")
    ax1.scatter(g["t"], g["obs"], s=8, c="k", label="관측", zorder=5)
    ax1.axhline(th["value"], color="r", ls="--", label=f"임계 {th['value']}{th['unit']}")
    ax1.set_title(f"[system] {river}/{station}/{target} {day}일차 예보 — {th['desc']}")
    ax1.legend(loc="upper right", fontsize=8); ax1.set_ylabel(target)

    if p is not None:
        ax2.plot(g["t"], p, color="darkorange", lw=1, label="P(임계 초과)")
        fired = np.isfinite(p) & (p >= alpha)
        ax2.scatter(g["t"][fired], p[fired], c="red", s=12, label=f"경보(α={alpha:.2f})", zorder=5)
        ax2.axhline(alpha, color="gray", ls=":")
    ax2.set_ylim(0, 1); ax2.set_ylabel("경보확률"); ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(C.FIGS, f"system_timeline_{river}_{station}_{target}.png")
    plt.savefig(out, dpi=120); plt.close()
    log(f"타임라인 저장 {out}", TASK)
    return out


def plot_representatives(targets=None):
    sidx = C.load_station_index()
    outs = []
    for r in sidx[sidx.is_representative].itertuples():
        for tg in (targets or C.TARGETS):
            o = plot_timeline(r.river, str(r.station), tg)
            if o:
                outs.append(o)
    return outs
