# -*- coding: utf-8 -*-
"""산출물 스키마(프론트엔드 계약): ForecastRecord / AlertRecord / Scorecard.

- 저장소(parquet)와 API(JSON)가 동일 필드명을 공유하도록 단일 출처로 고정.
- 프론트엔드 미구현이나 이후 시연 연동을 위해 안정적 키를 보장한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

from src.system import config as C

# 원분위 컬럼(7개) + conformal 보정 밴드 컬럼
RAW_Q_COLS = [f"q{q}" for q in C.Q]                       # q0.05 ... q0.95
CAL_COLS = ["q0.1_cal", "q0.9_cal", "q0.05_cal", "q0.95_cal"]   # 80%/90% 보정 밴드


def _f(x):
    """JSON 직렬화용 float 정규화(NaN→None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 6)


def forecast_row(station, river, asof, target, day, qvals, cal, median,
                 persistence, climatology, obs) -> dict:
    """예보 1행(=지점·origin·타깃·horizon). qvals: dict{level:val}, cal: dict{cal_col:val}."""
    row = {"station": station, "river": river, "asof": asof,
           "target": target, "horizon_day": int(day), "median": median,
           "persistence": persistence, "climatology": climatology, "obs": obs}
    for q in C.Q:
        row[f"q{q}"] = qvals.get(q, np.nan)
    for c in CAL_COLS:
        row[c] = cal.get(c, np.nan)
    return row


def alert_row(station, river, asof, target, day, p_exceed, level, threshold,
              direction, is_event, alpha, fired, lead_day, max_level,
              low_confidence) -> dict:
    """경보 1행(=지점·origin·타깃·horizon) + origin-타깃 요약 필드 동봉."""
    return {"station": station, "river": river, "asof": asof, "target": target,
            "horizon_day": int(day), "p_exceed": p_exceed, "level": level,
            "threshold": threshold, "direction": direction, "is_event": is_event,
            "alpha": alpha, "fired": bool(fired), "lead_day": lead_day,
            "max_level": max_level, "low_confidence": bool(low_confidence)}


@dataclass
class Scorecard:
    target: str
    n: int = 0
    nse: float = np.nan
    rsr: float = np.nan
    pbias: float = np.nan
    crps_skill: float = np.nan
    cov80_raw: float = np.nan
    cov80_cal: float = np.nan
    cov90_raw: float = np.nan
    cov90_cal: float = np.nan
    bss: float = np.nan
    pr_auc: float = np.nan
    f1: float = np.nan
    lead_day: float = np.nan
    vs_persistence: float = np.nan     # ΔNSE(model - persistence), 가드레일 핵심
    rating: str = "—"
    slo_pass: bool = False

    def as_dict(self):
        return asdict(self)


def forecast_json(station, river, name, asof, per_target: dict) -> dict:
    """API 응답(프론트 계약). per_target[tg] = dict(mode, threshold, unit, direction,
    alpha, low_confidence, lead_day, max_level, baseline_persistence, horizons[list])."""
    out = {"station": station, "river": river, "name": name, "asof": asof, "targets": {}}
    for tg, d in per_target.items():
        hz = [{"day": int(h["day"]), "median": _f(h["median"]),
               "q10": _f(h["q0.1"]), "q90": _f(h["q0.9"]),
               "q10_cal": _f(h["q0.1_cal"]), "q90_cal": _f(h["q0.9_cal"]),
               "q05_cal": _f(h["q0.05_cal"]), "q95_cal": _f(h["q0.95_cal"]),
               "p_exceed": _f(h["p_exceed"]), "level": h["level"]}
              for h in d["horizons"]]
        out["targets"][tg] = {
            "mode": C.TARGET_MODE.get(tg), "threshold": _f(d["threshold"]),
            "unit": d.get("unit"), "direction": d.get("direction"),
            "alpha": _f(d.get("alpha")), "low_confidence": bool(d.get("low_confidence", False)),
            "lead_day": (int(d["lead_day"]) if d.get("lead_day") not in (None, np.nan)
                         and np.isfinite(d.get("lead_day", np.nan)) else None),
            "max_level": d.get("max_level"),
            "baseline_persistence": _f(d.get("baseline_persistence")),
            "horizons": hz,
        }
    return out
