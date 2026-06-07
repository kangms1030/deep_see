# -*- coding: utf-8 -*-
"""시스템 중앙 설정: 임계·분위·보정창·경보등급·비용가중·SLO·경로·지점목록.

모든 스테이지(forecast/conformal/alerting/scorecard/replay/serve)가 본 모듈을 단일
출처로 참조한다. 임계는 기존 src/alert/thresholds.THRESHOLDS를 그대로 재사용.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.data import sources as S
from src.alert import thresholds as TH

# ---- 예보 구성(연구 결론 반영) ----
CTX = 512                       # 컨텍스트 길이(하이퍼파라미터, 성능 레버)
OUT_W = 120                     # 예측 horizon(시간) = 5일
HORIZON_DAYS = 5
STRIDE = 24                     # 롤링 origin 간격(시간)
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
TARGETS = S.TARGETS
RIVERS = ["han", "nak", "geum", "yeong"]

# ---- conformal 온라인 보정 ----
CONFORMAL_WINDOW = 90           # 최근 실현 origin 표본 수(롤링창)
CONFORMAL_MIN = 10              # 미만이면 풀링 폴백(없으면 원분위)

# ---- 경보 정책 ----
# 확률 등급 밴드(내림차순: 높은 등급 먼저 매칭)
ALERT_LEVELS = [("경계", 0.7), ("주의", 0.5), ("관심", 0.3)]
ALPHA_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2)   # 발령 임계 α 탐색 격자
ALPHA_DEFAULT = 0.5
# 비용비 c_FN:c_FP (미탐지가 오경보보다 비싸다). 타깃별 오버라이드.
COST_FN_FP = {"_default": 5.0, "do": 5.0, "tn": 4.0, "toc": 3.0, "tp": 3.0, "chl-a": 6.0}

# ---- 타깃 운영 모드(절대 적합성 판정 반영) ----
TARGET_MODE = {
    "do":    "point+alert",     # 점추정 신뢰 + 경보
    "tn":    "point+alert",
    "toc":   "point(보수)+alert",
    "tp":    "alert(보조)",      # 점추정 약함 → 경보 보조
    "chl-a": "alert(녹조중심)",  # 점추정 부적합, 경보 유용
}

# ---- SLO(서비스 목표; 모니터링 비교용) ----
SLO = {
    "do":    {"nse": 0.70, "cov80": (0.76, 0.84), "pr_auc": 0.70},
    "tn":    {"nse": 0.60, "cov80": (0.76, 0.84), "pr_auc": 0.70},
    "toc":   {"nse": 0.50, "cov80": (0.76, 0.84), "pr_auc": 0.60},
    "tp":    {"nse": 0.40, "cov80": (0.76, 0.84), "pr_auc": 0.50},
    "chl-a": {"nse": 0.25, "cov80": (0.76, 0.84), "pr_auc": 0.70},
}

# ---- 경로 ----
DEEP_SEE = S.DEEP_SEE
DATA_PROC = os.path.join(DEEP_SEE, "data_processed")
MODELS = os.path.join(DEEP_SEE, "models")
SYS_OUT = os.path.join(DEEP_SEE, "system_out")
FORECASTS = os.path.join(SYS_OUT, "forecasts")
ALERTS = os.path.join(SYS_OUT, "alerts")
SCORECARD = os.path.join(SYS_OUT, "scorecard")
API_SAMPLES = os.path.join(SYS_OUT, "api_samples")
FIGS = os.path.join(SYS_OUT, "figures")
REGISTRY_PATH = os.path.join(SYS_OUT, "registry.json")
LOGS = os.path.join(DEEP_SEE, "logs")


def ensure_dirs():
    for d in (SYS_OUT, FORECASTS, ALERTS, SCORECARD, API_SAMPLES, FIGS, LOGS):
        os.makedirs(d, exist_ok=True)


def cost_ratio(target: str) -> float:
    return COST_FN_FP.get(target, COST_FN_FP["_default"])


def alert_level(p: float) -> str | None:
    """확률 p → 경보 등급(없으면 None)."""
    if not np.isfinite(p):
        return None
    for name, thr in ALERT_LEVELS:
        if p >= thr:
            return name
    return None


def load_station_index() -> pd.DataFrame:
    sidx = pd.read_csv(os.path.join(DATA_PROC, "station_index.csv"))
    sidx["station"] = sidx["station"].astype(str)
    return sidx


def station_list(sidx: pd.DataFrame, scope: str = "all") -> list[tuple[str, str]]:
    """(river, station) 목록. scope='all'(67) | 'rep'(대표 4)."""
    df = sidx[sidx.is_representative] if scope == "rep" else sidx
    return list(zip(df["river"].astype(str), df["station"].astype(str)))


def threshold(target: str, obs_series=None, mode: str = "standard") -> dict:
    return TH.get_threshold(target, obs_series, mode)
