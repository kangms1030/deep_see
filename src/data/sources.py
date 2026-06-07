# -*- coding: utf-8 -*-
"""AIHub '수질측정 및 오염원' — 자동측정망(시간단위) 로더/스키마.

데이터 실체(EDA 확인):
- 자동측정망 CSV는 **헤더 없음**(컬럼 순서 = 동일 폴더 json 키 순서, 15개), **cp949**.
- LONG 포맷: 한 행 = (측정소, 측정_일시, 항목) 하나의 값.
- 일부 CSV 첫 행에 한글 헤더가 섞여 있음 → 측정_일시가 14자리(YYYYMMDDHHMMSS)가 아니면 제거.
- 측정소_아이디(예: S01001) 영문자 포함 → 반드시 dtype=str.
- 파일은 **시간 분할**(파일 1개 = 특정 기간, 전 지점 포함) → 전 파일을 합쳐 지점별 시계열 구성.
- 수계 키 = 측정소_아이디 접두(S01=한강, S02=낙동강, S03=금강, S04=영산강/섬진강).
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # .../deep_see/src/data
DEEP_SEE = os.path.dirname(os.path.dirname(HERE))            # .../deep_see
DATA_ROOT = os.path.dirname(DEEP_SEE)                        # .../수질측정 및 오염원

# 15개 컬럼(원본 json 키 순서; 공백 제거 단순화한 이름 사용)
AUTO_COLS = ["측정_일시", "하천_아이디", "측정소_아이디", "측정소명", "수질정보_진행_플래그",
             "측정_일시_재적재", "체크_오류", "항목_코드", "항목_명", "항목_값",
             "항목정제여부", "위도", "경도", "CAT_ID", "CAT_DID"]

AUTO_REL = os.path.join("1. 물환경측정망", "13. 자동측정망", "csv")

# 캐논 채널 -> 항목_코드 변형 목록(센서 1/2/3 등). 변형은 지점별 관측수 많은 순으로 coalesce.
CHANNELS = {
    "do":    ["M72", "M41", "M05"],   # 용존산소 3/2/1
    "toc":   ["M06"],                 # 총유기탄소
    "tn":    ["M27"],                 # 총질소
    "tp":    ["M28"],                 # 총인
    "chl-a": ["M29"],                 # 클로로필-a
    "temp":  ["M69", "M38", "M02"],   # 수온 3/2/1
    "ph":    ["M70", "M39", "M03"],   # 수소이온농도 3/2/1
    "ec":    ["M71", "M40", "M04"],   # 전기전도도 3/2/1
    "turb":  ["M73"],                 # 탁도
    "no3":   ["M37"],                 # 질산성질소
}
TARGETS = ["do", "toc", "tn", "tp", "chl-a"]               # 레거시 5타깃
AUX = ["temp", "ph", "ec", "turb", "no3"]                  # 다변량 보조 수질 채널
CHANNEL_ORDER = TARGETS + AUX

WANTED_CODES = [c for codes in CHANNELS.values() for c in codes]
CODE_TO_CANON = {c: canon for canon, codes in CHANNELS.items() for c in codes}

RIVER_BY_PREFIX = {"S01": "han", "S02": "nak", "S03": "geum", "S04": "yeong"}
RIVER_KR = {"han": "한강", "nak": "낙동강", "geum": "금강", "yeong": "영산강"}

_DATE_RE = re.compile(r"^(19|20)\d{12}$")


def auto_csv_paths(splits=("Training", "Validation")) -> list[str]:
    paths = []
    for sp in splits:
        d = os.path.join(DATA_ROOT, sp, f"[라벨]{sp}", AUTO_REL)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".csv"):
                    paths.append(os.path.join(d, fn))
    return paths


def load_auto_csv(path: str) -> pd.DataFrame:
    """단일 CSV → 정제된 long DF [time, station, river, code, canon, value, lat, lon, cat_id, name].

    원하는 항목코드만 남기고, 헤더혼입/깨진 날짜 제거, 값 수치화.
    """
    df = pd.read_csv(path, header=None, names=AUTO_COLS, dtype=str,
                     encoding="cp949", on_bad_lines="skip", engine="python")
    t = df["측정_일시"].astype("string").str.strip()
    df = df[t.str.match(_DATE_RE).fillna(False)].copy()
    df = df[df["항목_코드"].isin(WANTED_CODES)]
    if df.empty:
        return df.iloc[0:0]
    df["time"] = pd.to_datetime(df["측정_일시"], format="%Y%m%d%H%M%S", errors="coerce").dt.floor("h")
    df = df[df["time"].notna()]
    df["value"] = pd.to_numeric(df["항목_값"], errors="coerce")
    df["station"] = df["측정소_아이디"].str.strip()
    df["river"] = df["station"].str[:3].map(RIVER_BY_PREFIX)
    df = df[df["river"].notna()]
    df["code"] = df["항목_코드"]
    df["canon"] = df["code"].map(CODE_TO_CANON)
    df["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    df["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    df["name"] = df["측정소명"].str.strip()
    df["cat_id"] = df["CAT_ID"].str.strip()
    return df[["time", "station", "river", "code", "canon", "value",
               "lat", "lon", "cat_id", "name"]]


def coalesce_channels(piv: pd.DataFrame) -> pd.DataFrame:
    """항목_코드별 wide(piv: index=time, columns=code) → 캐논 채널 wide.

    변형 코드는 지점 내 관측수 많은 순으로 row-wise coalesce.
    """
    out = pd.DataFrame(index=piv.index)
    for canon in CHANNEL_ORDER:
        present = [c for c in CHANNELS[canon] if c in piv.columns]
        if not present:
            out[canon] = np.nan
            continue
        present.sort(key=lambda c: piv[c].notna().sum(), reverse=True)
        out[canon] = piv[present].bfill(axis=1).iloc[:, 0].astype("float64")
    return out.astype("float32")
