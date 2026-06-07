# -*- coding: utf-8 -*-
"""Phase 6: 수문(유량/수위/댐) 공변량을 자동측정망 지점에 최근접 링크 → 시간단위 결합.

직접 조사 결과(2026-06-06):
- 유량/수위/댐 = 시간단위·헤더없음·cp949, 위도/경도+CAT_ID 보유, 2009/2010/2003~ → 2012~2019 전구간 커버.
- 자동측정망 67지점 기준 최근접 유량 median 2.2km(전지점<20km), 수위 1.8km(전지점<10km), 댐 median 28.7km(근거리만).
- ASOS/AWS는 2017~만·좌표 미포함(외부표 필요) → 1차 범위 제외(별도 단계).
- 우량(54)은 표본 희소/불규칙 → 제외(유량이 유출 대리).

링크: 각 WQ지점 ↔ 최근접 유량/수위 관측소(haversine), 댐은 <20km만. 시간격자 정렬해 결합.
산출: data_processed/{river}_covariates_hourly.parquet [time, station, cov_flow, cov_level,
       cov_dam_discharge, cov_dam_inflow, cov_dam_level]
실행: PYTHONIOENCODING=utf-8 python -m src.data.build_covariates
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.data import sources as S
from src.utils.progress import track, log

HYDRO = os.path.join(S.DATA_ROOT, "Training", "[라벨]Training", "5. 수리수문기상")
OUT = os.path.join(S.DEEP_SEE, "data_processed")
TASK = "build_covariates"

# 폴더별 (코드열, 시간열, 위도열, 경도열, {출력채널: 값열})
SPEC = {
    "flow":  {"dir": "53. 수리수문기상(유량)", "code": 1, "time": 2, "lat": 6, "lon": 7,
              "vals": {"cov_flow": 5}},
    "level": {"dir": "55. 수리수문기상(수위)", "code": 1, "time": 0, "lat": 6, "lon": 7,
              "vals": {"cov_level": 5}},
    "dam":   {"dir": "51. 수리수문기상(댐)", "code": 2, "time": 0, "lat": 26, "lon": 27,
              "vals": {"cov_dam_discharge": 13, "cov_dam_inflow": 9, "cov_dam_level": 5},
              "max_km": 20.0},
}


def _haversine(la1, lo1, la2, lo2):
    R = 6371.0; p = np.pi / 180
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _files(folder):
    d = os.path.join(HYDRO, folder, "csv")
    return d, sorted(os.listdir(d))


def station_meta(spec):
    """표본 파일에서 관측소 코드→lat/lon 메타(전 지점 거의 포함)."""
    d, files = _files(spec["dir"])
    idxs = sorted(set([0, len(files) // 4, len(files) // 2, 3 * len(files) // 4, len(files) - 1]))
    rows = []
    for i in idxs:
        df = pd.read_csv(os.path.join(d, files[i]), header=None, dtype=str, encoding="cp949",
                         on_bad_lines="skip", usecols=[spec["code"], spec["lat"], spec["lon"]])
        df.columns = ["code", "lat", "lon"]
        rows.append(df.drop_duplicates("code"))
    m = pd.concat(rows).drop_duplicates("code")
    m["lat"] = pd.to_numeric(m["lat"], errors="coerce")
    m["lon"] = pd.to_numeric(m["lon"], errors="coerce")
    return m.dropna().reset_index(drop=True)


def nearest_map(wq, meta, max_km=None):
    """WQ지점 → 최근접 hydro 코드 (max_km 초과 시 None)."""
    out = {}
    for _, w in wq.iterrows():
        dkm = _haversine(float(w["lat"]), float(w["lon"]), meta["lat"].values, meta["lon"].values)
        j = int(np.argmin(dkm))
        if max_km is None or dkm[j] <= max_km:
            out[w["station"]] = (meta.iloc[j]["code"], float(dkm[j]))
    return out


def load_series(spec, needed_codes):
    """전 파일 1패스 스캔 → 필요한 코드만 long [time, code, <val채널들>]."""
    d, files = _files(spec["dir"])
    usecols = [spec["code"], spec["time"]] + list(spec["vals"].values())
    names = ["code", "time"] + list(spec["vals"].keys())
    needed = set(needed_codes)
    chunks = []
    for fn in track(files, desc=f"{spec['dir'][:14]} 로드", task=TASK):
        df = pd.read_csv(os.path.join(d, fn), header=None, dtype=str, encoding="cp949",
                         on_bad_lines="skip", usecols=usecols)
        df.columns = [c for _, c in sorted(zip(usecols, names))]  # usecols 정렬 보정
        df = df[["code", "time"] + list(spec["vals"].keys())]
        df["code"] = df["code"].str.strip()
        df = df[df["code"].isin(needed)]
        if df.empty:
            continue
        t = df["time"].astype(str).str.strip()
        df = df[t.str.match(r"^\d{10,14}$").fillna(False)]
        if df.empty:
            continue
        df["time"] = pd.to_datetime(df["time"].str[:10], format="%Y%m%d%H", errors="coerce").dt.floor("h")
        df = df[df["time"].notna()]
        for c in spec["vals"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        chunks.append(df)
    if not chunks:
        return pd.DataFrame(columns=["code", "time"] + list(spec["vals"].keys()))
    long = pd.concat(chunks, ignore_index=True)
    # 동일 (code,time) 평균
    return long.groupby(["code", "time"], as_index=False).mean()


def main():
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv"))
    sidx["station"] = sidx["station"].astype(str)

    # 1) 메타 + 최근접 매핑
    nmap = {}; meta_cache = {}
    for key, spec in SPEC.items():
        meta_cache[key] = station_meta(spec)
        nmap[key] = nearest_map(sidx, meta_cache[key], spec.get("max_km"))
        log(f"{key}: 관측소 {len(meta_cache[key])}개, 링크된 WQ지점 {len(nmap[key])}/{len(sidx)}", TASK)

    # 2) 필요한 코드 series 로드
    series = {}  # key -> {code -> wide df(index time, cols val채널)}
    for key, spec in SPEC.items():
        needed = sorted({c for c, _ in nmap[key].values()})
        log(f"{key}: 필요 코드 {len(needed)}개 series 로드 시작", TASK)
        long = load_series(spec, needed)
        series[key] = {}
        for code, g in long.groupby("code"):
            series[key][code] = g.set_index("time")[list(spec["vals"].keys())].sort_index()

    # 3) 수계별 WQ 시간격자에 결합
    for river in ["han", "nak", "geum", "yeong"]:
        fp = os.path.join(OUT, f"{river}_auto_hourly_wide.parquet")
        if not os.path.exists(fp):
            continue
        wide = pd.read_parquet(fp, columns=["time", "station"])
        wide["station"] = wide["station"].astype(str)
        outs = []
        for st, g in track(wide.groupby("station"), desc=f"{river} 결합", task=TASK):
            grid = pd.to_datetime(g["time"]).sort_values()
            base = pd.DataFrame({"time": grid.values, "station": st})
            base = base.set_index("time")
            for key, spec in SPEC.items():
                cols = list(spec["vals"].keys())
                code = nmap[key].get(st, (None, None))[0]
                if code is not None and code in series[key]:
                    s = series[key][code].reindex(base.index)
                    for c in cols:
                        base[c] = s[c].values
                else:
                    for c in cols:
                        base[c] = np.nan
            outs.append(base.reset_index())
        cov = pd.concat(outs, ignore_index=True)
        cov["station"] = cov["station"].astype("category")
        outfp = os.path.join(OUT, f"{river}_covariates_hourly.parquet")
        cov.to_parquet(outfp, index=False)
        covcols = [c for c in cov.columns if c.startswith("cov_")]
        covstat = {c: round(cov[c].notna().mean(), 3) for c in covcols}
        log(f"[{river}] 저장 {outfp} shape={cov.shape} 커버리지={covstat}", TASK)

    # 링크 거리 요약 저장
    link_rows = []
    for st in sidx["station"]:
        row = {"station": st}
        for key in SPEC:
            code, dkm = nmap[key].get(st, (None, None))
            row[f"{key}_code"] = code; row[f"{key}_km"] = round(dkm, 2) if dkm else None
        link_rows.append(row)
    pd.DataFrame(link_rows).to_csv(os.path.join(OUT, "covariate_links.csv"),
                                   index=False, encoding="utf-8-sig")
    log("covariate_links.csv 저장 완료", TASK)


if __name__ == "__main__":
    main()
