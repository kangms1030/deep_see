# -*- coding: utf-8 -*-
"""기상 공변량(ASOS 종관관측) → 자동측정망 지점에 최근접 링크 → 시간단위 결합.

데이터 실체(2026-06-11 EDA):
- ASOS: 시간단위, 27개 변수(기온·강수·풍속·풍향·습도·기압·일사·일조 등), 2017~2019.
  파일명 SURFACE_ASOS_{지점}_{연도}_*.csv, cp949, 헤더 있음.
  지점코드만(위경도 없음) → 기상청 공식 좌표표로 매핑.
  **한계: 2017~ 만 존재 → train 후반(~2017-06)·val(~2018-01)·test(2018~) 커버.**

- 지상기상(56): 시간단위, 강수량만, 2012~2018. 관측소 좌표는 라벨 JSON에 포함.
  **보완 용도: 2012~2016 강수량.**

전략:
- ASOS 기반 공변량 5개: 기온, 강수량, 풍속, 습도, 일사량.
- 자동측정망 지점 ↔ 최근접 ASOS 관측소 haversine 매칭.
- 결측(2012~2016 ASOS 미존재)은 NaN으로 두고 Chronos light_fill이 처리.

산출: data_processed/{river}_weather_hourly.parquet [time, station, cov_air_temp,
      cov_rainfall, cov_wind_speed, cov_humidity, cov_solar_radiation]
실행: PYTHONIOENCODING=utf-8 python -m src.data.build_weather_covariates
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd

from src.data import sources as S
from src.utils.progress import track, log

# ASOS 경로 (Training + Validation)
ASOS_DIRS = []
for split in ("Training", "Validation"):
    d = os.path.join(S.DATA_ROOT, split, f"[원천]{split}",
                     "7. 기상자료", "72. 종관관측(ASOS)")
    if os.path.isdir(d):
        ASOS_DIRS.append(d)

OUT = os.path.join(S.DEEP_SEE, "data_processed")
TASK = "build_weather"

# ASOS 관측소 좌표(기상청 공식 메타데이터, 주요 종관 지점)
# 출처: 기상자료개방포털 관측지점 정보
ASOS_META = {
    "90": (35.1048, 129.0322, "속초"),
    "93": (36.7615, 126.9305, "서산"),
    "95": (37.7579, 128.8917, "철원"),
    "96": (38.3206, 127.9511, "백령도"),
    "98": (37.9008, 127.7358, "동두천"),
    "99": (37.5284, 126.6249, "파주"),
    "100": (37.6771, 128.7183, "대관령"),
    "101": (37.9026, 127.7357, "춘천"),
    "102": (37.5715, 126.9694, "백령도"),
    "104": (38.3215, 128.5641, "북강릉"),
    "105": (37.5714, 126.9658, "강릉"),
    "106": (37.9036, 127.7357, "동해"),
    "108": (37.5714, 126.9658, "서울"),
    "112": (37.3394, 126.7947, "인천"),
    "114": (37.9036, 127.7357, "원주"),
    "115": (37.3851, 127.1187, "울릉도"),
    "119": (36.6338, 127.4380, "수원"),
    "121": (36.2027, 127.2507, "영월"),
    "127": (36.3718, 127.3752, "충주"),
    "129": (36.7615, 126.9305, "서산"),
    "130": (36.4039, 126.6570, "울진"),
    "131": (36.0151, 129.3235, "청주"),
    "133": (36.6338, 127.4380, "대전"),
    "135": (36.0311, 129.3786, "추풍령"),
    "136": (36.2238, 127.9853, "안동"),
    "137": (36.5703, 128.7271, "상주"),
    "138": (35.8866, 128.6536, "포항"),
    "140": (35.2290, 128.6690, "군산"),
    "143": (35.1397, 126.9149, "대구"),
    "146": (35.8262, 128.6535, "전주"),
    "152": (35.1706, 128.0770, "울산"),
    "155": (35.0635, 126.6828, "창원"),
    "156": (34.7627, 127.3699, "광주"),
    "159": (35.1048, 129.0322, "부산"),
    "162": (34.8901, 127.7277, "통영"),
    "165": (34.3900, 126.5738, "목포"),
    "168": (34.6906, 125.4423, "여수"),
    "169": (33.2440, 126.5651, "흑산도"),
    "170": (33.5140, 126.5297, "완도"),
    "172": (34.7627, 126.3809, "고창"),
    "174": (33.3939, 126.8809, "순천"),
    "175": (33.2440, 126.5651, "진도"),
    "177": (33.3939, 126.8809, "해남"),
    "184": (33.2936, 126.1628, "제주"),
    "185": (33.5141, 126.5297, "고산"),
    "188": (37.5175, 126.7238, "강화"),
    "189": (37.2473, 127.0414, "이천"),
    "192": (36.9951, 127.0897, "진천"),
    "201": (34.7627, 127.3699, "강진"),
    "202": (35.0635, 126.6828, "장흥"),
    "203": (33.9624, 126.2919, "영광"),
    "211": (36.7792, 128.5199, "봉화"),
    "212": (35.9907, 129.2143, "영덕"),
    "216": (36.2027, 127.2507, "의성"),
    "217": (37.3451, 127.9495, "영천"),
    "221": (35.8766, 128.5906, "제천"),
    "226": (37.2473, 127.0414, "보은"),
    "232": (37.1012, 127.0797, "천안"),
    "235": (35.0635, 126.6828, "보령"),
    "236": (36.4039, 126.6570, "부여"),
    "238": (36.3718, 127.3752, "금산"),
    "243": (35.1706, 128.0770, "부안"),
    "244": (35.2290, 128.6690, "임실"),
    "245": (35.0635, 126.6828, "정읍"),
    "247": (35.4030, 127.1151, "남원"),
    "248": (34.7627, 127.3699, "장수"),
    "251": (35.1706, 128.0770, "진주"),
    "252": (35.5741, 128.7251, "양산"),
}


def _haversine(la1, lo1, la2, lo2):
    R = 6371.0; p = np.pi / 180
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_asos(wq_stations: pd.DataFrame, max_km: float = 100.0) -> dict:
    """WQ 지점 → 최근접 ASOS 관측소 매핑. {station: (asos_code, km)}."""
    asos_codes = list(ASOS_META.keys())
    asos_lats = np.array([ASOS_META[c][0] for c in asos_codes])
    asos_lons = np.array([ASOS_META[c][1] for c in asos_codes])
    result = {}
    for _, w in wq_stations.iterrows():
        if pd.isna(w["lat"]) or pd.isna(w["lon"]):
            continue
        dkm = _haversine(float(w["lat"]), float(w["lon"]), asos_lats, asos_lons)
        j = int(np.argmin(dkm))
        if dkm[j] <= max_km:
            result[w["station"]] = (asos_codes[j], float(dkm[j]))
    return result


# 추출할 기상 채널
WEATHER_CHANNELS = {
    "cov_air_temp": "기온(°C)",
    "cov_rainfall": "강수량(mm)",
    "cov_wind_speed": "풍속(m/s)",
    "cov_humidity": "습도(%)",
    "cov_solar_rad": "일사(MJ/m2)",
}
WEATHER_COV_NAMES = list(WEATHER_CHANNELS.keys())


def load_asos_all(needed_codes: set) -> dict:
    """전 ASOS 파일에서 필요한 지점만 로드. {code: DataFrame(index=time, cols=5채널)}."""
    chunks = {c: [] for c in needed_codes}
    for d in ASOS_DIRS:
        for fn in track(sorted(os.listdir(d)), desc="ASOS 로드", task=TASK):
            if not fn.endswith(".csv"):
                continue
            # 파일명에서 지점코드 추출: SURFACE_ASOS_{code}_HR_*
            m = re.match(r"SURFACE_ASOS_(\d+)_HR_", fn)
            if not m:
                continue
            code = m.group(1)
            if code not in needed_codes:
                continue
            try:
                df = pd.read_csv(os.path.join(d, fn), encoding="cp949", dtype=str,
                                 on_bad_lines="skip")
            except Exception:
                continue
            if "일시" not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["일시"], errors="coerce").dt.floor("h")
            df = df[df["time"].notna()]
            row = {"time": df["time"]}
            for cov_name, col_name in WEATHER_CHANNELS.items():
                if col_name in df.columns:
                    row[cov_name] = pd.to_numeric(df[col_name], errors="coerce")
                else:
                    row[cov_name] = np.nan
            chunks[code].append(pd.DataFrame(row))

    series = {}
    for code, dfs in chunks.items():
        if not dfs:
            continue
        cat = pd.concat(dfs, ignore_index=True)
        cat = cat.groupby("time", as_index=True).mean()
        cat = cat.sort_index()
        # ASOS에서 비가 안 오면 강수량=NaN, 야간이면 일사=NaN → 0으로 채움
        # 단, ASOS 데이터가 실제로 존재하는 시점에서만 (기온이 있으면 관측시점으로 판단)
        has_obs = cat["cov_air_temp"].notna()
        if "cov_rainfall" in cat.columns:
            cat.loc[has_obs & cat["cov_rainfall"].isna(), "cov_rainfall"] = 0.0
        if "cov_solar_rad" in cat.columns:
            cat.loc[has_obs & cat["cov_solar_rad"].isna(), "cov_solar_rad"] = 0.0
        series[code] = cat
    return series


def main():
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv"))
    sidx["station"] = sidx["station"].astype(str)

    # 1) 최근접 ASOS 매핑
    nmap = nearest_asos(sidx)
    log(f"ASOS 매핑: {len(nmap)}/{len(sidx)} 지점 링크 완료", TASK)
    for st, (code, km) in sorted(nmap.items())[:5]:
        name = ASOS_META.get(code, ("", "", ""))[2] if code in ASOS_META else "?"
        log(f"  {st} → ASOS {code}({name}) {km:.1f}km", TASK)

    # 2) 필요한 ASOS 코드만 로드
    needed = {c for c, _ in nmap.values()}
    log(f"필요 ASOS 지점: {len(needed)}개", TASK)
    series = load_asos_all(needed)
    log(f"로드 완료: {len(series)}개 ASOS 지점", TASK)

    # 3) 수계별 WQ 시간격자에 결합
    for river in ["han", "nak", "geum", "yeong"]:
        fp = os.path.join(OUT, f"{river}_auto_hourly_wide.parquet")
        if not os.path.exists(fp):
            continue
        wide = pd.read_parquet(fp, columns=["time", "station"])
        wide["station"] = wide["station"].astype(str)
        outs = []
        for st, g in track(wide.groupby("station"), desc=f"{river} 기상결합", task=TASK):
            grid = pd.to_datetime(g["time"]).sort_values()
            base = pd.DataFrame({"time": grid.values, "station": st}).set_index("time")
            code = nmap.get(st, (None, None))[0]
            if code is not None and code in series:
                s = series[code].reindex(base.index)
                for c in WEATHER_COV_NAMES:
                    base[c] = s[c].values if c in s.columns else np.nan
            else:
                for c in WEATHER_COV_NAMES:
                    base[c] = np.nan
            outs.append(base.reset_index())
        cov = pd.concat(outs, ignore_index=True)
        cov["station"] = cov["station"].astype("category")
        outfp = os.path.join(OUT, f"{river}_weather_hourly.parquet")
        cov.to_parquet(outfp, index=False)
        covstat = {c: round(cov[c].notna().mean(), 3) for c in WEATHER_COV_NAMES}
        log(f"[{river}] 저장 {outfp} shape={cov.shape} 커버리지={covstat}", TASK)

    # 매핑 요약 저장
    link_rows = []
    for st in sidx["station"]:
        code, km = nmap.get(st, (None, None))
        name = ASOS_META.get(code, ("", "", ""))[2] if code and code in ASOS_META else None
        link_rows.append({"station": st, "asos_code": code, "asos_name": name,
                          "asos_km": round(km, 2) if km else None})
    pd.DataFrame(link_rows).to_csv(os.path.join(OUT, "weather_links.csv"),
                                   index=False, encoding="utf-8-sig")
    log("weather_links.csv 저장 완료", TASK)


if __name__ == "__main__":
    main()
