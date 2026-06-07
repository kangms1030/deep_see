# -*- coding: utf-8 -*-
"""Phase 1: 자동측정망 53지점 → 전 지역 5타깃(+보조) 시간단위 wide 데이터셋.

산출:
- data_processed/{river}_auto_hourly_wide.parquet : [time, station, <10 채널 raw(NaN=결측)>]
- data_processed/station_index.csv : 지점 메타 + 타깃별 커버리지 + 대표지점 플래그
- data_processed/splits.json : 지점별 시점기준 70/10/20 분할 경계

실행: (deep_see/ 에서)  PYTHONIOENCODING=utf-8 python -m src.data.build_dataset
"""
from __future__ import annotations
import os
import json
import pandas as pd

from src.data import sources as S
from src.utils.progress import track, log

OUT = os.path.join(S.DEEP_SEE, "data_processed")
os.makedirs(OUT, exist_ok=True)
TASK = "build_dataset"


def main():
    paths = S.auto_csv_paths()
    log(f"자동측정망 CSV {len(paths)}개 발견 (Training+Validation)", TASK)

    # --- pass 1: 전 파일 로드 → 수계별 long 누적 + 지점 메타 ---
    per_river: dict[str, list[pd.DataFrame]] = {r: [] for r in S.RIVER_BY_PREFIX.values()}
    meta: dict[str, dict] = {}
    total_rows = 0
    for p in track(paths, desc="CSV 로드", task=TASK):
        df = S.load_auto_csv(p)
        if df.empty:
            continue
        total_rows += len(df)
        for r, g in df.groupby("river", observed=True):
            per_river[r].append(g[["time", "station", "code", "value"]])
            # 지점 메타(최초 1회)
            for st, gg in g.groupby("station", observed=True):
                if st not in meta:
                    row = gg.iloc[0]
                    meta[st] = {"station": st, "river": r, "name": row["name"],
                                "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                                "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                                "cat_id": row["cat_id"]}
    log(f"정제 long 행수={total_rows:,} | 지점수={len(meta)}", TASK)

    # --- pass 2: 수계별·지점별 pivot → 시간 그리드 reindex → parquet ---
    station_rows = []
    splits = {}
    for river, chunks in per_river.items():
        if not chunks:
            continue
        long = pd.concat(chunks, ignore_index=True)
        stations = sorted(long["station"].unique())
        log(f"[{river}] long 행수={len(long):,} 지점={len(stations)} → pivot 시작", TASK)
        wides = []
        for st in track(stations, desc=f"{river} pivot", task=TASK):
            sub = long[long["station"] == st]
            piv = sub.pivot_table(index="time", columns="code", values="value", aggfunc="mean")
            piv = piv.sort_index()
            canon = S.coalesce_channels(piv)              # index=time, cols=10 채널
            full = pd.date_range(canon.index.min(), canon.index.max(), freq="h")
            wide = canon.reindex(full)
            wide.index.name = "time"
            wide = wide.reset_index()
            wide.insert(1, "station", st)
            wides.append(wide)

            # 메타 갱신: 커버리지/스팬
            n = len(wide)
            m = meta[st]
            m["start"] = str(wide["time"].iloc[0]); m["end"] = str(wide["time"].iloc[-1])
            m["n_hours_grid"] = int(n)
            for tgt in S.TARGETS:
                obs = int(wide[tgt].notna().sum())
                m[f"cov_{tgt}"] = round(obs / n, 4) if n else 0.0
                m[f"n_{tgt}"] = obs
            # 시점기준 70/10/20 분할 경계
            tr = wide["time"].iloc[int(n * 0.7)] if n > 10 else wide["time"].iloc[-1]
            va = wide["time"].iloc[int(n * 0.8)] if n > 10 else wide["time"].iloc[-1]
            splits[st] = {"train_end": str(tr), "val_end": str(va), "n": n}
            station_rows.append(m)

        river_df = pd.concat(wides, ignore_index=True)
        river_df["station"] = river_df["station"].astype("category")
        fp = os.path.join(OUT, f"{river}_auto_hourly_wide.parquet")
        river_df.to_parquet(fp, index=False)
        log(f"[{river}] 저장 {fp}  shape={river_df.shape}", TASK)
        del long, river_df, wides

    # --- station_index + 대표지점 선정 ---
    sidx = pd.DataFrame(station_rows).drop_duplicates("station").reset_index(drop=True)
    sidx["is_representative"] = False
    for river in sorted(sidx["river"].unique()):
        cand = sidx[sidx["river"] == river].copy()
        # 5타깃 모두 관측 있는 지점만, 최소 타깃 커버리지·길이 우선
        cand = cand[(cand[[f"cov_{t}" for t in S.TARGETS]] > 0).all(axis=1)]
        if cand.empty:
            cand = sidx[sidx["river"] == river].copy()
        cand["min_cov"] = cand[[f"cov_{t}" for t in S.TARGETS]].min(axis=1)
        cand = cand.sort_values(["min_cov", "n_hours_grid"], ascending=False)
        rep = cand.iloc[0]["station"]
        if river == "han" and (sidx["station"] == "S01001").any():  # 레거시=가평
            s01 = sidx[sidx["station"] == "S01001"].iloc[0]
            if all(s01[f"cov_{t}"] > 0 for t in S.TARGETS):
                rep = "S01001"
        sidx.loc[sidx["station"] == rep, "is_representative"] = True
        log(f"[{river}] 대표지점 = {rep} ({sidx[sidx.station==rep].iloc[0]['name']})", TASK)

    sidx.to_csv(os.path.join(OUT, "station_index.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    log(f"station_index.csv ({len(sidx)}지점) + splits.json 저장 완료", TASK)
    # 요약
    log("=== 대표지점 타깃 커버리지 ===", TASK)
    rep = sidx[sidx["is_representative"]]
    for _, r in rep.iterrows():
        covs = " ".join(f"{t}={r[f'cov_{t}']:.2f}" for t in S.TARGETS)
        log(f"  {r['river']:5s} {r['station']} {r['name']:8s} n={r['n_hours_grid']:6d} {covs}", TASK)


if __name__ == "__main__":
    main()
