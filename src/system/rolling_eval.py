# -*- coding: utf-8 -*-
"""레거시 vs 최종 모델(Chronos-2 LoRA ctx512) 롤링 윈도우 head-to-head 시각화.

보유 test 구간을 **겹치지 않는 5일 윈도우(stride=120h)**로 타일링하여 각 날짜를 정확히
한 번만 예측 → 레거시(reports/predictions/legacy_*)와 Chronos(system_out/forecasts/*)를
날짜 기준으로 병합해 비교. 타깃별 PNG(대표 4지점 서브플롯) 저장.

- 관측 기준: Chronos 저장소 obs(일평균). 동일 관측 대비 두 모델 NSE를 함께 표기(직접 비교).
- 두 모델 origin 시각이 8h 어긋나나 둘 다 일평균이라 날짜 정렬로 정합(시각화 오차 무시 수준).

실행: PYTHONIOENCODING=utf-8 python -m src.system.rolling_eval
산출: plots/rolling_{target}.png  (plots/ 는 .gitignore)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

from src.system import config as C
from src.eval import metrics as Mx
from src.alert import thresholds as TH
from src.utils.progress import log

TASK = "rolling_eval"
OUT_DIR = os.path.join(C.DEEP_SEE, "plots")
HORIZONS = C.HORIZON_DAYS         # 5
STEP = HORIZONS                   # 겹치지 않으려면 5 origin(=120h)마다 1개


def _chronos_nonoverlap(df_tg: pd.DataFrame) -> pd.DataFrame:
    """Chronos 저장소 → 겹치지 않는 날짜별 (median, lo, hi, obs)."""
    df = df_tg.copy()
    df["asof"] = pd.to_datetime(df["asof"])
    origins = np.sort(df["asof"].unique())[::STEP]      # 120h 간격
    rows = []
    for T in origins:
        g = df[df["asof"] == T]
        for d in range(1, HORIZONS + 1):
            r = g[g["horizon_day"] == d]
            if r.empty:
                continue
            r = r.iloc[0]
            date = (pd.Timestamp(T) + pd.Timedelta(days=d)).normalize()
            rows.append({"date": date, "median": r["median"],
                         "lo": r["q0.1_cal"], "hi": r["q0.9_cal"], "obs": r["obs"]})
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def _legacy_nonoverlap(df: pd.DataFrame) -> pd.DataFrame:
    """레거시 저장소 → 겹치지 않는 날짜별 pred."""
    d = df.copy()
    d["origin_time"] = pd.to_datetime(d["origin_time"])
    origins = np.sort(d["origin_time"].unique())[::STEP]
    rows = []
    for T in origins:
        r = d[d["origin_time"] == T].iloc[0]
        for k in range(1, HORIZONS + 1):
            date = (pd.Timestamp(T) + pd.Timedelta(days=k)).normalize()
            rows.append({"date": date, "legacy": r[f"pred_d{k}"]})
    return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def _panel(ax, river, st, tg):
    cfp = os.path.join(C.FORECASTS, f"{river}_{st}.parquet")
    lfp = os.path.join(C.DEEP_SEE, "reports", "predictions", f"legacy_{river}_{st}_{tg}.parquet")
    if not (os.path.exists(cfp) and os.path.exists(lfp)):
        ax.set_title(f"{river}/{st}/{tg}: 데이터 없음"); return None
    chr = _chronos_nonoverlap(pd.read_parquet(cfp).query("target == @tg"))
    leg = _legacy_nonoverlap(pd.read_parquet(lfp))
    m = chr.join(leg, how="inner")                      # 공통 날짜
    if m.empty:
        ax.set_title(f"{river}/{st}/{tg}: 공통구간 없음"); return None
    obs = m["obs"].to_numpy()
    nse_c = Mx.nse(obs, m["median"].to_numpy())
    nse_l = Mx.nse(obs, m["legacy"].to_numpy())
    th = TH.THRESHOLDS[tg]

    ax.fill_between(m.index, m["lo"], m["hi"], color="tab:blue", alpha=0.18, label="Chronos 80% PI")
    ax.plot(m.index, m["legacy"], color="tab:orange", lw=1.1, label=f"레거시 (NSE={nse_l:.2f})")
    ax.plot(m.index, m["median"], color="tab:blue", lw=1.3, label=f"Chronos (NSE={nse_c:.2f})")
    ax.scatter(m.index, obs, s=10, c="k", zorder=5, label="관측")
    ax.axhline(th["value"], color="r", ls="--", lw=0.8, alpha=0.7)
    rk = {"han": "한강", "nak": "낙동강", "geum": "금강", "yeong": "영산강"}[river]
    ax.set_title(f"{rk} {st} — {tg}  (n={int(np.isfinite(obs).sum())}일, 겹침없음)", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.2)
    return {"river": river, "target": tg, "nse_chronos": nse_c, "nse_legacy": nse_l,
            "n": int(np.isfinite(obs).sum())}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sidx = C.load_station_index()
    reps = sidx[sidx.is_representative].copy()
    reps["station"] = reps["station"].astype(str)
    order = {"han": 0, "nak": 1, "geum": 2, "yeong": 3}
    reps = reps.sort_values("river", key=lambda s: s.map(order))
    summary = []
    for tg in C.TARGETS:
        fig, axes = plt.subplots(len(reps), 1, figsize=(14, 3.1 * len(reps)), sharex=False)
        if len(reps) == 1:
            axes = [axes]
        for ax, r in zip(axes, reps.itertuples()):
            res = _panel(ax, r.river, str(r.station), tg)
            if res:
                summary.append(res)
        fig.suptitle(f"레거시(GAIN+GRU) vs 최종모델(Chronos-2 LoRA ctx512) — 겹치지 않는 롤링 예측: {tg}",
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = os.path.join(OUT_DIR, f"rolling_{tg}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        log(f"저장 {out}", TASK)
    if summary:
        s = pd.DataFrame(summary)
        s["delta"] = s["nse_chronos"] - s["nse_legacy"]
        s.to_csv(os.path.join(OUT_DIR, "rolling_summary.csv"), index=False, encoding="utf-8-sig")
        print("\n=== 롤링(겹침없음) NSE: 레거시 vs Chronos ===")
        print(s[["river", "target", "n", "nse_legacy", "nse_chronos", "delta"]].round(3).to_string(index=False))
        print(f"\n평균 ΔNSE(Chronos-레거시) = {s['delta'].mean():+.3f} "
              f"(Chronos 우위 {int((s['delta']>0).sum())}/{len(s)})")


if __name__ == "__main__":
    main()
