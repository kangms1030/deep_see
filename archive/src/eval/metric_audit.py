# -*- coding: utf-8 -*-
"""평가지표 감사: 절대 사용적합성 판정용 정규화·스킬스코어 계산.

기존 산출물(final_chr512_*.parquet, legacy_*.parquet, alert_skill_lora.csv) 재사용.
- 점추정: NSE, RSR(=RMSE/STDEV_obs, 스케일 무관 절대지표), PBIAS(편향), R².
- 확률: CRPS, **CRPS skill(vs 기후값)**, coverage80/90 gap(절대 목표 대비).
- 경보: **Brier Skill Score(vs 기후값)** = 1 - Brier/(p(1-p)).
절대 등급(Moriasi 2015 / 레거시 매뉴얼): NSE>0.8 VG·0.7~0.8 G·0.5~0.7 S·≤0.5 U;
RSR 0~0.5 VG·~0.6 G·~0.7 S·>0.7 U; |PBIAS| 작을수록 좋음.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
from src.data import sources as S

PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def nse(o, s):
    d = ((o - o.mean()) ** 2).sum()
    return 1 - ((o - s) ** 2).sum() / d if d > 0 else np.nan

def rsr(o, s):
    sd = o.std()
    return np.sqrt(((o - s) ** 2).mean()) / sd if sd > 0 else np.nan

def pbias(o, s):
    return (o - s).sum() / o.sum() * 100 if o.sum() != 0 else np.nan

def crps_q(o, qmat):
    v = []
    for j, ql in enumerate(Q):
        e = o - qmat[:, j]
        v.append(np.mean(np.maximum(ql * e, (ql - 1) * e)))
    return 2 * np.mean(v)

def crps_clim(o):
    # 기후값(경험분포) 예보의 CRPS = 0.5 * 평균절대차(Gini MD)
    n = len(o)
    if n > 1500:  # 표본 과대 시 샘플
        o = np.random.default_rng(0).choice(o, 1500, replace=False)
    md = np.mean(np.abs(o[:, None] - o[None, :]))
    return 0.5 * md

def rate(nse_v, rsr_v):
    if not np.isfinite(nse_v): return "—"
    if nse_v > 0.8: return "VeryGood"
    if nse_v > 0.7: return "Good"
    if nse_v > 0.5: return "Satisfactory"
    return "Unsatisfactory"


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(PRED, "final_chr512_*.parquet"))):
        base = os.path.basename(fp)[len("final_chr512_"):-len(".parquet")]
        # river_station_target  (target에 '-' 포함: chl-a)
        river, station, target = base.split("_", 2)
        df = pd.read_parquet(fp)
        m = df["obs_d5"].notna()
        o = df.loc[m, "obs_d5"].to_numpy()
        if len(o) < 10: continue
        chr_med = df.loc[m, "chr_median_d5"].to_numpy()
        leg = df.loc[m, "legacy_pred_d5"].to_numpy()
        qmat = df.loc[m, [f"chr_q{q}_d5" for q in Q]].to_numpy()
        cc = crps_clim(o); cm = crps_q(o, qmat)
        cov80 = np.mean((o >= qmat[:, 1]) & (o <= qmat[:, 5]))
        cov90 = np.mean((o >= qmat[:, 0]) & (o <= qmat[:, 6]))
        mleg = np.isfinite(leg)
        rows.append({"river": river, "target": target,
                     "chr_nse": nse(o, chr_med), "chr_rsr": rsr(o, chr_med), "chr_pbias": pbias(o, chr_med),
                     "leg_nse": nse(o[mleg], leg[mleg]) if mleg.sum() > 5 else np.nan,
                     "leg_rsr": rsr(o[mleg], leg[mleg]) if mleg.sum() > 5 else np.nan,
                     "leg_pbias": pbias(o[mleg], leg[mleg]) if mleg.sum() > 5 else np.nan,
                     "crps": cm, "crps_clim": cc, "crps_skill": 1 - cm / cc if cc > 0 else np.nan,
                     "cov80": cov80, "cov90": cov90, "n": len(o)})
    df = pd.DataFrame(rows)
    df["rating"] = [rate(r.chr_nse, r.chr_rsr) for r in df.itertuples()]
    df.to_csv(os.path.join(REP, "metric_audit.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== 타깃별 평균 (Chronos-LoRA512, 5일차) ===")
    g = df.groupby("target").agg(chr_nse=("chr_nse","mean"), chr_rsr=("chr_rsr","mean"),
        chr_pbias=("chr_pbias","mean"), crps_skill=("crps_skill","mean"),
        cov80=("cov80","mean"), cov90=("cov90","mean"), leg_nse=("leg_nse","mean")).round(3)
    g["abs_rating"] = [rate(v, np.nan) for v in g["chr_nse"]]
    print(g.to_string())
    print(f"\n전체 평균 NSE={df.chr_nse.mean():.3f} median={df.chr_nse.median():.3f} | "
          f"RSR={df.chr_rsr.mean():.3f} | |PBIAS|={df.chr_pbias.abs().mean():.1f}% | "
          f"CRPS_skill={df.crps_skill.mean():.3f} | cov80={df.cov80.mean():.2f} cov90={df.cov90.mean():.2f}")

    # ===== 경보: Brier Skill Score =====
    ap = os.path.join(REP, "alert_skill_lora.csv")
    if os.path.exists(ap):
        a = pd.read_csv(ap)
        if "chr_brier" in a and "chr_event_rate" in a:
            p = a["chr_event_rate"]; clim = p * (1 - p)
            a["chr_BSS"] = 1 - a["chr_brier"] / clim.replace(0, np.nan)
            print("\n=== 경보 Brier Skill Score (vs 기후값, >0=유용) ===")
            print(a.groupby("target").agg(BSS=("chr_BSS","mean"), F1=("chr_f1","mean"),
                  PRAUC=("chr_pr_auc","mean"), ev=("chr_event_rate","mean")).round(3).to_string())
            print(f"평균 BSS={a['chr_BSS'].mean():.3f}")


if __name__ == "__main__":
    main()
