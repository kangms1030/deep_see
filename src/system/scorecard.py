# -*- coding: utf-8 -*-
"""스코어카드/모니터링: replay 저장소(forecasts/alerts)에서 독립적으로 지표 집계.

점추정(NSE/RSR/PBIAS, day5) · 확률(CRPS-skill, coverage raw/cal) · 경보(BSS/PR-AUC/F1/lead)
· 가드레일(ΔNSE vs persistence) · 절대등급/SLO. 저장소만 입력으로 쓰므로 파이프라인
무결성 점검(sanity gate: 대표지점 NSE ≈ final_eval_metrics.csv)을 겸한다.

실행: PYTHONIOENCODING=utf-8 python -m src.system.run_system score
"""
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

from src.system import config as C
from src.eval import metrics as Mx
from src.eval.metric_audit import rate as abs_rate
from src.utils.progress import log

TASK = "sys_score"


def _crps_clim(o):
    o = o[np.isfinite(o)]
    if len(o) < 3:
        return np.nan
    if len(o) > 1200:
        o = np.random.default_rng(0).choice(o, 1200, replace=False)
    return 0.5 * np.mean(np.abs(o[:, None] - o[None, :]))


def _load_forecasts() -> pd.DataFrame:
    fps = glob.glob(os.path.join(C.FORECASTS, "*.parquet"))
    if not fps:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(fp) for fp in fps], ignore_index=True)


def _point_probrows(fc: pd.DataFrame) -> pd.DataFrame:
    """지점·타깃별 점추정/확률 지표(day5 기준, final_eval와 동일 규약)."""
    d5 = fc[fc.horizon_day == 5]
    rows = []
    for (rv, st, tg), g in d5.groupby(["river", "station", "target"]):
        o = g["obs"].to_numpy(); med = g["median"].to_numpy()
        qmat = g[[f"q{q}" for q in C.Q]].to_numpy()
        cc = _crps_clim(o); cm = Mx.crps_from_quantiles(o, C.Q, qmat)
        rows.append({"river": rv, "station": st, "target": tg, "n": int(np.isfinite(o).sum()),
                     "nse": Mx.nse(o, med), "rsr": np.sqrt(1 - Mx.nse(o, med)) if Mx.nse(o, med) <= 1 else np.nan,
                     "pbias": Mx.pbias(o, med),
                     "pers_nse": Mx.nse(o, g["persistence"].to_numpy()),
                     "crps": cm, "crps_skill": (1 - cm / cc) if (cc and cc > 0) else np.nan,
                     "cov80_raw": Mx.coverage(o, g["q0.1"], g["q0.9"]),
                     "cov80_cal": Mx.coverage(o, g["q0.1_cal"], g["q0.9_cal"]),
                     "cov90_raw": Mx.coverage(o, g["q0.05"], g["q0.95"]),
                     "cov90_cal": Mx.coverage(o, g["q0.05_cal"], g["q0.95_cal"])})
    return pd.DataFrame(rows)


def _alert_metrics(al: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score
    rows = []
    for tg, g in al.groupby("target"):
        y = g["is_event"].astype(int).to_numpy(); p = g["p_exceed"].to_numpy()
        m = np.isfinite(p)
        y, p, fired = y[m], p[m], g["fired"].to_numpy()[m]
        if len(y) == 0 or len(np.unique(y)) < 2:
            rows.append({"target": tg, "pr_auc": np.nan, "f1": np.nan, "bss": np.nan,
                         "event_rate": float(y.mean()) if len(y) else np.nan, "lead_day": np.nan})
            continue
        tp = int(((fired == 1) & (y == 1)).sum()); fp = int(((fired == 1) & (y == 0)).sum())
        fn = int(((fired == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        er = y.mean(); brier = float(np.mean((p - y) ** 2))
        bss = 1 - brier / (er * (1 - er)) if 0 < er < 1 else np.nan
        # 리드타임: 이벤트가 실제 발생한 origin에서 가장 이른 발령일 평균
        ev = g[g["is_event"] & g["fired"]]
        lead = float(ev["lead_day"].dropna().mean()) if len(ev) else np.nan
        rows.append({"target": tg, "pr_auc": average_precision_score(y, p), "f1": f1,
                     "bss": bss, "event_rate": float(er), "lead_day": lead})
    return pd.DataFrame(rows)


def _slo_pass(tg, nse, cov80_cal, pr_auc):
    s = C.SLO.get(tg, {})
    ok = (np.isfinite(nse) and nse >= s.get("nse", -np.inf))
    lo, hi = s.get("cov80", (0, 1))
    ok &= (np.isfinite(cov80_cal) and lo <= cov80_cal <= hi)
    if np.isfinite(pr_auc):
        ok &= pr_auc >= s.get("pr_auc", 0)
    return bool(ok)


def run():
    C.ensure_dirs()
    fc = _load_forecasts()
    if fc.empty:
        log("forecasts 저장소가 비어있음. 먼저 replay 실행.", TASK); return
    pr = _point_probrows(fc)
    al_path = os.path.join(C.ALERTS, "alert_log.parquet")
    am = _alert_metrics(pd.read_parquet(al_path)) if os.path.exists(al_path) else pd.DataFrame()

    # 타깃별 집계
    agg = pr.groupby("target").agg(
        n=("n", "sum"), nse=("nse", "median"), rsr=("rsr", "median"), pbias=("pbias", "mean"),
        crps_skill=("crps_skill", "mean"), cov80_raw=("cov80_raw", "mean"),
        cov80_cal=("cov80_cal", "mean"), cov90_raw=("cov90_raw", "mean"),
        cov90_cal=("cov90_cal", "mean"), pers_nse=("pers_nse", "median")).reset_index()
    agg["vs_persistence"] = agg["nse"] - agg["pers_nse"]
    if len(am):
        agg = agg.merge(am, on="target", how="left")
    agg["rating"] = [abs_rate(v, np.nan) for v in agg["nse"]]
    agg["slo_pass"] = [_slo_pass(r.target, r.nse, r.cov80_cal, getattr(r, "pr_auc", np.nan))
                       for r in agg.itertuples()]

    pr.to_csv(os.path.join(C.SCORECARD, "by_station.csv"), index=False, encoding="utf-8-sig")
    for tg, g in pr.groupby("target"):
        g.to_csv(os.path.join(C.SCORECARD, f"{tg}.csv"), index=False, encoding="utf-8-sig")
    agg.to_csv(os.path.join(C.SCORECARD, "summary.csv"), index=False, encoding="utf-8-sig")

    sanity = _sanity_gate(pr)
    _write_md(agg, sanity)
    _print(agg, sanity)
    return agg


def _sanity_gate(pr: pd.DataFrame) -> pd.DataFrame:
    """대표지점 day5 NSE가 final_eval_metrics.csv(chronos_lora512)와 일치하는지 비교."""
    fe_path = os.path.join(C.DEEP_SEE, "reports", "tables", "final_eval_metrics.csv")
    if not os.path.exists(fe_path):
        return pd.DataFrame()
    fe = pd.read_csv(fe_path)
    fe = fe[fe["model"] == "chronos_lora512"][["river", "target", "station", "nse"]]
    fe["station"] = fe["station"].astype(str)
    m = pr.merge(fe, on=["river", "target", "station"], suffixes=("_sys", "_fe"))
    if m.empty:
        return m
    m["abs_diff"] = (m["nse_sys"] - m["nse_fe"]).abs()
    return m[["river", "target", "station", "nse_sys", "nse_fe", "abs_diff"]]


def _write_md(agg, sanity):
    f = lambda x, d=3: (f"{x:.{d}f}" if pd.notna(x) else "—")
    L = ["# 시스템 스코어카드 (replay 백테스트, 전 지점)\n",
         "Chronos-2 LoRA@512 + conformal 보정 예보·경보 파이프라인의 보유데이터 검증 결과.\n",
         "## 타깃별 종합 (day5 중앙 NSE 기준)\n",
         "| 타깃 | 모드 | n | NSE | RSR | CRPS-skill | cov80 raw→cal | cov90 raw→cal | "
         "ΔNSE(vs pers) | PR-AUC | F1 | BSS | 리드(일) | 등급 | SLO |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        L.append(
            f"| {r.target} | {C.TARGET_MODE.get(r.target,'')} | {int(r.n)} | {f(r.nse)} | {f(r.rsr)} | "
            f"{f(r.crps_skill)} | {f(r.cov80_raw,2)}→{f(r.cov80_cal,2)} | "
            f"{f(r.cov90_raw,2)}→{f(r.cov90_cal,2)} | {f(r.vs_persistence,3)} | "
            f"{f(getattr(r,'pr_auc',np.nan),2)} | {f(getattr(r,'f1',np.nan),2)} | "
            f"{f(getattr(r,'bss',np.nan),2)} | {f(getattr(r,'lead_day',np.nan),1)} | "
            f"{r.rating} | {'✅' if r.slo_pass else '⚠️'} |")
    L += ["\n## 해석 가이드",
          "- **ΔNSE(vs persistence)>0** 이어야 모델이 강한 베이스라인 대비 가치를 줌(가드레일).",
          "- **cov80/90 cal**이 목표(0.80/0.90)에 raw보다 근접하면 conformal 보정 유효.",
          "- **PR-AUC/BSS/리드**는 경보(이벤트) 유용성. Chl-a·T-P는 경보 중심으로 해석."]
    if len(sanity):
        L += ["\n## Sanity gate (대표지점 NSE vs final_eval_metrics.csv)",
              f"- 비교쌍 {len(sanity)}개, 평균 |Δ|={sanity['abs_diff'].mean():.4f}, "
              f"최대 |Δ|={sanity['abs_diff'].max():.4f} "
              f"({'일치 ✅' if sanity['abs_diff'].mean() < 0.1 else '편차 확인 ⚠️'})",
              "  - 점추정 NSE는 conformal 보정과 무관(보정은 밴드만 조정). 소폭 편차는 **origin 집합 차이**에서 기인:",
              "    시스템은 컨텍스트가 test 구간에 완전히 포함되도록 origin을 i_va+512부터 잡아(누수 0) final_eval(i_va+240 오프셋)보다 엄격하다."]
    with open(os.path.join(C.SCORECARD, "scorecard.md"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(L) + "\n")
    log("scorecard.md 저장", TASK)


def _print(agg, sanity):
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== 타깃별 스코어카드 ===")
    cols = [c for c in ["target", "n", "nse", "rsr", "crps_skill", "cov80_cal", "cov90_cal",
                        "vs_persistence", "pr_auc", "f1", "bss", "lead_day", "rating", "slo_pass"]
            if c in agg.columns]
    print(agg[cols].round(3).to_string(index=False))
    if len(sanity):
        print(f"\n[sanity] 대표지점 NSE |Δ| 평균={sanity['abs_diff'].mean():.4f} "
              f"최대={sanity['abs_diff'].max():.4f}")
