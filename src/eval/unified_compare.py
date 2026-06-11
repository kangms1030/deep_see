# -*- coding: utf-8 -*-
"""3계층 통합 비교 리포트: 확률 품질 → 점추정 보조 → 경보 성능.

기존 compare.py(NSE 중심) + prob_compare.py(확률)를 통합하여,
확률분위 예측 모델(Chronos-2)과 점예측 모델(Legacy)의 공정한 비교를 수행.

3계층 평가 체계:
  Tier 1: 확률 분포 품질 (CRPS-skill, Coverage, Calibration, Winkler, PIT)
  Tier 2: 점추정 보조 (NSE, RSR, RMSE, PBIAS)
  Tier 3: 경보 성능 (BSS, PR-AUC, F1, POD/FAR/CSI/ETS, Lead Time)

입력: reports/predictions/{legacy,chronos}_*.parquet
산출: reports/tables/unified_compare.csv, reports/unified_compare.md
실행: python -m src.eval.unified_compare
"""
from __future__ import annotations
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data import sources as S
from src.eval import unified_metrics as UM
from src.alert import thresholds as TH

PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
TARGETS = ["do", "toc", "tn", "tp", "chl-a"]
WINDOW = 120
MIN_SAMPLES = 30


# ---------- 레거시 확률화 (인과 잔차분위) ----------
def legacy_quantiles_causal(times, pred, obs, label_lag_h, window=WINDOW, min_samples=MIN_SAMPLES):
    """레거시 점추정 → 인과 잔차분위 예측밴드 [n,7]."""
    times = pd.to_datetime(np.asarray(times))
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    resid = obs - pred
    realize = times + pd.Timedelta(hours=label_lag_h)
    n = len(pred)
    out = np.full((n, len(Q)), np.nan)
    rv = realize.values; tv = times.values
    for i in range(n):
        if not np.isfinite(pred[i]):
            continue
        avail = (rv <= tv[i]) & np.isfinite(resid)
        idx = np.where(avail)[0]
        if len(idx) < min_samples:
            continue
        past = resid[idx[-window:]]
        out[i] = pred[i] + np.quantile(past, Q)
    return out


# ---------- 데이터 로드 ----------
def _load_pair(river, station, target):
    lf = os.path.join(PRED, f"legacy_{river}_{station}_{target}.parquet")
    if not os.path.exists(lf):
        return None
    leg = pd.read_parquet(lf)
    leg["origin_time"] = pd.to_datetime(leg["origin_time"]).dt.normalize()

    # system_out 또는 chronos predictions
    sys_path = os.path.join(S.DEEP_SEE, "system_out", "forecasts", f"{river}_{station}.parquet")
    if os.path.exists(sys_path):
        df = pd.read_parquet(sys_path)
        df = df[df["target"] == target].copy()
        if df.empty:
            return None
        df["origin_time"] = pd.to_datetime(df["asof"]).dt.normalize()
        pivoted = pd.DataFrame(index=df["origin_time"].unique())
        pivoted.index.name = "origin_time"
        for d in range(1, 6):
            d_df = df[df["horizon_day"] == d].set_index("origin_time")
            if d_df.empty:
                continue
            d_df = d_df[~d_df.index.duplicated(keep='first')]
            pivoted[f"obs_d{d}"] = d_df["obs"]
            for q in Q:
                pivoted[f"q{q}_d{d}"] = d_df[f"q{q}"]
        chr_df = pivoted.reset_index()
    else:
        cf = os.path.join(PRED, f"chronos_lora_{river}_{station}_{target}.parquet")
        if not os.path.exists(cf):
            return None
        chr_df = pd.read_parquet(cf)
        chr_df["origin_time"] = pd.to_datetime(chr_df["origin_time"]).dt.normalize()

    return leg.merge(chr_df, on="origin_time", suffixes=("_leg", "_chr"))


def build_evaluation_data():
    """전 (river,station,target,horizon) 평가 데이터 구축."""
    rows = []
    for lf in sorted(glob.glob(os.path.join(PRED, "legacy_*.parquet"))):
        base = os.path.basename(lf)[len("legacy_"):-len(".parquet")]
        p = base.split("_"); river, station, target = p[0], p[1], "_".join(p[2:])
        m = _load_pair(river, station, target)
        if m is None or m.empty:
            continue
        m = m.sort_values("origin_time").reset_index(drop=True)
        times = m["origin_time"]

        for d in range(1, 6):
            obs_col = f"obs_d{d}_leg" if f"obs_d{d}_leg" in m else f"obs_d{d}"
            obs = m[obs_col].to_numpy(float)
            legp = m[f"pred_d{d}"].to_numpy(float)
            chrq = m[[f"q{q}_d{d}" for q in Q]].to_numpy(float)
            legq = legacy_quantiles_causal(times, legp, obs, label_lag_h=d * 24)

            ok = np.isfinite(legq).all(1) & np.isfinite(obs) & np.isfinite(chrq).all(1)
            for i in np.where(ok)[0]:
                row = {"river": river, "station": station, "target": target,
                       "horizon": d, "origin_time": times.iloc[i], "obs": obs[i],
                       "leg_point": legp[i], "chr_med": chrq[i, QI[0.5]]}
                for qi, q in enumerate(Q):
                    row[f"leg_q{q}"] = legq[i, qi]
                    row[f"chr_q{q}"] = chrq[i, qi]
                rows.append(row)
    return pd.DataFrame(rows)


# ---------- Tier 1: 확률 분포 품질 ----------
def tier1_distribution(long: pd.DataFrame) -> pd.DataFrame:
    """타깃별 확률 분포 품질 지표."""
    cells = []
    for tg, g in long.groupby("target"):
        o = g["obs"].to_numpy()
        legq = g[[f"leg_q{q}" for q in Q]].to_numpy()
        chrq = g[[f"chr_q{q}" for q in Q]].to_numpy()

        # CRPS
        leg_crps = UM.crps_from_quantiles(o, Q, legq)
        chr_crps = UM.crps_from_quantiles(o, Q, chrq)

        # Climatology CRPS
        o_clean = o[np.isfinite(o)]
        if len(o_clean) > 1200:
            o_sample = np.random.default_rng(0).choice(o_clean, 1200, replace=False)
        else:
            o_sample = o_clean
        crps_clim = float(0.5 * np.mean(np.abs(o_sample[:, None] - o_sample[None, :]))) if len(o_sample) > 1 else np.nan

        rec = {"target": tg, "n": len(o),
               "leg_crps": leg_crps, "chr_crps": chr_crps, "crps_clim": crps_clim,
               "leg_crps_skill": UM.crps_skill_score(leg_crps, crps_clim),
               "chr_crps_skill": UM.crps_skill_score(chr_crps, crps_clim),
               "leg_cov80": UM.coverage(o, legq[:, QI[0.1]], legq[:, QI[0.9]]),
               "chr_cov80": UM.coverage(o, chrq[:, QI[0.1]], chrq[:, QI[0.9]]),
               "leg_cov90": UM.coverage(o, legq[:, QI[0.05]], legq[:, QI[0.95]]),
               "chr_cov90": UM.coverage(o, chrq[:, QI[0.05]], chrq[:, QI[0.95]]),
               "leg_calib": UM.calibration_error(o, Q, legq),
               "chr_calib": UM.calibration_error(o, Q, chrq),
               "leg_winkler80": UM.winkler_score(o, legq[:, QI[0.1]], legq[:, QI[0.9]], 0.2),
               "chr_winkler80": UM.winkler_score(o, chrq[:, QI[0.1]], chrq[:, QI[0.9]], 0.2),
               "leg_sharp80": UM.sharpness(legq[:, QI[0.1]], legq[:, QI[0.9]]),
               "chr_sharp80": UM.sharpness(chrq[:, QI[0.1]], chrq[:, QI[0.9]])}

        # PIT
        leg_pit = UM.pit_values(o, Q, legq)
        chr_pit = UM.pit_values(o, Q, chrq)
        rec["leg_pit_dev"] = UM.pit_reliability(leg_pit)["deviation"]
        rec["chr_pit_dev"] = UM.pit_reliability(chr_pit)["deviation"]

        cells.append(rec)
    return pd.DataFrame(cells)


# ---------- Tier 2: 점추정 ----------
def tier2_point(long: pd.DataFrame) -> pd.DataFrame:
    cells = []
    for tg, g in long.groupby("target"):
        o = g["obs"].to_numpy()
        lp = g["leg_point"].to_numpy()
        cm = g["chr_med"].to_numpy()
        cells.append({
            "target": tg, "n": len(o),
            "leg_nse": UM.nse(o, lp), "chr_nse": UM.nse(o, cm),
            "leg_rsr": UM.rsr(o, lp), "chr_rsr": UM.rsr(o, cm),
            "leg_rmse": UM.rmse(o, lp), "chr_rmse": UM.rmse(o, cm),
            "leg_pbias": UM.pbias(o, lp), "chr_pbias": UM.pbias(o, cm),
            "leg_mae": UM.mae(o, lp), "chr_mae": UM.mae(o, cm),
        })
    return pd.DataFrame(cells)


# ---------- Tier 3: 경보 성능 ----------
def tier3_alert(long: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score

    rows = []
    for tg, g in long.groupby("target"):
        o = g["obs"].to_numpy()
        thr = TH.get_threshold(tg)
        direction = thr["direction"]
        y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)

        # 이벤트가 너무 희소하면 퍼센타일 기준으로 대체
        if len(np.unique(y)) < 2 or y.mean() < 0.02:
            thr = TH.get_threshold(tg, o, mode="percentile")
            y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)

        legq = g[[f"leg_q{q}" for q in Q]].to_numpy()
        chrq = g[[f"chr_q{q}" for q in Q]].to_numpy()
        er = float(y.mean()); clim_brier = er * (1 - er)

        rec = {"target": tg, "thr": thr["value"], "direction": direction,
               "n": len(y), "n_events": int(y.sum()), "event_rate": er}

        for name, qm, point in [("leg", legq, g["leg_point"].to_numpy()),
                                  ("chr", chrq, g["chr_med"].to_numpy())]:
            # 확률 경보: P(임계 초과)
            p = np.array([TH.exceed_prob_from_quantiles(qm[i], Q, thr["value"], direction)
                          for i in range(len(qm))])
            mm = np.isfinite(p)
            yy, pp = y[mm], p[mm]

            brier = float(np.mean((pp - yy) ** 2))
            bss = (1 - brier / clim_brier) if clim_brier > 0 else np.nan

            # PR-AUC
            pr_auc = average_precision_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan

            # 최적 임계에서 이벤트 탐지 지표
            best_f1, best_alpha = -1, 0.5
            for a in np.round(np.arange(0.1, 0.91, 0.05), 2):
                pred_bin = (pp >= a).astype(int)
                tp = int(((pred_bin == 1) & (yy == 1)).sum())
                fp = int(((pred_bin == 1) & (yy == 0)).sum())
                fn = int(((pred_bin == 0) & (yy == 1)).sum())
                prec = tp / (tp + fp) if tp + fp else 0
                recall_val = tp / (tp + fn) if tp + fn else 0
                f1 = 2 * prec * recall_val / (prec + recall_val) if prec + recall_val else 0
                if f1 > best_f1:
                    best_f1, best_alpha = f1, float(a)

            pred_opt = (pp >= best_alpha).astype(int)
            evt_metrics = UM.event_detection_metrics(yy, pred_opt)

            rec.update({
                f"{name}_brier": brier, f"{name}_bss": bss, f"{name}_pr_auc": pr_auc,
                f"{name}_f1": best_f1, f"{name}_alpha": best_alpha,
                f"{name}_pod": evt_metrics["pod"], f"{name}_far": evt_metrics["far"],
                f"{name}_csi": evt_metrics["csi"], f"{name}_ets": evt_metrics["ets"],
                f"{name}_hss": evt_metrics["hss"],
            })
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- 리포트 생성 ----------
def write_md(t1, t2, t3):
    f = lambda x, d=3: (f"{x:.{d}f}" if pd.notna(x) else "—")

    L = ["# 통합 비교 리포트: 레거시(점) vs Chronos-2(확률)\n",
         "> 레거시 점추정을 **인과 잔차분위(conformal)** 로 확률화 → 동일 기준에서 3계층 평가.\n",
         "---\n",

         "## Tier 1: 확률 분포 품질 (핵심)\n",
         "확률예측 모델의 본질적 성능을 평가하는 핵심 계층.\n",
         "| 타깃 | n | CRPS-skill leg/chr | cov80 leg/chr | cov90 leg/chr | Winkler80 leg/chr | Sharpness80 leg/chr | Calib leg/chr | PIT편차 leg/chr |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in t1.itertuples():
        L.append(f"| {r.target} | {int(r.n)} | {f(r.leg_crps_skill)} / **{f(r.chr_crps_skill)}** | "
                 f"{f(r.leg_cov80,2)} / {f(r.chr_cov80,2)} | {f(r.leg_cov90,2)} / {f(r.chr_cov90,2)} | "
                 f"{f(r.leg_winkler80,1)} / {f(r.chr_winkler80,1)} | "
                 f"{f(r.leg_sharp80,2)} / {f(r.chr_sharp80,2)} | "
                 f"{f(r.leg_calib)} / {f(r.chr_calib)} | {f(r.leg_pit_dev,2)} / {f(r.chr_pit_dev,2)} |")
    L += ["\n> **CRPS-skill** = 1 − CRPS/CRPS(기후값). >0이면 기후값보다 우수.",
          "> **Winkler** = 구간너비 + 2/α×위반패널티. 낮을수록 우수.",
          "> **PIT편차** = PIT 히스토그램 균일도. 0에 가까울수록 보정 완벽.\n"]

    L += ["## Tier 2: 점추정 보조\n",
          "Chronos q0.5 중앙값으로 환산한 점추정 비교 (하위 호환, 참조용).\n",
          "| 타깃 | NSE leg/chr | RSR leg/chr | RMSE leg/chr | PBIAS(%) leg/chr |",
          "|---|---|---|---|---|"]
    for r in t2.itertuples():
        L.append(f"| {r.target} | {f(r.leg_nse)} / **{f(r.chr_nse)}** | "
                 f"{f(r.leg_rsr)} / {f(r.chr_rsr)} | "
                 f"{f(r.leg_rmse)} / {f(r.chr_rmse)} | "
                 f"{f(r.leg_pbias,1)} / {f(r.chr_pbias,1)} |")
    L += ["\n> NSE는 **보조 지표**로만 사용. 확률예측의 분포를 중앙값 하나로 평가하므로 Chronos에 보수적.\n"]

    L += ["## Tier 3: 경보(예보) 성능\n",
          "프로젝트 목적('동적 확률 기반 수질 오염 예보')에 직결되는 핵심 평가.\n",
          "| 타깃 | 이벤트율 | BSS leg/chr | PR-AUC leg/chr | F1 leg/chr | POD leg/chr | CSI leg/chr | ETS leg/chr |",
          "|---|---|---|---|---|---|---|---|"]
    for r in t3.itertuples():
        L.append(f"| {r.target} | {f(r.event_rate,3)} | "
                 f"{f(r.leg_bss,2)} / **{f(r.chr_bss,2)}** | "
                 f"{f(r.leg_pr_auc,2)} / {f(r.chr_pr_auc,2)} | "
                 f"{f(r.leg_f1,2)} / {f(r.chr_f1,2)} | "
                 f"{f(r.leg_pod,2)} / {f(r.chr_pod,2)} | "
                 f"{f(r.leg_csi,2)} / {f(r.chr_csi,2)} | "
                 f"{f(r.leg_ets,2)} / {f(r.chr_ets,2)} |")
    L += ["\n> **BSS** = 1 − Brier/[p(1−p)]. 기후값 대비 스킬.",
          "> **CSI** = TP/(TP+FP+FN). 경보의 종합 정확도.",
          "> **ETS** = 우연 보정 CSI. >0이면 우연보다 스킬 있음.\n",

          "## 핵심 요약\n",
          "1. **Tier 1(확률 품질)이 주 비교 기준**: NSE는 점예측 모델에 유리한 지표이므로 확률분위 예측인 Chronos-2와 공정 비교에 부적합.",
          "2. **CRPS-skill**: 두 모델을 동일한 확률분포 출발선에 세운 상태에서 비교 → Chronos의 진정한 확률 예측 능력 평가.",
          "3. **Tier 3(경보)이 실용 가치**: 프로젝트 목적이 수질 오염 예보인 만큼, BSS/PR-AUC/ETS가 최종 의사결정 지표.",
          "4. **Tier 2(점추정)는 참조용**: 기존 리포트와의 하위 호환성 유지 목적.\n",
          "> 본 리포트의 모든 수치는 `reports/tables/unified_compare_*.csv`에서 재현 가능."]

    path = os.path.join(RPT, "unified_compare.md")
    with open(path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    return path


def main():
    os.makedirs(REP, exist_ok=True)
    print("[1/5] 예측쌍 로드 + 레거시 확률화 ...")
    long = build_evaluation_data()
    if long.empty:
        print("매칭되는 예측쌍이 없습니다."); return
    print(f"  평가행 {len(long):,} (타깃 {long['target'].nunique()})")

    print("[2/5] Tier 1: 확률 분포 품질 ...")
    t1 = tier1_distribution(long)
    t1.to_csv(os.path.join(REP, "unified_compare_tier1.csv"), index=False, encoding="utf-8-sig")

    print("[3/5] Tier 2: 점추정 보조 ...")
    t2 = tier2_point(long)
    t2.to_csv(os.path.join(REP, "unified_compare_tier2.csv"), index=False, encoding="utf-8-sig")

    print("[4/5] Tier 3: 경보 성능 ...")
    t3 = tier3_alert(long)
    t3.to_csv(os.path.join(REP, "unified_compare_tier3.csv"), index=False, encoding="utf-8-sig")

    print("[5/5] 통합 리포트 생성 ...")
    md = write_md(t1, t2, t3)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== Tier 1: 확률 분포 품질 ===")
    print(t1.round(3).to_string(index=False))
    print("\n=== Tier 2: 점추정 ===")
    print(t2.round(3).to_string(index=False))
    print("\n=== Tier 3: 경보 ===")
    cols = [c for c in t3.columns if not c.startswith("leg_alpha") and not c.startswith("chr_alpha")]
    print(t3[cols].round(3).to_string(index=False))
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()
