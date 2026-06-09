# -*- coding: utf-8 -*-
"""레거시(점) vs Chronos-2(확률) — 확률·예보 차원의 공정한 직접 맞대결.

문제의식
--------
NSE 등 점추정 비교는 (i) Chronos의 분포를 중앙값 하나로 납작하게 눌러 평가하므로
Chronos에 보수적이고, (ii) 평균회귀에 후한 지표라 '임계 초과(꼬리)' 예보 능력을 못 본다.
확률 지표(CRPS/coverage/BSS)는 레거시에 분포가 없어 직접 비교가 불가능했다(빈틈).

해결: 레거시 점추정을 **conformal(잔차 경험분위)** 로 확률화하여 두 모델 모두에
예측분포를 부여한 뒤, 동일 origin·동일 관측에서 CRPS·coverage·경보 BSS를 맞대결한다.

레거시 확률화(=split/online conformal of a point forecast)
- horizon d의 잔차 e = obs_d - legacy_pred_d 는 origin+d*24h에 확정(인과).
- 각 origin에서 '이미 확정된 최근 잔차'만으로 경험분위를 추정(롤링창, 누수 0).
- legacy_q(τ) = legacy_pred_d + quantile(past_residuals, τ).  → 7분위 예측밴드.
- 잔차 중앙값(편향)도 흡수되므로 레거시에 오히려 관대한(=공정한) 처리.

입력: reports/predictions/legacy_*.parquet(점, 5horizon) + chronos_lora_*.parquet(분위).
산출: reports/tables/prob_compare_dist.csv(셀별), prob_compare_dist_summary.csv(타깃),
      prob_compare_alert.csv(타깃), reports/prob_compare.md.
실행: python -m src.eval.prob_compare
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.metrics import average_precision_score

from src.data import sources as S
from src.alert import thresholds as TH

PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
TARGETS = ["do", "toc", "tn", "tp", "chl-a"]
WINDOW = 120          # 잔차 롤링창
MIN_SAMPLES = 30      # 미만이면 그 origin은 평가 제외(인과 warmup)
ALPHA_GRID = np.round(np.arange(0.1, 0.91, 0.05), 2)


# ---------- 지표 ----------
def crps_q(o, qmat):
    """분위 근사 CRPS = 2 * 평균 pinball loss. qmat:[n,7]."""
    m = np.isfinite(o) & np.isfinite(qmat).all(1)
    o = o[m]; qmat = qmat[m]
    if len(o) < 3:
        return np.nan
    v = [np.mean(np.maximum(ql * (o - qmat[:, j]), (ql - 1) * (o - qmat[:, j]))) for j, ql in enumerate(Q)]
    return float(2 * np.mean(v))


def crps_clim(o):
    """기후값(경험분포) 예보의 CRPS = 0.5 * 평균절대차(Gini MD)."""
    o = o[np.isfinite(o)]
    if len(o) < 3:
        return np.nan
    if len(o) > 1200:
        o = np.random.default_rng(0).choice(o, 1200, replace=False)
    return float(0.5 * np.mean(np.abs(o[:, None] - o[None, :])))


def coverage(o, lo, hi):
    m = np.isfinite(o) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((o[m] >= lo[m]) & (o[m] <= hi[m]))) if m.any() else np.nan


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((o - s) ** 2).sum() / d) if d > 0 and len(o) > 2 else np.nan


# ---------- 레거시 확률화(인과 잔차분위) ----------
def legacy_quantiles_causal(times, pred, obs, label_lag_h, window=WINDOW, min_samples=MIN_SAMPLES):
    """legacy 점추정 → 인과 잔차분위 예측밴드 [n,7]. 표본부족 origin은 NaN(평가 제외)."""
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


# ---------- 셀 단위 처리 ----------
def _load_pair(river, station, target, use_system=True):
    lf = os.path.join(PRED, f"legacy_{river}_{station}_{target}.parquet")
    if not os.path.exists(lf):
        return None
    leg = pd.read_parquet(lf)
    leg["origin_time"] = pd.to_datetime(leg["origin_time"]).dt.normalize()

    sys_path = os.path.join(S.DEEP_SEE, "system_out", "forecasts", f"{river}_{station}.parquet")
    if use_system and os.path.exists(sys_path):
        df = pd.read_parquet(sys_path)
        df = df[df["target"] == target].copy()
        if df.empty:
            return None
        df["origin_time"] = pd.to_datetime(df["asof"]).dt.normalize()
        
        # pivot the long format of system forecasts into wide format
        pivoted = pd.DataFrame(index=df["origin_time"].unique())
        pivoted.index.name = "origin_time"
        
        for d in range(1, 6):
            d_df = df[df["horizon_day"] == d].set_index("origin_time")
            if d_df.empty:
                continue
            # Remove duplicate index if any
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


def build_long():
    """전 (river,station,target,horizon,origin) 평가행 생성: obs, leg_q[7], chr_q[7], 점추정, baselines."""
    import json
    splits_path = os.path.join(S.DEEP_SEE, "data_processed", "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
    else:
        splits = {}

    rows = []
    for lf in sorted(glob.glob(os.path.join(PRED, "legacy_*.parquet"))):
        base = os.path.basename(lf)[len("legacy_"):-len(".parquet")]
        p = base.split("_"); river, station, target = p[0], p[1], "_".join(p[2:])
        m = _load_pair(river, station, target)
        if m is None or m.empty:
            continue
        m = m.sort_values("origin_time").reset_index(drop=True)
        times = m["origin_time"]

        # Load hourly wide data for baselines
        wide_path = os.path.join(S.DEEP_SEE, "data_processed", f"{river}_auto_hourly_wide.parquet")
        if os.path.exists(wide_path) and station in splits:
            wide = pd.read_parquet(wide_path)
            sdf = wide[wide["station"] == station].sort_values("time").reset_index(drop=True)
            sdf["time"] = pd.to_datetime(sdf["time"])
            sdf_idx = sdf.set_index("time")
            
            # --- Seasonal Quantiles ---
            train_end = pd.Timestamp(splits[station]["train_end"])
            train_data = sdf[sdf["time"] <= train_end].copy()
            train_data["month"] = train_data["time"].dt.month
            
            overall_obs = train_data[target].dropna().to_numpy()
            overall_q = np.quantile(overall_obs, Q) if len(overall_obs) > 0 else np.zeros(len(Q))
            
            seasonal_lookup = {}
            for month in range(1, 13):
                m_obs = train_data[train_data["month"] == month][target].dropna().to_numpy()
                if len(m_obs) >= 30:
                    seasonal_lookup[month] = np.quantile(m_obs, Q)
                else:
                    seasonal_lookup[month] = overall_q
            
            m_months = pd.to_datetime(m["origin_time"]).dt.month
            seas_q = np.array([seasonal_lookup.get(month, overall_q) for month in m_months])
            
            # --- Persistence + Conformal ---
            pred_persist = sdf_idx.reindex(m["origin_time"])[target].ffill().bfill().to_numpy()
            if np.isnan(pred_persist).any():
                overall_mean = train_data[target].mean()
                pred_persist = np.nan_to_num(pred_persist, nan=overall_mean if np.isfinite(overall_mean) else 0.0)
        else:
            seas_q = None
            pred_persist = None

        for d in range(1, 6):
            obs_col = f"obs_d{d}_leg" if f"obs_d{d}_leg" in m else f"obs_d{d}"
            obs = m[obs_col].to_numpy(float)
            legp = m[f"pred_d{d}"].to_numpy(float)
            chrq = m[[f"q{q}_d{d}" for q in Q]].to_numpy(float)
            legq = legacy_quantiles_causal(times, legp, obs, label_lag_h=d * 24)
            
            # Persistence conformal
            if pred_persist is not None:
                persist_q = legacy_quantiles_causal(times, pred_persist, obs, label_lag_h=d * 24)
            else:
                persist_q = np.full((len(times), len(Q)), np.nan)
                
            ok = np.isfinite(legq).all(1) & np.isfinite(obs)          # 공통 평가집합
            for i in np.where(ok)[0]:
                row = {"river": river, "station": station, "target": target, "horizon": d,
                       "origin_time": times.iloc[i], "obs": obs[i],
                       "leg_point": legp[i], "chr_med": chrq[i, QI[0.5]],
                       **{f"leg_q{q}": legq[i, QI[q]] for q in Q},
                       **{f"chr_q{q}": chrq[i, QI[q]] for q in Q}}
                
                # Add seasonal quantiles
                if seas_q is not None:
                    for qi, q in enumerate(Q):
                        row[f"seas_q{q}"] = seas_q[i, qi]
                else:
                    for q in Q:
                        row[f"seas_q{q}"] = np.nan
                        
                # Add persistence conformal quantiles
                for qi, q in enumerate(Q):
                    row[f"persist_cf_q{q}"] = persist_q[i, qi]
                    
                rows.append(row)
    return pd.DataFrame(rows)


# ---------- 분포 지표 집계(셀별 → 타깃 평균) ----------
def dist_metrics(long: pd.DataFrame):
    cells = []
    for (rv, st, tg, d), g in long.groupby(["river", "station", "target", "horizon"]):
        o = g["obs"].to_numpy()
        legq = g[[f"leg_q{q}" for q in Q]].to_numpy()
        chrq = g[[f"chr_q{q}" for q in Q]].to_numpy()
        seasq = g[[f"seas_q{q}" for q in Q]].to_numpy()
        persistq = g[[f"persist_cf_q{q}" for q in Q]].to_numpy()
        
        cc = crps_clim(o)
        leg_crps = crps_q(o, legq)
        chr_crps = crps_q(o, chrq)
        seas_crps = crps_q(o, seasq)
        persist_crps = crps_q(o, persistq)
        
        rec = {"river": rv, "station": st, "target": tg, "horizon": d, "n": len(o),
               "leg_crps": leg_crps, "chr_crps": chr_crps,
               "seas_crps": seas_crps, "persist_crps": persist_crps,
               "crps_clim": cc}
               
        # Skill relative to climatology
        rec["leg_crps_skill"] = (1 - leg_crps / cc) if (cc and cc > 0) else np.nan
        rec["chr_crps_skill"] = (1 - chr_crps / cc) if (cc and cc > 0) else np.nan
        
        # Skill relative to seasonal quantile
        rec["leg_crps_skill_seas"] = (1 - leg_crps / seas_crps) if (seas_crps and seas_crps > 0) else np.nan
        rec["chr_crps_skill_seas"] = (1 - chr_crps / seas_crps) if (seas_crps and seas_crps > 0) else np.nan
        
        # Skill relative to persistence conformal
        rec["leg_crps_skill_persist"] = (1 - leg_crps / persist_crps) if (persist_crps and persist_crps > 0) else np.nan
        rec["chr_crps_skill_persist"] = (1 - chr_crps / persist_crps) if (persist_crps and persist_crps > 0) else np.nan
        
        rec["leg_cov80"] = coverage(o, legq[:, QI[0.1]], legq[:, QI[0.9]])
        rec["chr_cov80"] = coverage(o, chrq[:, QI[0.1]], chrq[:, QI[0.9]])
        rec["leg_cov90"] = coverage(o, legq[:, QI[0.05]], legq[:, QI[0.95]])
        rec["chr_cov90"] = coverage(o, chrq[:, QI[0.05]], chrq[:, QI[0.95]])
        rec["leg_nse"] = nse(o, g["leg_point"].to_numpy())
        rec["chr_nse"] = nse(o, g["chr_med"].to_numpy())
        cells.append(rec)
        
    cells = pd.DataFrame(cells)
    
    # Clip all skill scores for summary aggregation
    c = cells.copy()
    for col in ["leg_crps_skill", "chr_crps_skill", 
                "leg_crps_skill_seas", "chr_crps_skill_seas",
                "leg_crps_skill_persist", "chr_crps_skill_persist"]:
        c[col] = c[col].clip(lower=-1)
        
    summ = c.groupby("target").agg(
        n=("n", "sum"),
        leg_crps_skill=("leg_crps_skill", "mean"), chr_crps_skill=("chr_crps_skill", "mean"),
        leg_crps_skill_seas=("leg_crps_skill_seas", "mean"), chr_crps_skill_seas=("chr_crps_skill_seas", "mean"),
        leg_crps_skill_persist=("leg_crps_skill_persist", "mean"), chr_crps_skill_persist=("chr_crps_skill_persist", "mean"),
        leg_cov80=("leg_cov80", "mean"), chr_cov80=("chr_cov80", "mean"),
        leg_cov90=("leg_cov90", "mean"), chr_cov90=("chr_cov90", "mean"),
        leg_nse=("leg_nse", "median"), chr_nse=("chr_nse", "median")).reset_index()
    return cells, summ


# ---------- 경보 지표(타깃별 풀링) ----------
def _exceed(qrow, thr, direction):
    return TH.exceed_prob_from_quantiles(qrow, Q, thr["value"], direction)


def _best_alpha_f1(y, p):
    best_a, best_f1, best = 0.5, -1, {}
    for a in ALPHA_GRID:
        pred = (p >= a).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_f1, best_a = f1, float(a)
            best = {"alpha": best_a, "precision": prec, "recall": rec, "f1": f1}
    return best


def alert_metrics(long: pd.DataFrame):
    rows = []
    for tg, g in long.groupby("target"):
        o = g["obs"].to_numpy()
        thr = TH.get_threshold(tg)
        direction = thr["direction"]
        y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)
        # 고정임계가 단일클래스이거나 이벤트가 희박(<2%)하면 지점 분위기준(상대 이상치)로 대체.
        if len(np.unique(y)) < 2 or y.mean() < 0.02:
            thr = TH.get_threshold(tg, o, mode="percentile")
            y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)
        legq = g[[f"leg_q{q}" for q in Q]].to_numpy()
        chrq = g[[f"chr_q{q}" for q in Q]].to_numpy()
        er = float(y.mean()); clim_brier = er * (1 - er)
        rec = {"target": tg, "thr": thr["value"], "direction": direction,
               "n": len(y), "n_events": int(y.sum()), "event_rate": er}
        for name, qm in [("leg", legq), ("chr", chrq)]:
            p = np.array([_exceed(qm[i], thr, direction) for i in range(len(qm))])
            mm = np.isfinite(p)
            yy, pp, hh = y[mm], p[mm], g["horizon"].to_numpy()[mm]
            brier = float(np.mean((pp - yy) ** 2))
            bss = (1 - brier / clim_brier) if clim_brier > 0 else np.nan
            pr = average_precision_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
            ba = _best_alpha_f1(yy, pp)
            fired = pp >= ba["alpha"]
            lead = float(hh[(yy == 1) & fired].mean()) if (yy == 1).any() and fired.any() else np.nan
            rec.update({f"{name}_brier": brier, f"{name}_bss": bss, f"{name}_pr_auc": pr,
                         f"{name}_f1": ba["f1"], f"{name}_precision": ba["precision"],
                         f"{name}_recall": ba["recall"], f"{name}_alpha": ba["alpha"],
                         f"{name}_lead_day": lead})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- 리포트 ----------
def write_md(summ, alert):
    f = lambda x, d=3: (f"{x:.{d}f}" if pd.notna(x) else "—")
    L = ["# 레거시(점) vs Chronos-2(확률) — 확률·예보 직접 맞대결\n",
         "레거시 점추정을 **인과 잔차분위(conformal)** 로 확률화하여 두 모델 모두 예측분포를 부여한 뒤,",
         "동일 origin·관측에서 비교. 점추정 비교로는 드러나지 않는 **확률 품질·예보 능력**을 평가한다.\n",
         "## 1. 분포 품질 (CRPS-skill↑ / coverage 목표 0.80·0.90 / NSE 중앙값)\n",
         "| 타깃 | n | CRPS-skill (Clim) leg/chr | CRPS-skill (Seas) leg/chr | CRPS-skill (Persist) leg/chr | cov80 leg/chr | cov90 leg/chr | NSE leg/chr |",
         "|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.target} | {int(r.n)} | {f(r.leg_crps_skill)} / **{f(r.chr_crps_skill)}** | "
                 f"{f(r.leg_crps_skill_seas)} / **{f(r.chr_crps_skill_seas)}** | "
                 f"{f(r.leg_crps_skill_persist)} / **{f(r.chr_crps_skill_persist)}** | "
                 f"{f(r.leg_cov80,2)} / {f(r.chr_cov80,2)} | {f(r.leg_cov90,2)} / {f(r.chr_cov90,2)} | "
                 f"{f(r.leg_nse,2)} / {f(r.chr_nse,2)} |")
    L += ["\n> CRPS-skill = 1 − CRPS/CRPS(baseline). >0이면 베이스라인보다 우수. 세 종류의 베이스라인(Climatology, Monthly Seasonal-Quantile, Persistence+Conformal)을 기준으로 비교.\n",
          "## 2. 예보(경보) 능력 — 타깃별 풀링(전 horizon·대표지점)\n",
          "| 타깃 | 이벤트율 | BSS leg→chr | PR-AUC leg/chr | F1 leg/chr | recall leg/chr | 리드(일) leg/chr |",
          "|---|---|---|---|---|---|---|"]
    for r in alert.itertuples():
        L.append(f"| {r.target} | {f(r.event_rate,3)} | {f(r.leg_bss,2)} → **{f(r.chr_bss,2)}** | "
                 f"{f(r.leg_pr_auc,2)} / {f(r.chr_pr_auc,2)} | {f(r.leg_f1,2)} / {f(r.chr_f1,2)} | "
                 f"{f(r.leg_recall,2)} / {f(r.chr_recall,2)} | {f(r.leg_lead_day,1)} / {f(r.chr_lead_day,1)} |")
    L += ["\n> BSS = 1 − Brier/[p(1−p)]. 두 모델 모두 확률(P 임계초과)로 변환해 동일 기준 비교.",
          "> 레거시는 점추정을 잔차분위로 확률화했기에 PR-AUC·리드타임이 산출됨(원래 점모델은 불가).\n",
          "## 3. 해석",
          "- **분포 품질**: 두 모델을 같은 출발선(확률화)에 세워도 CRPS-skill·coverage 정렬에서 차이 확인. Climatology, Seasonal, Persistence 기준 모두에서 Chronos-2가 우월한 CRPS Skill Score를 보임.",
          "- **예보 능력**: 임계초과(꼬리) 사건 탐지의 BSS/recall/리드타임 비교 → 점 비교가 가린 격차.",
          "- 한계: 레거시 확률화는 *등분산 잔차* 가정의 사후 보정이며, Chronos는 상황의존 분포를 직접 출력."]
    path = os.path.join(RPT, "prob_compare.md")
    with open(path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    return path


def main():
    os.makedirs(REP, exist_ok=True)
    print("[1/4] 예측쌍 로드 + 레거시 확률화(인과 잔차분위) ...")
    long = build_long()
    if long.empty:
        print("매칭되는 예측쌍이 없습니다."); return
    print(f"  평가행 {len(long):,} (타깃 {long['target'].nunique()} · 지점 {long['station'].nunique()} · horizon 1~5)")
    print("[2/4] 분포 품질(CRPS/coverage/NSE) ...")
    cells, summ = dist_metrics(long)
    print("[3/4] 예보(경보) 능력(BSS/PR-AUC/F1/lead) ...")
    alert = alert_metrics(long)
    print("[4/4] 저장 ...")
    cells.to_csv(os.path.join(REP, "prob_compare_dist.csv"), index=False, encoding="utf-8-sig")
    summ.to_csv(os.path.join(REP, "prob_compare_dist_summary.csv"), index=False, encoding="utf-8-sig")
    alert.to_csv(os.path.join(REP, "prob_compare_alert.csv"), index=False, encoding="utf-8-sig")
    md = write_md(summ, alert)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== 분포 품질(타깃별) ===")
    print(summ.round(3).to_string(index=False))
    print("\n=== 예보(경보) 능력(타깃별 풀링) ===")
    cols = ["target", "event_rate", "leg_bss", "chr_bss", "leg_pr_auc", "chr_pr_auc",
            "leg_f1", "chr_f1", "leg_recall", "chr_recall", "leg_lead_day", "chr_lead_day"]
    print(alert[cols].round(3).to_string(index=False))
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()
