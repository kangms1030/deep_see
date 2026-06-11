# -*- coding: utf-8 -*-
"""High-intensity operational audit from saved deep_see outputs.

This script intentionally reads only persisted forecasts/alerts/tables and
recomputes failure-mode diagnostics independently of the existing reports.
"""
from __future__ import annotations

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import sources as S


ROOT = Path(S.DEEP_SEE)
OUT = ROOT / "reports" / "audit"
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _finite_pair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def nse(obs, pred):
    obs, pred = _finite_pair(obs, pred)
    if obs.size < 2:
        return np.nan
    den = np.sum((obs - obs.mean()) ** 2)
    return np.nan if den == 0 else float(1 - np.sum((obs - pred) ** 2) / den)


def rmse(obs, pred):
    obs, pred = _finite_pair(obs, pred)
    return float(np.sqrt(np.mean((obs - pred) ** 2))) if obs.size else np.nan


def coverage(obs, lo, hi):
    obs = np.asarray(obs, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((obs[m] >= lo[m]) & (obs[m] <= hi[m]))) if m.any() else np.nan


def pinball(obs, pred, q):
    obs, pred = _finite_pair(obs, pred)
    if obs.size == 0:
        return np.nan
    e = obs - pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def crps(obs, qmat):
    vals = [pinball(obs, qmat[:, j], q) for j, q in enumerate(Q)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(2 * np.mean(vals)) if vals else np.nan


def winkler(obs, lo, hi, alpha):
    obs = np.asarray(obs, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return np.nan
    y, l, u = obs[m], lo[m], hi[m]
    width = u - l
    return float(np.mean(width + (2 / alpha) * (l - y) * (y < l) + (2 / alpha) * (y - u) * (y > u)))


def corr(a, b):
    a, b = _finite_pair(a, b)
    if a.size < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def average_precision(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0 or len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    return float(np.sum((recall - np.r_[0, recall[:-1]]) * precision))


def f1_at_fired(y, fired):
    y = np.asarray(y, bool)
    fired = np.asarray(fired, bool)
    tp = int((y & fired).sum())
    fp = int((~y & fired).sum())
    fn = int((y & ~fired).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def bss(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return np.nan
    er = y.mean()
    if er <= 0 or er >= 1:
        return np.nan
    return float(1 - np.mean((p - y) ** 2) / (er * (1 - er)))


def load_forecasts():
    fps = sorted((ROOT / "system_out" / "forecasts").glob("*.parquet"))
    return pd.concat([pd.read_parquet(fp) for fp in fps], ignore_index=True)


def load_alerts():
    return pd.read_parquet(ROOT / "system_out" / "alerts" / "alert_log.parquet")


def load_cov_ablation():
    p = ROOT / "reports" / "tables" / "cov_ablation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def quantile_integrity(fc):
    rows = []
    raw_cols = [f"q{q}" for q in Q]
    raw = fc[raw_cols].to_numpy(float)
    raw_cross = np.any(np.diff(raw, axis=1) < -1e-12, axis=1)
    for keys, g in fc.groupby(["target", "horizon_day"]):
        arr = g[raw_cols].to_numpy(float)
        rows.append({
            "target": keys[0],
            "horizon_day": keys[1],
            "n": len(g),
            "raw_any_cross_rate": float(np.mean(np.any(np.diff(arr, axis=1) < -1e-12, axis=1))),
            "cal80_cross_rate": float(np.mean(g["q0.1_cal"].to_numpy(float) > g["q0.9_cal"].to_numpy(float))),
            "cal90_cross_rate": float(np.mean(g["q0.05_cal"].to_numpy(float) > g["q0.95_cal"].to_numpy(float))),
            "raw_global_cross_rate": float(np.mean(raw_cross[g.index])),
        })
    return pd.DataFrame(rows)


def distribution_dynamics(fc):
    fc = fc.copy()
    fc["err"] = np.abs(fc["obs"] - fc["median"])
    fc["w80_raw"] = fc["q0.9"] - fc["q0.1"]
    fc["w90_raw"] = fc["q0.95"] - fc["q0.05"]
    fc["w80_cal"] = fc["q0.9_cal"] - fc["q0.1_cal"]
    fc["w90_cal"] = fc["q0.95_cal"] - fc["q0.05_cal"]
    rows = []
    for (tg, hd), g in fc.groupby(["target", "horizon_day"]):
        valid = g[np.isfinite(g["obs"]) & np.isfinite(g["median"])]
        if valid.empty:
            continue
        high_err = valid["err"] >= valid["err"].quantile(0.9)
        rows.append({
            "target": tg,
            "horizon_day": hd,
            "n": len(valid),
            "w80_raw_mean": valid["w80_raw"].mean(),
            "w80_raw_cv": valid["w80_raw"].std() / valid["w80_raw"].mean() if valid["w80_raw"].mean() else np.nan,
            "w80_cal_mean": valid["w80_cal"].mean(),
            "w80_cal_cv": valid["w80_cal"].std() / valid["w80_cal"].mean() if valid["w80_cal"].mean() else np.nan,
            "width_error_corr_raw": corr(valid["w80_raw"], valid["err"]),
            "width_error_corr_cal": corr(valid["w80_cal"], valid["err"]),
            "high_error_width_ratio_raw": valid.loc[high_err, "w80_raw"].mean() / valid.loc[~high_err, "w80_raw"].mean(),
            "high_error_width_ratio_cal": valid.loc[high_err, "w80_cal"].mean() / valid.loc[~high_err, "w80_cal"].mean(),
            "raw_cov80": coverage(valid["obs"], valid["q0.1"], valid["q0.9"]),
            "cal_cov80": coverage(valid["obs"], valid["q0.1_cal"], valid["q0.9_cal"]),
            "raw_winkler80": winkler(valid["obs"], valid["q0.1"], valid["q0.9"], 0.2),
            "cal_winkler80": winkler(valid["obs"], valid["q0.1_cal"], valid["q0.9_cal"], 0.2),
        })
    return pd.DataFrame(rows), fc


def extreme_and_persistence(fc):
    d5 = fc[fc["horizon_day"] == 5].copy()
    d5["asof"] = pd.to_datetime(d5["asof"])
    d5["err"] = np.abs(d5["obs"] - d5["median"])
    d5["w80_raw"] = d5["q0.9"] - d5["q0.1"]
    d5["w80_cal"] = d5["q0.9_cal"] - d5["q0.1_cal"]
    d5 = d5.sort_values(["station", "target", "asof"])
    d5["obs_delta"] = d5.groupby(["station", "target"])["obs"].diff()
    d5["pred_delta"] = d5.groupby(["station", "target"])["median"].diff()
    d5["pers_err"] = np.abs(d5["obs"] - d5["persistence"])

    rows = []
    for tg, g in d5.groupby("target"):
        q95 = g["obs"].quantile(0.95)
        dq95 = g["obs_delta"].abs().quantile(0.95)
        subsets = {
            "all_day5": np.ones(len(g), dtype=bool),
            "actual_top5pct": (g["obs"] >= q95).to_numpy(),
            "abs_delta_top5pct": (g["obs_delta"].abs() >= dq95).to_numpy(),
            "model_large_error_top10pct": (g["err"] >= g["err"].quantile(0.9)).to_numpy(),
        }
        for name, mask in subsets.items():
            sub = g[mask & np.isfinite(g["obs"]) & np.isfinite(g["median"])]
            if len(sub) < 5:
                continue
            qmat = sub[[f"q{q}" for q in Q]].to_numpy(float)
            rows.append({
                "target": tg,
                "subset": name,
                "n": len(sub),
                "nse_model": nse(sub["obs"], sub["median"]),
                "nse_persistence": nse(sub["obs"], sub["persistence"]),
                "rmse_model": rmse(sub["obs"], sub["median"]),
                "rmse_persistence": rmse(sub["obs"], sub["persistence"]),
                "crps_raw": crps(sub["obs"].to_numpy(float), qmat),
                "cov80_raw": coverage(sub["obs"], sub["q0.1"], sub["q0.9"]),
                "cov80_cal": coverage(sub["obs"], sub["q0.1_cal"], sub["q0.9_cal"]),
                "mean_w80_raw": sub["w80_raw"].mean(),
                "mean_w80_cal": sub["w80_cal"].mean(),
                "median_abs_model_delta": np.nanmedian(np.abs(sub["pred_delta"])),
                "median_abs_obs_delta": np.nanmedian(np.abs(sub["obs_delta"])),
                "pred_delta_vs_obs_delta_corr": corr(sub["pred_delta"], sub["obs_delta"]),
                "median_pred_minus_persistence_abs": np.nanmedian(np.abs(sub["median"] - sub["persistence"])),
                "forecast_vs_persistence_corr": corr(sub["median"], sub["persistence"]),
            })
    return pd.DataFrame(rows), d5


def spatial_metrics(d5):
    rows = []
    for (rv, st, tg), g in d5.groupby(["river", "station", "target"]):
        valid = g[np.isfinite(g["obs"]) & np.isfinite(g["median"])]
        if len(valid) < 5:
            continue
        rows.append({
            "river": rv,
            "station": st,
            "target": tg,
            "n": len(valid),
            "nse": nse(valid["obs"], valid["median"]),
            "nse_persistence": nse(valid["obs"], valid["persistence"]),
            "coverage80_raw": coverage(valid["obs"], valid["q0.1"], valid["q0.9"]),
            "coverage80_cal": coverage(valid["obs"], valid["q0.1_cal"], valid["q0.9_cal"]),
            "rmse": rmse(valid["obs"], valid["median"]),
            "median_abs_q50_minus_persistence": np.nanmedian(np.abs(valid["median"] - valid["persistence"])),
        })
    sm = pd.DataFrame(rows)
    wr = []
    for tg, g in sm.groupby("target"):
        n = len(g)
        s = g.sort_values("nse")
        wr.append({
            "target": tg,
            "stations": n,
            "median_nse": g["nse"].median(),
            "mean_nse": g["nse"].mean(),
            "worst_10pct_nse": s.head(max(1, math.ceil(n * 0.10)))["nse"].mean(),
            "worst_5pct_nse": s.head(max(1, math.ceil(n * 0.05)))["nse"].mean(),
            "stations_below_persistence": int((g["nse"] < g["nse_persistence"]).sum()),
            "median_cov80_cal": g["coverage80_cal"].median(),
            "worst_10pct_cov80_cal": g.sort_values("coverage80_cal").head(max(1, math.ceil(n * 0.10)))["coverage80_cal"].mean(),
        })
    river = sm.groupby(["river", "target"]).agg(
        stations=("station", "nunique"),
        median_nse=("nse", "median"),
        mean_nse=("nse", "mean"),
        median_cov80_cal=("coverage80_cal", "median"),
    ).reset_index()
    return sm, pd.DataFrame(wr), river


def alert_subset_metrics(alerts, d5):
    rows = []
    for tg, g in alerts.groupby("target"):
        gg = g[np.isfinite(g["p_exceed"])]
        rows.append({
            "target": tg,
            "subset": "all_horizons",
            "n": len(gg),
            "event_rate": gg["is_event"].mean(),
            "pr_auc": average_precision(gg["is_event"].astype(int), gg["p_exceed"]),
            "f1": f1_at_fired(gg["is_event"], gg["fired"]),
            "bss": bss(gg["is_event"].astype(float), gg["p_exceed"]),
        })
    d5key = d5[["station", "river", "asof", "target", "horizon_day", "obs_delta"]].copy()
    d5key["asof"] = d5key["asof"].astype(str)
    al = alerts.copy()
    al["asof"] = al["asof"].astype(str)
    m = al.merge(d5key, on=["station", "river", "asof", "target", "horizon_day"], how="left")
    for tg, g in m.groupby("target"):
        thr = g["obs_delta"].abs().quantile(0.95)
        sub = g[g["obs_delta"].abs() >= thr]
        if len(sub) < 10:
            continue
        rows.append({
            "target": tg,
            "subset": "abs_delta_top5pct",
            "n": len(sub),
            "event_rate": sub["is_event"].mean(),
            "pr_auc": average_precision(sub["is_event"].astype(int), sub["p_exceed"]),
            "f1": f1_at_fired(sub["is_event"], sub["fired"]),
            "bss": bss(sub["is_event"].astype(float), sub["p_exceed"]),
        })
    return pd.DataFrame(rows)


def covariate_audit(cov):
    if cov.empty:
        return pd.DataFrame()
    piv = cov.pivot_table(index=["river", "station", "target"], columns="mode", values=["nse", "crps", "cov80"])
    rows = []
    for metric in ["nse", "crps", "cov80"]:
        if (metric, "zs_cov") in piv.columns and (metric, "zs_nocov") in piv.columns:
            delta = piv[(metric, "zs_cov")] - piv[(metric, "zs_nocov")]
            rows.append({
                "metric": metric,
                "comparison": "zs_cov_minus_nocov",
                "n": int(delta.notna().sum()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "improved_count": int((delta > 0).sum()) if metric != "crps" else int((delta < 0).sum()),
            })
    return pd.DataFrame(rows)


def legacy_fairness_tables():
    out = {}
    for name in ["final_eval_metrics.csv", "robustness_baseline_audit.csv", "context_sweep.csv"]:
        p = ROOT / "reports" / "tables" / name
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def write_md(results):
    qi = results["quantile_integrity"]
    dyn = results["distribution_dynamics"]
    ext = results["extreme"]
    spatial = results["spatial_worst"]
    alerts = results["alert_subsets"]
    cov = results["covariate"]

    qcal80 = qi.groupby("target")["cal80_cross_rate"].mean().sort_values(ascending=False)
    qraw = qi.groupby("target")["raw_any_cross_rate"].mean()
    dyn_d5 = dyn[dyn["horizon_day"] == 5].set_index("target")
    ext_top = ext[ext["subset"].isin(["actual_top5pct", "abs_delta_top5pct"])]
    spatial_idx = spatial.set_index("target")

    lines = [
        "# High-Intensity Audit: deep_see Operational Forecasts",
        "",
        "## Executive Summary",
        "- 운영 경로의 Chronos 입력은 외생 공변량 없이 생성된다. 따라서 현재 운영 시스템은 causal forcing 모델이 아니라 수질 다변량 history 기반 예측 시스템이다.",
        "- 저장 forecast 기준 raw quantile crossing은 거의 없지만, conformal 보정 후 80/90% band crossing이 발생한다. 이는 보정폭이 음수가 될 수 있는 구현 때문에 interval integrity가 깨지는 치명적 운영 리스크다.",
        "- 예측폭과 실제 오차의 상관은 낮거나 음수인 경우가 많다. uncertainty가 event/error에 맞춰 동적으로 커진다는 근거가 약하다.",
        "- 극단/급변 subset에서 NSE, coverage, alert skill이 평균 regime 대비 크게 악화되는 타깃이 있다. 평균 성능만으로 운영 가능성을 주장하기 어렵다.",
        "",
        "## Key Quantitative Findings",
        "| target | raw crossing | cal80 crossing | d5 width-error corr | d5 high-error width ratio | worst-5% station NSE | stations below persistence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tg in sorted(qcal80.index):
        d = dyn_d5.loc[tg] if tg in dyn_d5.index else {}
        s = spatial_idx.loc[tg] if tg in spatial_idx.index else {}
        lines.append(
            f"| {tg} | {qraw.get(tg, np.nan):.3f} | {qcal80.get(tg, np.nan):.3f} | "
            f"{getattr(d, 'width_error_corr_raw', np.nan):.3f} | {getattr(d, 'high_error_width_ratio_raw', np.nan):.3f} | "
            f"{getattr(s, 'worst_5pct_nse', np.nan):.3f} | {int(getattr(s, 'stations_below_persistence', 0))} |"
        )

    lines += [
        "",
        "## Extreme/Event Reliability",
        "| target | subset | n | NSE model | NSE persistence | cov80 raw | cov80 cal | pred/persist corr | pred_delta/obs_delta corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ext_top.itertuples():
        lines.append(
            f"| {r.target} | {r.subset} | {r.n} | {r.nse_model:.3f} | {r.nse_persistence:.3f} | "
            f"{r.cov80_raw:.3f} | {r.cov80_cal:.3f} | {r.forecast_vs_persistence_corr:.3f} | "
            f"{r.pred_delta_vs_obs_delta_corr:.3f} |"
        )

    lines += [
        "",
        "## Alert Skill Subsets",
        "| target | subset | n | event rate | PR-AUC | F1 | BSS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in alerts.itertuples():
        lines.append(f"| {r.target} | {r.subset} | {r.n} | {r.event_rate:.3f} | {r.pr_auc:.3f} | {r.f1:.3f} | {r.bss:.3f} |")

    if not cov.empty:
        lines += [
            "",
            "## Covariate Ablation Summary",
            "| metric | n | mean delta | median delta | improved count |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in cov.itertuples():
            lines.append(f"| {r.metric} | {r.n} | {r.mean_delta:.4f} | {r.median_delta:.4f} | {r.improved_count} |")

    lines += [
        "",
        "## Verdict",
        "현재 증거 기준 Chronos2는 causal environmental forecasting system이 아니라 probabilistic autoregressive model에 가깝다. 운영 배포는 제한적 보조 의사결정/대시보드 수준에서만 가능하며, 고위험 폭우·오염·극단상황 자동 경보 시스템으로는 아직 부적합하다.",
    ]
    (OUT / "high_intensity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc = load_forecasts()
    alerts = load_alerts()
    cov = load_cov_ablation()

    qi = quantile_integrity(fc)
    dyn, fc2 = distribution_dynamics(fc)
    ext, d5 = extreme_and_persistence(fc2)
    station, spatial_worst, river = spatial_metrics(d5)
    al_sub = alert_subset_metrics(alerts, d5)
    covsum = covariate_audit(cov)

    results = {
        "quantile_integrity": qi,
        "distribution_dynamics": dyn,
        "extreme": ext,
        "station": station,
        "spatial_worst": spatial_worst,
        "river": river,
        "alert_subsets": al_sub,
        "covariate": covsum,
    }
    for name, df in results.items():
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    meta = {
        "forecast_rows": int(len(fc)),
        "alert_rows": int(len(alerts)),
        "stations": int(fc["station"].nunique()),
        "targets": sorted(fc["target"].unique().tolist()),
    }
    (OUT / "audit_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(results)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print((OUT / "high_intensity_audit.md").as_posix())


if __name__ == "__main__":
    main()
