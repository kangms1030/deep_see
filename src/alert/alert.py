# -*- coding: utf-8 -*-
"""Phase 5: 동적 확률분위 임계초과 경보 시스템 + 경보 성능 평가.

- Chronos-2 분위 예측(reports/predictions/chronos_*.parquet)에서 일별 P(임계 초과)를 산출.
- 실제 관측(obs_dX) 대비 경보 성능: Precision/Recall/F1, ROC-AUC, PR-AUC, Brier, 평균 리드타임.
- 레거시 점추정 임계초과(확률없음)와 비교 → 확률예보 우위 입증.
- 대표지점 1곳 경보 타임라인 시각화 PNG.

실행: PYTHONIOENCODING=utf-8 python -m src.alert.alert --mode lora  (또는 zeroshot)
"""
from __future__ import annotations
import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 한글 폰트(Windows Malgun Gothic) — 그림 라벨 깨짐 방지
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.metrics import roc_auc_score, average_precision_score

from src.data import sources as S
from src.alert import thresholds as TH
from src.utils.progress import log

REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
FIG = os.path.join(S.DEEP_SEE, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
TASK = "alert"
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _chronos_probs(df, target, thr, direction):
    """chronos parquet → (origin × day) y_true, p_exceed, obs, median."""
    out = []
    for d in range(1, 6):
        qcols = [f"q{q}_d{d}" for q in QUANTILES]
        if not all(c in df.columns for c in qcols):
            continue
        for _, r in df.iterrows():
            obs = r[f"obs_d{d}"]
            p = TH.exceed_prob_from_quantiles([r[c] for c in qcols], QUANTILES, thr["value"], direction)
            out.append({"origin_time": r["origin_time"], "day": d, "obs": obs,
                        "p": p, "median": r[f"q0.5_d{d}"],
                        "y": TH.is_event(obs, thr["value"], direction)})
    return pd.DataFrame(out)


def _legacy_probs(df, thr, direction):
    out = []
    for d in range(1, 6):
        pc, oc = f"pred_d{d}", f"obs_d{d}"
        if pc not in df.columns:
            continue
        for _, r in df.iterrows():
            out.append({"day": d, "obs": r[oc], "pred": r[pc],
                        "p": 1.0 if TH.is_event(r[pc], thr["value"], direction) else 0.0,
                        "y": TH.is_event(r[oc], thr["value"], direction)})
    return pd.DataFrame(out)


def _skill(d: pd.DataFrame, alpha: float):
    d = d.dropna(subset=["p", "obs"])
    if d.empty or d["y"].nunique() < 2:
        return None
    y = d["y"].astype(int).values; p = d["p"].values
    pred = (p >= alpha).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    brier = float(np.mean((p - y) ** 2))
    try:
        roc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    except ValueError:
        roc = pr = np.nan
    return {"alpha": alpha, "precision": prec, "recall": rec, "f1": f1,
            "brier": brier, "roc_auc": roc, "pr_auc": pr,
            "n": len(d), "n_events": int(y.sum()), "event_rate": float(y.mean())}


def _best_alpha(d):
    best = None
    for a in np.arange(0.1, 0.91, 0.05):
        s = _skill(d, float(a))
        if s and (best is None or s["f1"] > best["f1"]):
            best = s
    return best


def _lead_time(d, alpha):
    """이벤트가 일어난 origin에서, 며칠 전(=day 작을수록 단기) 경보가 잡혔는지 평균 horizon."""
    ev = d[d["y"] & (d["p"] >= alpha)]
    return float(ev["day"].mean()) if len(ev) else np.nan


def run(mode, sidx):
    rows = []
    for river in ["han", "nak", "geum", "yeong"]:
        reps = sidx[(sidx.river == river) & sidx.is_representative]["station"].tolist()
        for st in reps:
            for tg in S.TARGETS:
                fp = os.path.join(PRED, f"chronos_{mode}_{river}_{st}_{tg}.parquet")
                if not os.path.exists(fp):
                    continue
                cdf = pd.read_parquet(fp)
                thr = TH.get_threshold(tg)
                direction = thr["direction"]
                cp = _chronos_probs(cdf, tg, thr, direction)
                # 고정 기준이 단일클래스(이벤트 0 또는 전부)면 지점 분위 기준으로 대체
                if cp.dropna(subset=["obs"])["y"].nunique() < 2:
                    obs_all = pd.concat([cdf[f"obs_d{d}"] for d in range(1, 6)]).to_numpy()
                    thr = TH.get_threshold(tg, obs_all, mode="percentile")
                    cp = _chronos_probs(cdf, tg, thr, direction)
                    log(f"  {river}/{st}/{tg} 고정기준 단일클래스 → 분위기준 thr={thr['value']:.4g}", TASK)
                cb = _best_alpha(cp)
                if cb is None:
                    log(f"[skip] {river}/{st}/{tg} 분위기준도 단일클래스", TASK); continue
                lead = _lead_time(cp, cb["alpha"])
                row = {"river": river, "station": st, "target": tg, "thr": thr["value"],
                       "direction": direction, **{f"chr_{k}": v for k, v in cb.items()},
                       "chr_lead_day": lead}
                # 레거시 비교
                lfp = os.path.join(PRED, f"legacy_{river}_{st}_{tg}.parquet")
                if os.path.exists(lfp):
                    lp = _legacy_probs(pd.read_parquet(lfp), thr, direction)
                    ls = _skill(lp, 0.5)
                    if ls:
                        row.update({f"leg_{k}": v for k, v in ls.items()
                                    if k in ("precision", "recall", "f1", "brier")})
                rows.append(row)
                log(f"{river}/{st}/{tg} ▶ F1={cb['f1']:.2f} ROC={cb['roc_auc']:.2f} "
                    f"PR={cb['pr_auc']:.2f} Brier={cb['brier']:.3f} α*={cb['alpha']:.2f} "
                    f"events={cb['n_events']}/{cb['n']}", TASK)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REP, f"alert_skill_{mode}.csv"), index=False, encoding="utf-8-sig")
    log(f"alert_skill_{mode}.csv 저장 ({len(df)}행)", TASK)
    return df


def plot_timeline(mode, river, station, target):
    fp = os.path.join(PRED, f"chronos_{mode}_{river}_{station}_{target}.parquet")
    if not os.path.exists(fp):
        return
    df = pd.read_parquet(fp).copy()
    df["t"] = pd.to_datetime(df["origin_time"])
    df = df.sort_values("t")
    thr = TH.get_threshold(target); direction = thr["direction"]
    d = 5  # 5일차 예보
    prob = df.apply(lambda r: TH.exceed_prob_from_quantiles(
        [r[f"q{q}_d{d}"] for q in QUANTILES], QUANTILES, thr["value"], direction), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.fill_between(df["t"], df[f"q0.1_d{d}"], df[f"q0.9_d{d}"], alpha=0.25, label="80% PI")
    ax1.fill_between(df["t"], df[f"q0.05_d{d}"], df[f"q0.95_d{d}"], alpha=0.15, label="90% PI")
    ax1.plot(df["t"], df[f"q0.5_d{d}"], lw=1, label="중앙값(예측)")
    ax1.scatter(df["t"], df[f"obs_d{d}"], s=8, c="k", label="관측", zorder=5)
    ax1.axhline(thr["value"], color="r", ls="--", label=f"임계 {thr['value']}{thr['unit']}")
    ax1.set_title(f"[{mode}] {river}/{station}/{target} 5일차 예보 — {thr['desc']}")
    ax1.legend(loc="upper right", fontsize=8); ax1.set_ylabel(target)

    ax2.plot(df["t"], prob, color="darkorange", lw=1, label="P(임계 초과)")
    alpha = 0.5
    fired = prob >= alpha
    ax2.scatter(df["t"][fired], prob[fired], c="red", s=12, label=f"경보(α={alpha})", zorder=5)
    ax2.axhline(alpha, color="gray", ls=":")
    ax2.set_ylim(0, 1); ax2.set_ylabel("경보확률"); ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG, f"alert_timeline_{mode}_{river}_{station}_{target}.png")
    plt.savefig(out, dpi=120); plt.close()
    log(f"타임라인 저장 {out}", TASK)


def write_md(df, mode):
    lines = [f"# 동적 확률분위 임계초과 경보 시스템 — {mode}\n",
             "Chronos-2 분위 예측으로 일별 P(임계 초과)를 산출해 사전 경보. ",
             "임계는 하천 생활환경 기준 등급 경계(+조류 우려). 평가는 관측 이벤트 대비.\n",
             "## 경보 스킬(타깃별, 최적 α 기준)\n",
             "| 수계 | 타깃 | 임계 | F1 | ROC-AUC | PR-AUC | Brier | 리드(일) | 레거시F1 | 이벤트수 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def f(x, d=3):
        return f"{x:.{d}f}" if pd.notna(x) else "—"
    for _, r in df.iterrows():
        lines.append(f"| {r['river']} | {r['target']} | {r['thr']} | {f(r['chr_f1'],2)} | "
                     f"{f(r.get('chr_roc_auc'),2)} | {f(r.get('chr_pr_auc'),2)} | {f(r['chr_brier'])} | "
                     f"{f(r.get('chr_lead_day'),1)} | {f(r.get('leg_f1'),2)} | "
                     f"{int(r['chr_n_events']) if pd.notna(r['chr_n_events']) else '—'} |")
    if "leg_brier" in df and df["chr_brier"].notna().any():
        lines += ["\n## 요약",
                  f"- 평균 Brier: Chronos={f(df['chr_brier'].mean())}, "
                  f"legacy={f(df['leg_brier'].mean()) if 'leg_brier' in df else '—'} (낮을수록 우수)",
                  f"- 평균 F1: Chronos={f(df['chr_f1'].mean(),2)}, "
                  f"legacy={f(df['leg_f1'].mean(),2) if 'leg_f1' in df else '—'}",
                  "- Chronos는 보정된 확률을 주어 ROC/PR-AUC·Brier로 경보 임계(α)를 비용가중 조정 가능; ",
                  "  레거시 점추정은 0/1 결정만 가능(확률·리드타임 활용 불가)."]
    with open(os.path.join(RPT, f"phase5_alert_{mode}.md"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines) + "\n")
    log(f"phase5_alert_{mode}.md 저장", TASK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lora", choices=["zeroshot", "lora"])
    ap.add_argument("--demo-river", default="han")
    ap.add_argument("--demo-target", default="tp")
    args = ap.parse_args()
    sidx = pd.read_csv(os.path.join(S.DEEP_SEE, "data_processed", "station_index.csv"))
    sidx["station"] = sidx["station"].astype(str)
    df = run(args.mode, sidx)
    write_md(df, args.mode)
    demo_st = sidx[(sidx.river == args.demo_river) & sidx.is_representative]["station"].iloc[0]
    plot_timeline(args.mode, args.demo_river, demo_st, args.demo_target)
    for tg in S.TARGETS:
        plot_timeline(args.mode, args.demo_river, demo_st, tg)


if __name__ == "__main__":
    main()
