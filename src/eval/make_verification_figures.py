# -*- coding: utf-8 -*-
"""독립 검증 시각화: prob_compare/robustness_check 산출을 그림(PNG)으로.

기존 자산 불가침. 저장된 CSV와 예측만 사용. 컨테이너 한글폰트 부재로 라벨은 영문.
산출: reports/figures/verify_*.png
실행: python -m src.eval.make_verification_figures
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                            "figure.dpi": 130, "savefig.bbox": "tight"})

from src.data import sources as S
from src.alert import thresholds as TH
from src.eval.prob_compare import build_long, Q
from src.eval.robustness_check import station_boot

REP = os.path.join(S.DEEP_SEE, "reports", "tables")
FIG = os.path.join(S.DEEP_SEE, "reports", "figures")
TGS = ["do", "tn", "toc", "tp", "chl-a"]
DISP = {"do": "DO", "tn": "TN", "toc": "TOC", "tp": "TP", "chl-a": "Chl-a"}
C_LEG, C_CHR = "#c0504d", "#4472c4"     # legacy=red, chronos=blue
os.makedirs(FIG, exist_ok=True)


def _bar_pair(ax, names, leg, chr, ylabel, title, target_line=None):
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, leg, w, label="Legacy (conformalized)", color=C_LEG)
    ax.bar(x + w / 2, chr, w, label="Chronos-2", color=C_CHR)
    if target_line is not None:
        ax.axhline(target_line, color="gray", ls="--", lw=1, label="target/zero")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def fig_crps_skill():
    d = pd.read_csv(os.path.join(REP, "prob_compare_dist_summary.csv")).set_index("target")
    names = [DISP[t] for t in TGS]
    leg = [d.loc[t, "leg_crps_skill"] for t in TGS]; chr = [d.loc[t, "chr_crps_skill"] for t in TGS]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bar_pair(ax, names, leg, chr, "CRPS-skill (vs climatology, higher=better)",
              "Verification A: Distributional skill (both probabilized)", target_line=0)
    p = os.path.join(FIG, "verify_crps_skill.png"); fig.savefig(p); plt.close(fig); return p


def fig_alert_bss():
    d = pd.read_csv(os.path.join(REP, "prob_compare_alert.csv")).set_index("target")
    names = [DISP[t] for t in TGS]
    leg = [d.loc[t, "leg_bss"] for t in TGS]; chr = [d.loc[t, "chr_bss"] for t in TGS]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bar_pair(ax, names, leg, chr, "Alert BSS (vs climatology, >0=useful)",
              "Verification A: Warning skill (Brier Skill Score)", target_line=0)
    p = os.path.join(FIG, "verify_alert_bss.png"); fig.savefig(p); plt.close(fig); return p


def fig_delta_nse_ci():
    """타깃별 집계 ΔNSE(Chr−Leg)와 지점수준 95% CI."""
    bt = pd.read_csv(os.path.join(REP, "robustness_bootstrap.csv"))
    k = bt["station"].nunique() if not bt.empty else 4
    names, means, los, his = [], [], [], []
    for t in TGS:
        vals = bt[bt.target == t]["dNSE"].tolist()
        m, lo, hi = station_boot(vals)
        names.append(DISP[t]); means.append(m); los.append(m - lo); his.append(hi - m)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(names))
    colors = [C_CHR if lo > 0 else "#999999" for lo, m in zip([m - l for m, l in zip(means, los)], means)]
    ax.bar(x, means, 0.6, yerr=[los, his], capsize=5, color=colors,
           error_kw={"ecolor": "#333", "lw": 1.3})
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("ΔNSE  (Chronos − Legacy)")
    ax.set_title(f"Verification B: Aggregate ΔNSE with 95% CI (station bootstrap, K={k})")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.02, "blue = CI excludes 0 (significant) · gray = CI includes 0",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555")
    p = os.path.join(FIG, "verify_delta_nse_ci.png"); fig.savefig(p); plt.close(fig); return p


def fig_dm_significance():
    dm = pd.read_csv(os.path.join(REP, "robustness_dm.csv"))
    names = [DISP[t] for t in TGS]
    pt = [int(dm[dm.target == t]["point_sig5"].sum()) for t in TGS]
    cr = [int(dm[dm.target == t]["crps_sig5"].sum()) for t in TGS]
    max_pts = [int(dm[dm.target == t]["point_sig5"].count()) for t in TGS]
    max_cells = max_pts[0] if max_pts else 4
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, pt, w, label="Point (squared error)", color="#7f7f7f")
    ax.bar(x + w / 2, cr, w, label="Probabilistic (CRPS)", color=C_CHR)
    max_y = max(max(pt), max(cr), 4)
    ax.set_ylim(0, max_y + 0.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(f"# cells significant (of {max_cells})")
    ax.set_title("Verification B: Diebold-Mariano significance (p<0.05, Chronos better)\nHAC=Newey-West + HLN")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    p = os.path.join(FIG, "verify_dm_significance.png"); fig.savefig(p); plt.close(fig); return p


def fig_baseline_audit():
    a = pd.read_csv(os.path.join(REP, "robustness_baseline_audit.csv"))
    a = a.copy()
    a["legacy_nse_obs"] = a["legacy_nse_obs"].clip(lower=-1)     # 발산셀 clip(-1) 표시
    a["cell"] = a["river"].str[:4] + "/" + a["target"]
    a = a.sort_values("chronos_lora_nse").reset_index(drop=True)
    y = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(a["legacy_paper_nse"], y, color="#e6a23c", label="Legacy (paper-reported)", zorder=3)
    ax.scatter(a["legacy_nse_obs"], y, color=C_LEG, label="Legacy (re-impl, clip -1)", zorder=3)
    ax.scatter(a["chronos_lora_nse"], y, color=C_CHR, label="Chronos-2 LoRA512", zorder=3)
    for i, r in a.iterrows():
        ax.plot([r["legacy_nse_obs"], r["chronos_lora_nse"]], [i, i], color="#ccc", lw=1, zorder=1)
    flip = a[a["would_flip"]]
    ax.scatter(flip["legacy_paper_nse"], flip.index, s=160, facecolors="none",
               edgecolors="red", lw=1.8, zorder=4, label="would FLIP if paper-level")
    ax.set_yticks(y); ax.set_yticklabels(a["cell"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("day5 NSE")
    ax.set_title("Verification B: Baseline integrity (paper vs re-impl vs Chronos)\n"
                 "red circles = cells that would flip if legacy matched paper (5/20)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
    p = os.path.join(FIG, "verify_baseline_audit.png"); fig.savefig(p); plt.close(fig); return p


def _exceed_series(long, model):
    """모델별 (p_exceed, y_event) 풀링 — reliability용. 타깃별 임계(분위 폴백 포함)."""
    P, Y = [], []
    for tg, g in long.groupby("target"):
        o = g["obs"].to_numpy(float)
        thr = TH.get_threshold(tg); direction = thr["direction"]
        y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)
        if len(np.unique(y)) < 2 or y.mean() < 0.02:
            thr = TH.get_threshold(tg, o, mode="percentile")
            y = np.array([TH.is_event(v, thr["value"], direction) for v in o], int)
        qm = g[[f"{model}_q{q}" for q in Q]].to_numpy(float)
        p = np.array([TH.exceed_prob_from_quantiles(qm[i], Q, thr["value"], direction)
                      for i in range(len(qm))])
        m = np.isfinite(p)
        P.append(p[m]); Y.append(y[m])
    return np.concatenate(P), np.concatenate(Y)


def _reliability(p, y, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, nb - 1)
    xs, ys, ns = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    ece = sum(n / len(p) * abs(x - yy) for x, yy, n in zip(xs, ys, ns))
    return np.array(xs), np.array(ys), np.array(ns), ece


def fig_reliability(long):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for model, color, name in [("leg", C_LEG, "Legacy (conformalized)"), ("chr", C_CHR, "Chronos-2")]:
        p, y = _exceed_series(long, model)
        xs, ys, ns, ece = _reliability(p, y)
        sizes = 20 + 380 * np.array(ns) / max(ns)
        ax.plot(xs, ys, "-", color=color, lw=1.2)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.7, label=f"{name} (ECE={ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(exceedance)"); ax.set_ylabel("Observed frequency")
    ax.set_title("Verification A: Reliability diagram (pooled, all targets)\nmarker size ∝ #samples")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    p = os.path.join(FIG, "verify_reliability.png"); fig.savefig(p); plt.close(fig); return p


def main():
    print("[fig] CSV 기반 그림 ...")
    outs = [fig_crps_skill(), fig_alert_bss(), fig_delta_nse_ci(),
            fig_dm_significance(), fig_baseline_audit()]
    print("[fig] reliability(예측 재계산) ...")
    long = build_long()
    outs.append(fig_reliability(long))
    for p in outs:
        print("saved:", os.path.relpath(p, S.DEEP_SEE))


if __name__ == "__main__":
    main()
