# -*- coding: utf-8 -*-
"""최종 헤드투헤드: 파인튜닝(LoRA@512) Chronos-2 vs 레거시 GAIN+GRU.

- 레거시와 **동일 롤링 origin**(과거 240h 위치 기준, stride 24, 5일차 일평균)에서
  Chronos LoRA@512(컨텍스트 512h)로 5개 타깃 예측 → 동일 관측 기준 비교.
- 레거시 예측은 Phase 2 산출(reports/predictions/legacy_*.parquet) 재사용.
- 산출: reports/tables/final_compare.csv, reports/final_compare.md,
        reports/predictions/final_chr512_{river}_{st}_{tg}.parquet,
        reports/figures/final_{river}_{st}.png (지점별 5타깃 패널)

실행: PYTHONIOENCODING=utf-8 python -m src.eval.final_compare
"""
from __future__ import annotations
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.rcParams["font.family"] = _f; break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

from src.data import sources as S
from src.chronos import to_chronos as TC
from src.legacy import windows as W
from src.eval import metrics as Mx
from src.utils.progress import log
from src.utils import gpu

OUT = os.path.join(S.DEEP_SEE, "data_processed")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
FIG = os.path.join(S.DEEP_SEE, "reports", "figures")
MODELS = os.path.join(S.DEEP_SEE, "models")
TASK = "final_compare"
CTX = 512
PDI = 4
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
RIVERS = ["han", "nak", "geum", "yeong"]
TGT_KR = {"do": "DO(용존산소)", "toc": "TOC", "tn": "T-N(총질소)", "tp": "T-P(총인)", "chl-a": "Chl-a"}


def _i_va(times, st, splits):
    return int(np.searchsorted(times.values, np.datetime64(pd.Timestamp(splits[st]["val_end"]))))


def predict_chronos(pipe, filled, target, origins):
    inp = TC.build_predict_inputs(filled, target, origins, CTX, cov=None)
    ql, _ = pipe.predict_quantiles(inp, prediction_length=W.OUT_W, quantile_levels=Q)
    preds = np.stack([np.asarray(q)[0] for q in ql], axis=0)            # [N,120,nq]
    return preds[:, :120, :].reshape(len(origins), 5, 24, len(Q)).mean(axis=2)  # [N,5,nq]


def obs_daily(raw_t, origins):
    od = np.full((len(origins), 5), np.nan)
    for k, o in enumerate(origins):
        lab = raw_t[o:o + W.OUT_W]
        with np.errstate(invalid="ignore"):
            od[k] = np.nanmean(lab[:120].reshape(5, 24), axis=1)
    return od


def main():
    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    reps = {r: sidx[(sidx.river == r) & sidx.is_representative]["station"].tolist() for r in RIVERS}

    rows = []
    # 지점별 예측 저장(시각화용): {(river,st): {target: dict}}
    store = {}
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        for tg in S.TARGETS:
            ckpt = os.path.join(MODELS, f"chronos_lora512_{tg}", "finetuned-ckpt")
            pipe = Chronos2Pipeline.from_pretrained(ckpt, device_map="cuda")
            log(f"[{tg}] LoRA@512 어댑터 로드 | {gpu.fmt()}", TASK)
            for river in RIVERS:
                wide = pd.read_parquet(os.path.join(OUT, f"{river}_auto_hourly_wide.parquet"))
                wide["station"] = wide["station"].astype(str)
                for st in reps[river]:
                    times, raw, filled, _ = TC.station_series(wide, st)
                    n = len(filled[tg]); i_va = _i_va(times, st, splits)
                    origins = TC.test_origins(n, i_va, 240, W.OUT_W, stride=24)  # 레거시와 동일 origin
                    if len(origins) < 3:
                        continue
                    daily = predict_chronos(pipe, filled, tg, origins)            # [N,5,nq]
                    od = obs_daily(raw[tg], origins)
                    ot = times.iloc[origins].values
                    obs5 = od[:, PDI]; med5 = daily[:, PDI, QI[0.5]]
                    chr_pt = Mx.point_metrics(obs5, med5)
                    crps = Mx.crps_from_quantiles(obs5, Q, daily[:, PDI, :])
                    cov80 = Mx.coverage(obs5, daily[:, PDI, QI[0.1]], daily[:, PDI, QI[0.9]])

                    # 레거시 예측 로드(동일 origin_time으로 정렬)
                    lfp = os.path.join(PRED, f"legacy_{river}_{st}_{tg}.parquet")
                    leg_pred5 = np.full(len(origins), np.nan)
                    if os.path.exists(lfp):
                        lg = pd.read_parquet(lfp)
                        lg["origin_time"] = pd.to_datetime(lg["origin_time"])
                        m = dict(zip(lg["origin_time"], lg["pred_d5"]))
                        leg_pred5 = np.array([m.get(pd.Timestamp(t), np.nan) for t in ot])
                    leg_pt = Mx.point_metrics(obs5, leg_pred5)

                    # 저장(parquet) — 동일 origin에서 obs/legacy/chronos(5일차 + 분위)
                    df = pd.DataFrame({"origin_time": ot, "obs_d5": obs5,
                                       "legacy_pred_d5": leg_pred5, "chr_median_d5": med5})
                    for q in Q:
                        df[f"chr_q{q}_d5"] = daily[:, PDI, QI[q]]
                    df.to_parquet(os.path.join(PRED, f"final_chr512_{river}_{st}_{tg}.parquet"), index=False)
                    store.setdefault((river, st), {})[tg] = {
                        "ot": ot, "obs": obs5, "leg": leg_pred5, "med": med5,
                        "lo": daily[:, PDI, QI[0.1]], "hi": daily[:, PDI, QI[0.9]],
                        "chr_nse": chr_pt["nse"], "leg_nse": leg_pt["nse"]}

                    rows.append({"river": river, "station": st, "target": tg,
                                 "legacy_nse": leg_pt["nse"], "legacy_rmse": leg_pt["rmse"],
                                 "chr512_nse": chr_pt["nse"], "chr512_rmse": chr_pt["rmse"],
                                 "chr512_crps": crps, "chr512_cov80": cov80,
                                 "delta_nse": chr_pt["nse"] - leg_pt["nse"], "n": chr_pt["n"]})
                    log(f"{river}/{st}/{tg}: legacy NSE={leg_pt['nse']:.3f} | "
                        f"chronosLoRA512 NSE={chr_pt['nse']:.3f} (Δ={chr_pt['nse']-leg_pt['nse']:+.3f})", TASK)
            del pipe
            import torch; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(os.path.join(REP, "final_compare.csv"), index=False, encoding="utf-8-sig")

    _plots(store)
    _md(pd.DataFrame(rows))


def _plots(store):
    for (river, st), tgs in store.items():
        fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
        for ax, tg in zip(axes, S.TARGETS):
            if tg not in tgs:
                ax.set_visible(False); continue
            d = tgs[tg]; t = pd.to_datetime(d["ot"])
            ax.fill_between(t, d["lo"], d["hi"], color="tab:blue", alpha=0.2, label="Chronos 80% PI")
            ax.plot(t, d["med"], color="tab:blue", lw=1.2, label=f"Chronos-LoRA512 (NSE={d['chr_nse']:.2f})")
            ax.plot(t, d["leg"], color="tab:orange", lw=1.0, alpha=0.9,
                    label=f"레거시 GAIN+GRU (NSE={d['leg_nse']:.2f})")
            ax.scatter(t, d["obs"], s=10, c="k", zorder=5, label="관측")
            ax.set_ylabel(TGT_KR.get(tg, tg)); ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.2)
        axes[0].set_title(f"[{river}] {st} — 5일후(일평균) 롤링 예측: 레거시 vs 파인튜닝 Chronos-2(LoRA@512) vs 관측")
        plt.tight_layout()
        out = os.path.join(FIG, f"final_{river}_{st}.png")
        plt.savefig(out, dpi=110); plt.close()
        log(f"그림 저장 {out}", TASK)


def _md(df):
    f = lambda x, d=3: f"{x:.{d}f}" if pd.notna(x) else "—"
    L = ["# 최종 비교: 레거시 GAIN+GRU vs 파인튜닝 Chronos-2 (LoRA@512)\n",
         "동일 롤링 origin(과거 240h 위치·stride 24·5일차 일평균)·동일 관측 기준. "
         "Chronos는 컨텍스트 512h + LoRA(rank16) 파인튜닝 모델.\n",
         "| 수계 | 타깃 | 레거시 NSE | Chronos-LoRA512 NSE | ΔNSE | 레거시 RMSE | Chronos RMSE | Chronos CRPS | cov80 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['river']} | {r['target']} | {f(r['legacy_nse'])} | {f(r['chr512_nse'])} | "
                 f"**{f(r['delta_nse'])}** | {f(r['legacy_rmse'])} | {f(r['chr512_rmse'])} | "
                 f"{f(r['chr512_crps'])} | {f(r['chr512_cov80'],2)} |")
    leg = df["legacy_nse"].clip(lower=-1); chr_ = df["chr512_nse"]
    win = int((df["chr512_nse"] > df["legacy_nse"]).sum())
    L += ["\n## 요약",
          f"- Chronos-LoRA512가 레거시 우위: **{win}/{len(df)}** 조합",
          f"- 평균 NSE(레거시 −1클립) = {f(leg.mean())}, Chronos-LoRA512 = **{f(chr_.mean())}**",
          f"- 중앙값 NSE: 레거시 {f(df['legacy_nse'].median())}, Chronos-LoRA512 **{f(chr_.median())}**",
          f"- Chronos 평균 CRPS={f(df['chr512_crps'].mean())}, cov80={f(df['chr512_cov80'].mean(),2)}(목표0.8) — 확률·보정은 레거시(점추정) 불가",
          "\n시각화: `reports/figures/final_{river}_{station}.png` (지점별 5타깃 패널)."]
    with open(os.path.join(RPT, "final_compare.md"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    log("final_compare.md 저장", TASK)
    print("\n".join(L))


if __name__ == "__main__":
    main()
