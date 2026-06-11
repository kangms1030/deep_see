# -*- coding: utf-8 -*-
"""Phase 7: 수문 공변량 효과 ablation (공변량-가용 구간에서 공정 비교).

배경(직접 확인): 수문 공변량(유량/수위/댐)은 2012~2015/2017만 존재 → 기존 2018~ test와 무겹침.
따라서 **공변량이 존재하는 구간**으로 지점별 윈도우를 재정의하고, 그 안에서 70/10/20 분할 후
**동일 origin에서 ±공변량**을 비교(zero-shot)·**±공변량 LoRA**(튜닝)로 4-way ablation.

조건: zs_nocov / zs_cov / lora_nocov / lora_cov.  동일 윈도우·동일 origin → 순수 공변량/튜닝 효과 분리.
산출: reports/tables/cov_ablation.csv, reports/cov_ablation.md
실행: PYTHONIOENCODING=utf-8 python -m src.chronos.run_cov_ablation [--num-steps 1000]
"""
from __future__ import annotations
import os
import argparse
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.data import sources as S
from src.chronos import to_chronos as TC
from src.legacy import windows as W
from src.eval import metrics as Mx
from src.utils.progress import log
from src.utils import gpu

OUT = os.path.join(S.DEEP_SEE, "data_processed")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
TASK = "cov_ablation"
PREDICT_DAY_IDX = 4
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
CTX = 240


def cov_window(cov_wide, station, times):
    """공변량(유량/수위)이 관측된 구간의 [lo,hi] 인덱스(WQ 시간격자 기준)."""
    cdf = cov_wide[cov_wide["station"].astype(str) == station].copy()
    cdf["time"] = pd.to_datetime(cdf["time"])
    mask = cdf[["cov_flow", "cov_level"]].notna().any(axis=1)
    obs_t = cdf.loc[mask, "time"]
    if len(obs_t) < CTX + W.OUT_W + 50:
        return None
    t = times.values
    lo = int(np.searchsorted(t, np.datetime64(obs_t.min())))
    hi = int(np.searchsorted(t, np.datetime64(obs_t.max())))
    return lo, hi


def daily_from_q(preds):  # preds [N,120,nq] -> [N,5,nq]
    N = preds.shape[0]
    return preds[:, :120, :].reshape(N, 5, 24, len(Q)).mean(axis=2)


def obs_daily(raw_t, origins):
    N = len(origins); od = np.full((N, 5), np.nan)
    for k, o in enumerate(origins):
        lab = raw_t[o:o + W.OUT_W]
        with np.errstate(invalid="ignore"):
            od[k] = np.nanmean(lab[:120].reshape(5, 24), axis=1)
    return od


def metrics_row(daily, obs5):
    med5 = daily[:, PREDICT_DAY_IDX, QI[0.5]]
    pt = Mx.point_metrics(obs5, med5)
    qmat5 = daily[:, PREDICT_DAY_IDX, :]
    return {"nse": pt["nse"], "rmse": pt["rmse"], "crps": Mx.crps_from_quantiles(obs5, Q, qmat5),
            "cov80": Mx.coverage(obs5, daily[:, PREDICT_DAY_IDX, QI[0.1]], daily[:, PREDICT_DAY_IDX, QI[0.9]]),
            "calib": Mx.calibration_error(obs5, Q, qmat5), "n": pt["n"]}


def predict_metrics(pipe, filled, raw, target, origins, cov):
    inp = TC.build_predict_inputs(filled, target, origins, CTX, cov=cov)
    ql, _ = pipe.predict_quantiles(inp, prediction_length=W.OUT_W, quantile_levels=Q)
    preds = np.stack([np.asarray(q)[0] for q in ql], axis=0)
    daily = daily_from_q(preds)
    obs5 = obs_daily(raw[target], origins)[:, PREDICT_DAY_IDX]
    return metrics_row(daily, obs5)


def build_window_split(reps_only, rivers, sidx, all_wide, all_cov):
    """대표지점별 공변량 윈도우 + 윈도우내 70/10/20 인덱스."""
    info = {}
    for river in rivers:
        wide = all_wide[river]; cov_wide = all_cov[river]
        sts = sidx[(sidx.river == river) & (sidx.is_representative)]["station"].tolist() if reps_only \
            else sorted(wide["station"].unique())
        for st in sts:
            times, raw, filled, cov = TC.station_series(wide, st, cov_wide)
            w = cov_window(cov_wide, st, times)
            if w is None:
                continue
            lo, hi = w; L = hi - lo
            tr = lo + int(L * 0.7); va = lo + int(L * 0.8)
            info[st] = {"river": river, "lo": lo, "hi": hi, "train_end": tr, "val_end": va,
                        "times": times, "raw": raw, "filled": filled, "cov": cov}
    return info


def window_origins(va, hi):
    return [o for o in range(va + CTX, hi - W.OUT_W + 1, 24)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--lora-lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", default=None)
    ap.add_argument("--zs-only", action="store_true", help="zero-shot ±cov만(LoRA 생략)")
    ap.add_argument("--out", default="cov_ablation.csv")
    args = ap.parse_args()

    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    rivers = ["han", "nak", "geum", "yeong"]
    targets = [args.target] if args.target else S.TARGETS
    all_wide = {r: pd.read_parquet(os.path.join(OUT, f"{r}_auto_hourly_wide.parquet")) for r in rivers}
    for r in all_wide:
        all_wide[r]["station"] = all_wide[r]["station"].astype(str)
    all_cov = {r: pd.read_parquet(os.path.join(OUT, f"{r}_covariates_hourly.parquet")) for r in rivers}
    for r in all_cov:
        all_cov[r]["station"] = all_cov[r]["station"].astype(str)

    rep_info = build_window_split(True, rivers, sidx, all_wide, all_cov)
    log(f"대표지점 공변량 윈도우 확보: {list(rep_info)}", TASK)

    rows = []
    rows_path = os.path.join(REP, args.out)
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
        log(f"로드 완료 | {gpu.fmt()}", TASK)

        # 전 지점 윈도우(학습용)
        all_info = build_window_split(False, rivers, sidx, all_wide, all_cov)

        for tg in targets:
            # ---- zero-shot ±cov (대표지점) ----
            for st, I in rep_info.items():
                origins = window_origins(I["val_end"], I["hi"])
                if len(origins) < 3:
                    log(f"[skip] {st}/{tg} origins<3", TASK); continue
                m_no = predict_metrics(pipe, I["filled"], I["raw"], tg, origins, cov=None)
                m_cov = predict_metrics(pipe, I["filled"], I["raw"], tg, origins, cov=I["cov"])
                for mode, m in [("zs_nocov", m_no), ("zs_cov", m_cov)]:
                    rows.append({"mode": mode, "river": I["river"], "station": st, "target": tg, **m})
                log(f"{st}/{tg} zs NSE nocov={m_no['nse']:.3f} cov={m_cov['nse']:.3f} "
                    f"(Δ={m_cov['nse']-m_no['nse']:+.3f}) | CRPS {m_no['crps']:.3f}->{m_cov['crps']:.3f}", TASK)
                pd.DataFrame(rows).to_csv(rows_path, index=False, encoding="utf-8-sig")

            # ---- LoRA ±cov (전 지점 윈도우 train으로 적합, 대표지점 평가) ----
            if args.zs_only:
                continue
            for use_cov in (False, True):
                fit_inputs, val_inputs = [], []
                for st, I in all_info.items():
                    covs = [c for c in S.CHANNEL_ORDER if c != tg]
                    def mk(hi):
                        pc = {c: I["filled"][c][I["lo"]:hi].astype(np.float32) for c in covs}
                        if use_cov:
                            for c in I["cov"]:
                                pc[c] = I["cov"][c][I["lo"]:hi].astype(np.float32)
                        return {"target": I["filled"][tg][I["lo"]:hi].astype(np.float32), "past_covariates": pc}
                    if I["train_end"] - I["lo"] > CTX + W.OUT_W:
                        fit_inputs.append(mk(I["train_end"]))
                        val_inputs.append(mk(I["val_end"]))
                tagm = "lora_cov" if use_cov else "lora_nocov"
                log(f"[{tagm}/{tg}] fit items={len(fit_inputs)} steps={args.num_steps}", TASK)
                ft = pipe.fit(fit_inputs, prediction_length=W.OUT_W, validation_inputs=val_inputs or None,
                              finetune_mode="lora", learning_rate=args.lora_lr, num_steps=args.num_steps,
                              batch_size=args.batch_size, context_length=CTX,
                              output_dir=os.path.join(S.DEEP_SEE, "models", f"{tagm}_{tg}"))
                gpu.check(where=tagm)
                for st, I in rep_info.items():
                    origins = window_origins(I["val_end"], I["hi"])
                    if len(origins) < 3:
                        continue
                    cov = I["cov"] if use_cov else None
                    m = predict_metrics(ft, I["filled"], I["raw"], tg, origins, cov=cov)
                    rows.append({"mode": tagm, "river": I["river"], "station": st, "target": tg, **m})
                    log(f"{st}/{tg} {tagm} NSE={m['nse']:.3f} CRPS={m['crps']:.3f}", TASK)
                pd.DataFrame(rows).to_csv(rows_path, index=False, encoding="utf-8-sig")
                del ft
                import torch; torch.cuda.empty_cache()

    log(f"완료. cov_ablation.csv ({len(rows)}행)", TASK)
    _write_md(pd.DataFrame(rows))


def _write_md(df):
    if df.empty:
        return
    piv = df.pivot_table(index=["river", "target"], columns="mode", values="nse")
    order = [c for c in ["zs_nocov", "zs_cov", "lora_nocov", "lora_cov"] if c in piv.columns]
    piv = piv[order]
    lines = ["# 수문 공변량 ablation (공변량-가용 구간, 5일차 NSE)\n",
             "공변량(유량/수위/댐)이 존재하는 구간으로 지점별 윈도우 재정의 후, 동일 origin에서 비교.",
             "수문 공변량은 2012~2015/2017만 존재하여 기존 2018~ test와 무겹침 → 본 ablation은 별도 윈도우.\n",
             "## NSE 비교 (높을수록 우수)\n",
             "| 수계 | 타깃 | zs_nocov | zs_cov | Δzs | lora_nocov | lora_cov | Δlora |",
             "|---|---|---|---|---|---|---|---|"]
    def g(r, c):
        return r[c] if c in r and pd.notna(r[c]) else np.nan
    for (rv, tg), r in piv.iterrows():
        dzs = g(r, "zs_cov") - g(r, "zs_nocov")
        dlo = g(r, "lora_cov") - g(r, "lora_nocov")
        f = lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        lines.append(f"| {rv} | {tg} | {f(g(r,'zs_nocov'))} | {f(g(r,'zs_cov'))} | {f(dzs)} | "
                     f"{f(g(r,'lora_nocov'))} | {f(g(r,'lora_cov'))} | {f(dlo)} |")
    # 요약
    for c in order:
        lines.append("")
        lines.append(f"- 평균 {c} NSE = {df[df['mode']==c]['nse'].mean():.3f}, "
                     f"CRPS = {df[df['mode']==c]['crps'].mean():.3f}, "
                     f"cov80 = {df[df['mode']==c]['cov80'].mean():.2f}")
    with open(os.path.join(RPT, "cov_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("cov_ablation.md 저장", TASK)


if __name__ == "__main__":
    main()
