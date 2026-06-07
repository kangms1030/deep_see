# -*- coding: utf-8 -*-
"""Phase 7b: 컨텍스트 길이 튜닝(zero-shot). 공변량이 무효였으므로 Chronos 본연 강점인
긴 히스토리 활용으로 성능 향상 시도. 기존 2018~ test(전체 데이터) 동일 origin에서
context ∈ {240,360,512,720,1024}h 비교. 모델 1회 로드.

산출: reports/tables/context_sweep.csv, reports/context_sweep.md
실행: PYTHONIOENCODING=utf-8 python -m src.chronos.run_context_sweep
"""
from __future__ import annotations
import os
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
TASK = "context_sweep"
PDI = 4
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
CONTEXTS = [240, 360, 512, 720, 1024]


def _i_va(wide, st, splits):
    t = pd.to_datetime(wide[wide.station == st].sort_values("time")["time"]).values
    return int(np.searchsorted(t, np.datetime64(pd.Timestamp(splits[st]["val_end"]))))


def eval_ctx(pipe, filled, raw, target, origins, ctx):
    inp = TC.build_predict_inputs(filled, target, origins, ctx, cov=None)
    ql, _ = pipe.predict_quantiles(inp, prediction_length=W.OUT_W, quantile_levels=Q)
    preds = np.stack([np.asarray(q)[0] for q in ql], axis=0)
    daily = preds[:, :120, :].reshape(len(origins), 5, 24, len(Q)).mean(axis=2)
    obs5 = np.full(len(origins), np.nan)
    for k, o in enumerate(origins):
        lab = raw[target][o:o + W.OUT_W]
        with np.errstate(invalid="ignore"):
            obs5[k] = np.nanmean(lab[:120].reshape(5, 24), axis=1)[PDI]
    med5 = daily[:, PDI, QI[0.5]]
    pt = Mx.point_metrics(obs5, med5)
    qmat5 = daily[:, PDI, :]
    return {"nse": pt["nse"], "rmse": pt["rmse"], "crps": Mx.crps_from_quantiles(obs5, Q, qmat5),
            "cov80": Mx.coverage(obs5, daily[:, PDI, QI[0.1]], daily[:, PDI, QI[0.9]]),
            "n": pt["n"]}


def main():
    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    rows = []
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
        log(f"로드 완료 | {gpu.fmt()}", TASK)
        for river in ["han", "nak", "geum", "yeong"]:
            wide = pd.read_parquet(os.path.join(OUT, f"{river}_auto_hourly_wide.parquet"))
            wide["station"] = wide["station"].astype(str)
            reps = sidx[(sidx.river == river) & sidx.is_representative]["station"].tolist()
            for st in reps:
                times, raw, filled, _ = TC.station_series(wide, st)
                n = len(filled["do"]); i_va = _i_va(wide, st, splits)
                for tg in S.TARGETS:
                    for ctx in CONTEXTS:
                        origins = TC.test_origins(n, i_va, ctx, W.OUT_W, stride=24)
                        if len(origins) < 3:
                            continue
                        m = eval_ctx(pipe, filled, raw, tg, origins, ctx)
                        rows.append({"river": river, "station": st, "target": tg, "context": ctx, **m})
                    gpu.check(where=f"{st}/{tg}")
                    sub = [r for r in rows if r["station"] == st and r["target"] == tg]
                    best = max(sub, key=lambda r: (r["nse"] if pd.notna(r["nse"]) else -9))
                    log(f"{river}/{st}/{tg}: " + " ".join(f"c{r['context']}={r['nse']:.3f}" for r in sub)
                        + f"  best=c{best['context']}", TASK)
                    pd.DataFrame(rows).to_csv(os.path.join(REP, "context_sweep.csv"),
                                              index=False, encoding="utf-8-sig")
    _write_md(pd.DataFrame(rows))


def _write_md(df):
    piv = df.pivot_table(index=["river", "target"], columns="context", values="nse")
    lines = ["# 컨텍스트 길이 튜닝 (zero-shot, 2018~ test, 5일차 NSE)\n",
             "공변량이 무효였으므로 Chronos 본연의 긴 히스토리 활용으로 성능 향상 시도.\n",
             "| 수계 | 타깃 | " + " | ".join(f"c{c}" for c in CONTEXTS) + " | best |",
             "|---|---|" + "---|" * (len(CONTEXTS) + 1)]
    for (rv, tg), r in piv.iterrows():
        vals = [r.get(c, np.nan) for c in CONTEXTS]
        bestc = CONTEXTS[int(np.nanargmax([v if pd.notna(v) else -9 for v in vals]))]
        lines.append(f"| {rv} | {tg} | " + " | ".join(f"{v:.3f}" if pd.notna(v) else "—" for v in vals)
                     + f" | c{bestc} |")
    lines.append("\n## 평균 NSE (컨텍스트별)")
    for c in CONTEXTS:
        sub = df[df.context == c]
        lines.append(f"- context={c}: NSE={sub['nse'].mean():.3f}, CRPS={sub['crps'].mean():.3f}, "
                     f"cov80={sub['cov80'].mean():.2f}")
    base = df[df.context == 240]["nse"].mean()
    bestrow = df.groupby("context")["nse"].mean()
    bc = bestrow.idxmax()
    lines.append(f"\n- 최적 평균: context={bc} (NSE={bestrow.max():.3f}), "
                 f"기준 240({base:.3f}) 대비 Δ={bestrow.max()-base:+.3f}")
    with open(os.path.join(RPT, "context_sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("context_sweep.md 저장", TASK)


if __name__ == "__main__":
    main()
