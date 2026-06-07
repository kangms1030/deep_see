# -*- coding: utf-8 -*-
"""Phase 9: context=512 통합 실험 (기존 결과와 완전 독립, 새 파일로만 저장).

(a) zero-shot@512 — 전 67지점×5타깃 일괄 평가(학습 없음) → ctx512_zeroshot_all.csv
(b) LoRA@512 (rank·steps 상향 = 하이퍼파라미터 튜닝) — 전 지점 train으로 어댑터 학습,
    대표 4지점 평가 → ctx512_lora_reps.csv (동일 origin에서 zs512 vs lora512 직접 비교)

용어: '컨텍스트=512'는 하이퍼파라미터(추론 입력 길이). 'LoRA'는 파인튜닝(어댑터 가중치 학습).
기존 산출물 미변경. 산출: reports/tables/ctx512_*.csv, reports/phase9_ctx512.md, models/chronos_lora512_*/

실행: PYTHONIOENCODING=utf-8 python -m src.chronos.run_ctx512 [--num-steps 1500 --rank 16]
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
from src.utils.progress import track, log
from src.utils import gpu

OUT = os.path.join(S.DEEP_SEE, "data_processed")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
MODELS = os.path.join(S.DEEP_SEE, "models")
TASK = "ctx512"
CTX = 512
PDI = 4
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(Q)}
RIVERS = ["han", "nak", "geum", "yeong"]


def _i_va(times, st, splits):
    t = times.values
    return int(np.searchsorted(t, np.datetime64(pd.Timestamp(splits[st]["val_end"]))))


def eval_metrics(pipe, filled, raw, target, origins):
    inp = TC.build_predict_inputs(filled, target, origins, CTX, cov=None)
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
    return {"nse": pt["nse"], "rmse": pt["rmse"], "mae": pt["mae"],
            "crps": Mx.crps_from_quantiles(obs5, Q, qmat5),
            "cov80": Mx.coverage(obs5, daily[:, PDI, QI[0.1]], daily[:, PDI, QI[0.9]]),
            "cov90": Mx.coverage(obs5, daily[:, PDI, QI[0.05]], daily[:, PDI, QI[0.95]]),
            "calib": Mx.calibration_error(obs5, Q, qmat5), "n": pt["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=1500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    all_wide = {r: pd.read_parquet(os.path.join(OUT, f"{r}_auto_hourly_wide.parquet")) for r in RIVERS}
    for r in all_wide:
        all_wide[r]["station"] = all_wide[r]["station"].astype(str)
    reps = {r: sidx[(sidx.river == r) & sidx.is_representative]["station"].tolist() for r in RIVERS}

    zs_path = os.path.join(REP, "ctx512_zeroshot_all.csv")
    lora_path = os.path.join(REP, "ctx512_lora_reps.csv")
    zs_rows, lora_rows = [], []

    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
        log(f"로드 완료 | {gpu.fmt()}", TASK)

        # ===== (a) zero-shot @512 — 전 67지점 =====
        # 지점별 시리즈 캐시(LoRA 평가에서 재사용)
        ser = {}
        for river in RIVERS:
            wide = all_wide[river]
            stations = sorted(wide["station"].unique())
            for st in track(stations, desc=f"{river} zs@512", task=TASK):
                times, raw, filled, _ = TC.station_series(wide, st)
                ser[st] = (times, raw, filled, river)
                n = len(filled["do"]); i_va = _i_va(times, st, splits)
                origins = TC.test_origins(n, i_va, CTX, W.OUT_W, stride=24)
                if len(origins) < 3:
                    continue
                for tg in S.TARGETS:
                    if np.isfinite(raw[tg]).sum() < 50:
                        continue
                    m = eval_metrics(pipe, filled, raw, tg, origins)
                    zs_rows.append({"mode": "zs512", "river": river, "station": st, "target": tg, **m})
                gpu.check(where=f"zs {st}")
                pd.DataFrame(zs_rows).to_csv(zs_path, index=False, encoding="utf-8-sig")
        log(f"(a) zero-shot@512 완료: {len(zs_rows)}행 → {zs_path}", TASK)

        # ===== (b) LoRA @512 (rank/steps 상향) — 대표지점 평가 =====
        lora_cfg = {"r": args.rank, "lora_alpha": 2 * args.rank, "lora_dropout": 0.05}
        log(f"LoRA@512 config={lora_cfg} steps={args.num_steps} lr={args.lora_lr} bs={args.batch_size}", TASK)
        for tg in S.TARGETS:
            fit_inputs, val_inputs = [], []
            for river in RIVERS:
                sts = sorted(all_wide[river]["station"].unique())
                fi, vi = TC.build_finetune_inputs(all_wide[river], sts, tg, splits, cov_wide=None)
                fit_inputs += fi; val_inputs += vi
            log(f"[LoRA512/{tg}] fit items={len(fit_inputs)} val={len(val_inputs)}", TASK)
            ft = pipe.fit(fit_inputs, prediction_length=W.OUT_W, validation_inputs=val_inputs or None,
                          finetune_mode="lora", lora_config=lora_cfg, learning_rate=args.lora_lr,
                          num_steps=args.num_steps, batch_size=args.batch_size, context_length=CTX,
                          output_dir=os.path.join(MODELS, f"chronos_lora512_{tg}"))
            gpu.check(where=f"LoRA512 fit {tg}")
            log(f"[LoRA512/{tg}] 학습 완료 | {gpu.fmt()}", TASK)
            for river in RIVERS:
                for st in reps[river]:
                    times, raw, filled, _ = ser[st]
                    n = len(filled["do"]); i_va = _i_va(times, st, splits)
                    origins = TC.test_origins(n, i_va, CTX, W.OUT_W, stride=24)
                    if len(origins) < 3:
                        continue
                    m = eval_metrics(ft, filled, raw, tg, origins)
                    lora_rows.append({"mode": "lora512", "river": river, "station": st, "target": tg, **m})
                    log(f"{river}/{st}/{tg} lora512 NSE={m['nse']:.3f} CRPS={m['crps']:.3f}", TASK)
            pd.DataFrame(lora_rows).to_csv(lora_path, index=False, encoding="utf-8-sig")
            del ft
            import torch; torch.cuda.empty_cache()

    _write_md(pd.DataFrame(zs_rows), pd.DataFrame(lora_rows), sidx, args)


def _write_md(zs, lora, sidx, args):
    rep_st = set(sidx[sidx.is_representative]["station"])
    zs_rep = zs[zs["station"].isin(rep_st)].set_index(["river", "target"])
    lo = lora.set_index(["river", "target"]) if len(lora) else pd.DataFrame()
    L = ["# Phase 9: context=512 통합 (zero-shot 전지점 + LoRA@512 튜닝)\n",
         "기존 결과(240 baseline 등)와 독립. 용어: context=512는 하이퍼파라미터(추론 길이), LoRA는 파인튜닝.\n",
         f"LoRA 설정: rank={args.rank}, alpha={2*args.rank}, steps={args.num_steps}, lr={args.lora_lr}, ctx=512.\n",
         "## A. zero-shot@512 — 전 67지점 일반화 요약\n",
         f"- 평가 조합 {len(zs)}개 (지점×타깃). 평균 NSE={zs['nse'].mean():.3f}, median={zs['nse'].median():.3f}, "
         f"CRPS={zs['crps'].mean():.3f}, cov80={zs['cov80'].mean():.2f}, cov90={zs['cov90'].mean():.2f}",
         "- 타깃별 평균 NSE(전지점):"]
    for tg in S.TARGETS:
        g = zs[zs.target == tg]
        L.append(f"  - {tg}: NSE={g['nse'].mean():.3f} (n지점={g['station'].nunique()})")
    L += ["\n## B. 대표지점 — zero-shot@512 vs LoRA@512 직접 비교 (5일차 NSE)\n",
          "| 수계 | 타깃 | zs@512 | lora@512 | Δ(lora-zs) | zs CRPS | lora CRPS |",
          "|---|---|---|---|---|---|---|"]
    f = lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    dlist = []
    for rv in RIVERS:
        for tg in S.TARGETS:
            z = zs_rep["nse"].get((rv, tg), np.nan)
            l = lo["nse"].get((rv, tg), np.nan) if len(lo) else np.nan
            zc = zs_rep["crps"].get((rv, tg), np.nan)
            lc = lo["crps"].get((rv, tg), np.nan) if len(lo) else np.nan
            if pd.notna(z) and pd.notna(l):
                dlist.append(l - z)
            L.append(f"| {rv} | {tg} | {f(z)} | {f(l)} | {f(l-z) if pd.notna(z) and pd.notna(l) else '—'} | {f(zc)} | {f(lc)} |")
    if dlist:
        L += ["\n## 결론",
              f"- 대표지점 평균 Δ(LoRA@512 − zeroshot@512) NSE = **{np.mean(dlist):+.4f}** "
              f"(개선 {sum(1 for d in dlist if d>0)}/{len(dlist)})",
              f"- 평균 NSE: zs@512(대표)={zs_rep['nse'].reindex([(r,t) for r in RIVERS for t in S.TARGETS]).mean():.3f}, "
              f"lora@512={lora['nse'].mean():.3f}" if len(lora) else "",
              "- 해석: Δ가 0 근처면 본 데이터에서 **파인튜닝(LoRA)은 부가가치 미미**, 성능은 **컨텍스트 길이(하이퍼파라미터)**가 좌우."]
    with open(os.path.join(RPT, "phase9_ctx512.md"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join([x for x in L if x is not None]) + "\n")
    log("phase9_ctx512.md 저장", TASK)


if __name__ == "__main__":
    main()
