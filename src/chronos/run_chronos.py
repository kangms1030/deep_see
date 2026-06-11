# -*- coding: utf-8 -*-
"""Phase 3: Chronos-2 zero-shot + LoRA 파인튜닝 평가.

- 레거시와 동일 origin(240h→120h, stride24, 5일차 일평균)에서 평가 → 직접 비교.
- 점추정(중앙값 q0.5)으로 NSE/RMSE/..., 분위로 CRPS/coverage/calibration.
- 전 5일치 분위를 reports/predictions/chronos_*.parquet 로 저장(Phase5 경보용).
- LoRA: pipe.fit(finetune_mode='lora') 네이티브 API 사용(타깃별 전 지점 글로벌 어댑터).

실행: PYTHONIOENCODING=utf-8 python -m src.chronos.run_chronos --mode both [--river han] [--target do]
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
PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
MODELS = os.path.join(S.DEEP_SEE, "models")
for d in (REP, PRED, MODELS):
    os.makedirs(d, exist_ok=True)
TASK = "chronos"
PREDICT_DAY_IDX = 4
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QI = {q: i for i, q in enumerate(QUANTILES)}


def _i_va(wide, station, splits):
    t = pd.to_datetime(wide[wide.station == station].sort_values("time")["time"]).values
    return int(np.searchsorted(t, np.datetime64(pd.Timestamp(splits[station]["val_end"]))))


def eval_station(pipe, wide, river, station, target, splits, context_length, mode_tag,
                 cov_wide=None, weather_wide=None):
    tag = f"{river}/{station}/{target}"
    times, raw, filled, cov = TC.station_series(wide, station, cov_wide, weather_wide=weather_wide)
    n = len(filled[target])
    i_va = _i_va(wide, station, splits)
    origins = TC.test_origins(n, i_va, context_length, W.OUT_W, stride=24)
    if len(origins) < 2:
        log(f"[skip] {tag} origins 부족", TASK); return None
    inputs = TC.build_predict_inputs(filled, target, origins, context_length, cov=cov)

    ql, _ = pipe.predict_quantiles(inputs, prediction_length=W.OUT_W, quantile_levels=QUANTILES)
    # ql[i]: [1,120,nq] → preds [N,120,nq]
    preds = np.stack([np.asarray(q)[0] for q in ql], axis=0)         # [N,120,nq]
    gpu.check(where=f"chronos predict {tag}")

    # 일평균: [N,5,nq]
    N = preds.shape[0]
    daily = preds[:, :120, :].reshape(N, 5, 24, len(QUANTILES)).mean(axis=2)

    # 관측 일평균(raw, observed-only) — 레거시와 동일 방식
    raw_t = raw[target]
    obs_daily = np.full((N, 5), np.nan)
    for k, o in enumerate(origins):
        lab = raw_t[o:o + W.OUT_W]
        with np.errstate(invalid="ignore"):
            obs_daily[k] = np.nanmean(lab[:120].reshape(5, 24), axis=1)

    med = daily[:, :, QI[0.5]]
    obs5 = obs_daily[:, PREDICT_DAY_IDX]
    med5 = med[:, PREDICT_DAY_IDX]
    pt = Mx.point_metrics(obs5, med5)

    # 확률지표(5일차)
    qmat5 = daily[:, PREDICT_DAY_IDX, :]                              # [N,nq]
    crps = Mx.crps_from_quantiles(obs5, QUANTILES, qmat5)
    cov80 = Mx.coverage(obs5, daily[:, PREDICT_DAY_IDX, QI[0.1]], daily[:, PREDICT_DAY_IDX, QI[0.9]])
    cov90 = Mx.coverage(obs5, daily[:, PREDICT_DAY_IDX, QI[0.05]], daily[:, PREDICT_DAY_IDX, QI[0.95]])
    calib = Mx.calibration_error(obs5, QUANTILES, qmat5)

    # 예측 저장(전 5일치 분위 + 관측)
    rows = {"origin_idx": origins, "origin_time": times.iloc[origins].values}
    for d in range(5):
        rows[f"obs_d{d+1}"] = obs_daily[:, d]
        for q in QUANTILES:
            rows[f"q{q}_d{d+1}"] = daily[:, d, QI[q]]
    pd.DataFrame(rows).to_parquet(
        os.path.join(PRED, f"chronos_{mode_tag}_{river}_{station}_{target}.parquet"), index=False)

    row = {"mode": mode_tag, "river": river, "station": station, "target": target,
           "nse": pt["nse"], "rmse": pt["rmse"], "mae": pt["mae"], "r2": pt["r2"],
           "pbias": pt["pbias"], "n_test": pt["n"],
           "crps": crps, "cov80": cov80, "cov90": cov90, "calib_err": calib}
    log(f"{tag} [{mode_tag}] ▶ NSE={pt['nse']:.3f} RMSE={pt['rmse']:.3f} "
        f"CRPS={crps:.3f} cov80={cov80:.2f} cov90={cov90:.2f}", TASK)
    return row


def _load_cov(river, use_cov):
    if not use_cov:
        return None
    fp = os.path.join(OUT, f"{river}_covariates_hourly.parquet")
    if not os.path.exists(fp):
        return None
    cw = pd.read_parquet(fp); cw["station"] = cw["station"].astype(str)
    return cw


def _load_weather(river, use_weather):
    if not use_weather:
        return None
    fp = os.path.join(OUT, f"{river}_weather_hourly.parquet")
    if not os.path.exists(fp):
        log(f"[WARN] 기상 파일 없음: {fp}", TASK)
        return None
    ww = pd.read_parquet(fp); ww["station"] = ww["station"].astype(str)
    return ww


def run_zeroshot(pipe, rivers, targets, sidx, splits, context_length, rows, use_cov, use_weather, suffix):
    for river in rivers:
        wide = pd.read_parquet(os.path.join(OUT, f"{river}_auto_hourly_wide.parquet"))
        wide["station"] = wide["station"].astype(str)
        cov_wide = _load_cov(river, use_cov)
        weather_wide = _load_weather(river, use_weather)
        reps = sidx[(sidx.river == river) & (sidx.is_representative)]["station"].tolist()
        for st in reps:
            for tg in targets:
                r = eval_station(pipe, wide, river, st, tg, splits, context_length,
                                 "zeroshot" + suffix, cov_wide=cov_wide,
                                 weather_wide=weather_wide)
                if r:
                    rows.append(r); _save(rows)


def run_lora(base_pipe, rivers, targets, sidx, splits, context_length, num_steps,
             lora_lr, batch_size, rows, use_cov, use_weather, suffix):
    """타깃별 글로벌 LoRA: 전 수계 train 시계열(+공변량)로 1개 어댑터 학습 → 대표지점 평가."""
    all_wide = {r: pd.read_parquet(os.path.join(OUT, f"{r}_auto_hourly_wide.parquet"))
                for r in ["han", "nak", "geum", "yeong"]}
    for r in all_wide:
        all_wide[r]["station"] = all_wide[r]["station"].astype(str)
    all_cov = {r: _load_cov(r, use_cov) for r in all_wide}
    all_weather = {r: _load_weather(r, use_weather) for r in all_wide}

    for tg in targets:
        fit_inputs, val_inputs = [], []
        for r, wide in all_wide.items():
            sts = sorted(wide["station"].unique())
            fi, vi = TC.build_finetune_inputs(wide, sts, tg, splits, cov_wide=all_cov[r],
                                               weather_wide=all_weather[r])
            fit_inputs += fi; val_inputs += vi
        log(f"[LoRA{suffix}/{tg}] fit items={len(fit_inputs)} val={len(val_inputs)} "
            f"steps={num_steps} bs={batch_size}", TASK)
        outdir = os.path.join(MODELS, f"chronos_lora{suffix}_{tg}")
        ft = base_pipe.fit(
            fit_inputs, prediction_length=W.OUT_W, validation_inputs=val_inputs or None,
            finetune_mode="lora", learning_rate=lora_lr, num_steps=num_steps,
            batch_size=batch_size, output_dir=outdir, context_length=context_length)
        gpu.check(where=f"after LoRA fit {tg}")
        log(f"[LoRA{suffix}/{tg}] 학습 완료 | {gpu.fmt()}", TASK)
        for river in rivers:
            wide = all_wide[river]
            reps = sidx[(sidx.river == river) & (sidx.is_representative)]["station"].tolist()
            for st in reps:
                row = eval_station(ft, wide, river, st, tg, splits, context_length,
                                   "lora" + suffix, cov_wide=all_cov[river],
                                   weather_wide=all_weather[river])
                if row:
                    rows.append(row); _save(rows)
        del ft
        import torch; torch.cuda.empty_cache()


_ROWS_PATH = os.path.join(REP, "chronos_metrics.csv")


def _save(rows):
    pd.DataFrame(rows).to_csv(_ROWS_PATH, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["zeroshot", "lora", "both"])
    ap.add_argument("--river", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--context", type=int, default=240)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--lora-lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-cov", action="store_true", help="수문 공변량 결합")
    ap.add_argument("--use-weather", action="store_true", help="기상 공변량 결합 (ASOS 기온/강수/풍속/습도/일사)")
    ap.add_argument("--out", default=None, help="결과 csv 경로 override")
    args = ap.parse_args()

    global _ROWS_PATH
    parts = []
    if args.use_cov:     parts.append("cov")
    if args.use_weather: parts.append("wx")
    suffix = ("_" + "_".join(parts)) if parts else ""
    if args.out:
        _ROWS_PATH = os.path.join(REP, args.out)
    elif parts:
        _ROWS_PATH = os.path.join(REP, f"chronos_metrics{'_' + '_'.join(parts)}.csv")

    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    rivers = [args.river] if args.river else ["han", "nak", "geum", "yeong"]
    targets = [args.target] if args.target else S.TARGETS

    rows = []
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        log(f"Chronos-2 로드... mode={args.mode} context={args.context}", TASK)
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
        log(f"로드 완료 | use_cov={args.use_cov} use_weather={args.use_weather} "
            f"out={_ROWS_PATH} | {gpu.fmt()}", TASK)
        if args.mode in ("zeroshot", "both"):
            run_zeroshot(pipe, rivers, targets, sidx, splits, args.context, rows,
                         args.use_cov, args.use_weather, suffix)
        if args.mode in ("lora", "both"):
            run_lora(pipe, rivers, targets, sidx, splits, args.context,
                     args.num_steps, args.lora_lr, args.batch_size, rows,
                     args.use_cov, args.use_weather, suffix)
    log(f"완료. chronos_metrics.csv ({len(rows)}행)", TASK)
    if rows:
        df = pd.DataFrame(rows)
        print(df[["mode", "river", "target", "nse", "rmse", "crps", "cov80"]].to_string(index=False))


if __name__ == "__main__":
    main()
