# -*- coding: utf-8 -*-
"""Phase 2: 레거시 GAIN+GRU 예측(수계별 대표지점 × 5타깃).

흐름(레거시 main.py 재현):
 wide parquet → 10채널 행렬 → GAIN 보간(train 적합) → z-score(train 통계)
 → 240h→120h 윈도우 → GRU 학습 → test 롤링(stride 24) → 일평균 → 5일차 NSE 등.

평가는 두 가지로 보고: (a) legacy(보간된 라벨 기준, 0.915 등 재현 확인용),
(b) observed-only(실관측만 채점, 공정비교용 — Phase4에서 Chronos와 동일 규약).

실행: PYTHONIOENCODING=utf-8 python -m src.legacy.run_legacy [--river han] [--target do]
"""
from __future__ import annotations
import os
import argparse
import json
import warnings
import random
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slice 등 무해 경고

from src.data import sources as S
from src.legacy.gain import GAINImputer
from src.legacy.gru import GRUForecaster, train_gru, predict
from src.legacy import windows as W
from src.eval import metrics as Mx
from src.utils.progress import log
from src.utils import gpu

OUT = os.path.join(S.DEEP_SEE, "data_processed")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
os.makedirs(REP, exist_ok=True); os.makedirs(PRED, exist_ok=True)
TASK = "legacy"
PREDICT_DAY_IDX = 4   # 5일차(0-base)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_bounds(times: pd.Series, st_split: dict):
    tr = pd.Timestamp(st_split["train_end"]); va = pd.Timestamp(st_split["val_end"])
    t = pd.to_datetime(times).values
    i_tr = int(np.searchsorted(t, np.datetime64(tr)))
    i_va = int(np.searchsorted(t, np.datetime64(va)))
    return i_tr, i_va


def run_one(river, station, target, wide, imp10, splits, epochs, device):
    tag = f"{river}/{station}/{target}"
    sdf = wide[wide["station"] == station].sort_values("time").reset_index(drop=True)
    sdf = W.add_calendar(sdf)
    chans = S.CHANNEL_ORDER                                   # 10 채널
    cal = ["Day_sin", "Day_cos", "Year_sin", "Year_cos"]
    target_idx = chans.index(target)

    raw10 = sdf[chans].to_numpy(np.float64)                   # NaN 포함
    if np.isfinite(raw10[:, target_idx]).sum() < 50:
        log(f"[skip] {tag} 타깃 관측 부족", TASK); return None

    i_tr, i_va = _split_bounds(sdf["time"], splits[station])
    if i_tr < W.IN_W + W.OUT_W:
        log(f"[skip] {tag} train 구간 너무 짧음 ({i_tr})", TASK); return None

    cal_arr = sdf[cal].to_numpy(np.float64)
    feat = np.concatenate([imp10, cal_arr], axis=1)           # [T, 14]
    label_raw = raw10.copy()                                  # 관측-only 채점용(NaN 유지)

    # --- z-score: train 통계 ---
    mu = feat[:i_tr].mean(0); sd = feat[:i_tr].std(0); sd[sd == 0] = 1.0
    featn = (feat - mu) / sd
    # 라벨 정규화(타깃 채널만 필요)
    t_mu, t_sd = mu[target_idx], sd[target_idx]
    labeln_imp = (feat[:, target_idx] - t_mu) / t_sd
    labeln_raw = (label_raw[:, target_idx] - t_mu) / t_sd     # NaN 유지

    # 윈도우 생성(세그먼트 내) — inputs=featn, labels=타깃
    def windows_in(lo, hi, stride):
        f = featn[lo:hi]
        X, Y, orig = W.make_windows(f, target_idx, stride=stride)
        # 관측-only 라벨도 같은 origin으로 추출
        Yr = []
        width = W.IN_W + W.OUT_W
        for i in range(0, f.shape[0] - width + 1, stride):
            Yr.append(labeln_raw[lo + i + W.IN_W: lo + i + width])
        Yr = np.asarray(Yr, np.float32) if Yr else np.empty((0, W.OUT_W), np.float32)
        return X, Y, Yr, (orig + lo if len(orig) else orig)

    Xtr, Ytr, _, _ = windows_in(0, i_tr, stride=12)
    Xva, Yva, _, _ = windows_in(i_tr, i_va, stride=24)
    Xte, Yte_imp, Yte_raw, orig_te = windows_in(i_va, len(featn), stride=24)
    if len(Xtr) < 8 or len(Xte) < 2:
        log(f"[skip] {tag} 윈도우 부족 tr={len(Xtr)} te={len(Xte)}", TASK); return None
    if len(Xva) < 2:
        Xva, Yva = Xtr[-4:], Ytr[-4:]

    log(f"{tag} 윈도우 tr={Xtr.shape} va={Xva.shape} te={Xte.shape}", TASK)
    model = GRUForecaster(n_features=featn.shape[1], out_steps=W.OUT_W)
    model = train_gru(model, Xtr, Ytr, Xva, Yva, epochs=epochs, device=device, task=TASK, tag=tag)

    pred = predict(model, Xte, device=device)                 # [N,120] normalized
    pred = pred * t_sd + t_mu                                 # denorm
    obs_imp = Yte_imp * t_sd + t_mu
    obs_raw = Yte_raw * t_sd + t_mu

    pred_d = W.hour_to_day_mean(pred)                         # [N,5]
    obs_imp_d = W.hour_to_day_mean(obs_imp)
    obs_raw_d = W.hour_to_day_mean(obs_raw, observed=True)    # nanmean

    legacy = Mx.point_metrics(obs_imp_d[:, PREDICT_DAY_IDX], pred_d[:, PREDICT_DAY_IDX])
    obsonly = Mx.point_metrics(obs_raw_d[:, PREDICT_DAY_IDX], pred_d[:, PREDICT_DAY_IDX])

    # 예측 저장(Phase4/5용): 모든 5일치
    pq = pd.DataFrame({"origin_idx": orig_te})
    for d in range(5):
        pq[f"pred_d{d+1}"] = pred_d[:, d]
        pq[f"obs_d{d+1}"] = obs_raw_d[:, d]
    pq["origin_time"] = sdf["time"].iloc[orig_te].values
    pq.to_parquet(os.path.join(PRED, f"legacy_{river}_{station}_{target}.parquet"), index=False)

    row = {"river": river, "station": station, "target": target,
           "nse": legacy["nse"], "rmse": legacy["rmse"], "mae": legacy["mae"],
           "r2": legacy["r2"], "pbias": legacy["pbias"], "n_test": legacy["n"],
           "nse_obs": obsonly["nse"], "rmse_obs": obsonly["rmse"], "n_obs": obsonly["n"]}
    log(f"{tag} ▶ NSE(legacy)={legacy['nse']:.3f} NSE(obs)={obsonly['nse']:.3f} "
        f"RMSE={legacy['rmse']:.3f}", TASK)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--river", default=None, help="han/nak/geum/yeong, 미지정=전체")
    ap.add_argument("--target", default=None, help="do/toc/tn/tp/chl-a, 미지정=전체")
    ap.add_argument("--station", default=None, help="특정 측정소코드(대표지점 무시)")
    ap.add_argument("--scope", default="rep", choices=["rep", "all"], help="rep: 대표지점만, all: 전체 67지점")
    ap.add_argument("--gain-iters", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(42)

    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv"))
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    rivers = [args.river] if args.river else ["han", "nak", "geum", "yeong"]
    targets = [args.target] if args.target else S.TARGETS

    rows = []
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        for river in rivers:
            fp = os.path.join(OUT, f"{river}_auto_hourly_wide.parquet")
            if not os.path.exists(fp):
                log(f"[skip] {river} parquet 없음", TASK); continue
            wide = pd.read_parquet(fp)
            wide["station"] = wide["station"].astype(str)
            if args.station:
                stations = [args.station]
            elif args.scope == "all":
                stations = sidx[sidx["river"] == river]["station"].tolist()
            else:
                stations = sidx[(sidx["river"] == river) & (sidx["is_representative"])]["station"].tolist()
            log(f"=== {river} 지점목록 {stations} (scope={args.scope}) ===", TASK)
            for station in stations:
                sdf = wide[wide["station"] == station].sort_values("time").reset_index(drop=True)
                if sdf.empty:
                    log(f"[skip] {river}/{station} 데이터 비어있음", TASK); continue
                
                chans = S.CHANNEL_ORDER
                raw10 = sdf[chans].to_numpy(np.float64)
                
                if station not in splits:
                    log(f"[skip] {river}/{station} splits 정보 없음", TASK); continue
                i_tr, i_va = _split_bounds(sdf["time"], splits[station])
                if i_tr < W.IN_W + W.OUT_W:
                    log(f"[skip] {river}/{station} train 구간 너무 짧음 ({i_tr})", TASK); continue
                
                # Run GAIN once per station
                try:
                    gain = GAINImputer(dim=len(chans), iterations=args.gain_iters, device=args.device, task=TASK)
                    gain.fit(raw10[:i_tr])
                    imp10 = gain.transform(raw10)
                    imp10 = np.nan_to_num(imp10, nan=0.0)
                except Exception as e:
                    log(f"[ERROR GAIN] {river}/{station}: {type(e).__name__}: {e}", TASK)
                    continue

                for target in targets:
                    try:
                        r = run_one(river, station, target, wide, imp10, splits,
                                    args.epochs, args.device)
                        if r:
                            rows.append(r)
                            pd.DataFrame(rows).to_csv(os.path.join(REP, f"legacy_metrics_{river}.csv"),
                                                      index=False, encoding="utf-8-sig")
                    except Exception as e:
                        log(f"[ERROR GRU] {river}/{station}/{target}: {type(e).__name__}: {e}", TASK)
    log(f"완료. legacy_metrics_{river}.csv 저장 ({len(rows)} 조합)", TASK)
    if rows:
        print(pd.DataFrame(rows)[["river", "station", "target", "nse", "nse_obs", "rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
