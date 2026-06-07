# -*- coding: utf-8 -*-
"""백테스트/replay 오케스트레이터(검증의 중심).

보유 데이터의 test 구간을 스트림처럼 재생하여 운영 파이프라인(예보→보정→경보)을
인과적으로 구동·검증한다. 산출: system_out/forecasts/{river}_{station}.parquet,
system_out/alerts/alert_log.parquet.

타깃 외부 루프(어댑터 1개씩 로드 → VRAM 캡 준수), 지점 내부 루프. 행은 메모리 누적 후
지점별로 묶어 저장. α(발령임계)는 val 구간에서 비용가중 최적화(누수 없음).

실행: PYTHONIOENCODING=utf-8 python -m src.system.run_system replay [--scope all|rep]
                                  [--stations S01001,...] [--targets do,tp] [--smoke]
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from src.system import config as C
from src.system import context as CX
from src.system import forecast as FC
from src.system import conformal as CF
from src.system import alerting as AL
from src.system import registry as REG
from src.system import schemas as SC
from src.eval import metrics as Mx
from src.utils.progress import track, log
from src.utils import gpu

TASK = "sys_replay"
BANDS = {80: (0.1, 0.9, 0.2), 90: (0.05, 0.95, 0.1)}   # band%: (lo_q, hi_q, alpha_cqr)


def _pooled_deltas(val_packs, di):
    """타깃·horizon(di)별 풀링 보정폭(여러 지점 val 잔차 합산)."""
    out = {}
    for band, (lq, hq, a) in BANDS.items():
        lo = np.concatenate([p["qv"][:, di, C.QI[lq]] for p in val_packs]) if val_packs else np.array([])
        hi = np.concatenate([p["qv"][:, di, C.QI[hq]] for p in val_packs]) if val_packs else np.array([])
        y = np.concatenate([p["obs"][:, di] for p in val_packs]) if val_packs else np.array([])
        out[band] = CF.pooled_delta_from(lo, hi, y, a)
    return out


def _station_packs(pipe, wide, stations, tg, splits, ctx=C.CTX, min_test=8, desc="predict"):
    """지점별 val/test 예측·관측·메타 묶음 생성(ctx=타깃 추론 컨텍스트)."""
    packs = []
    for st in track(stations, desc=desc, task=TASK):
        if st not in splits:
            continue
        sd = CX.StationData(wide, st)
        if np.isfinite(sd.raw[tg]).sum() < 50:
            continue
        i_tr, i_va = CX.split_indices(sd.times, splits[st])
        test_o = sd.test_origins(i_va, ctx=ctx)
        val_o = sd.val_origins(i_tr, i_va, ctx=ctx)
        if len(test_o) < min_test:
            continue
        qv = FC.predict_daily_quantiles(pipe, sd.filled, tg, val_o, ctx=ctx) if val_o else \
            np.empty((0, C.HORIZON_DAYS, len(C.Q)))
        qt = FC.predict_daily_quantiles(pipe, sd.filled, tg, test_o, ctx=ctx)
        packs.append({"sd": sd, "i_tr": i_tr, "i_va": i_va, "val_o": val_o, "test_o": test_o,
                      "qv": qv, "qt": qt,
                      "obs": sd.obs_daily(tg, val_o) if val_o else np.empty((0, C.HORIZON_DAYS)),
                      "obs_t": sd.obs_daily(tg, test_o),
                      "ot_t": sd.origin_times(test_o), "ot_v": sd.origin_times(val_o),
                      "pers_t": sd.persistence(tg, test_o),
                      "clim_t": sd.climatology(tg, test_o, i_tr)})
    return packs


def _resolve_threshold(tg, packs):
    """고정 기준이 test+val 관측에서 단일클래스면 지점-공통 분위기준으로 대체."""
    thr = C.threshold(tg)
    obs_all = np.concatenate([p["obs_t"].ravel() for p in packs] +
                             [p["obs"].ravel() for p in packs if p["obs"].size])
    yev = np.array([AL.TH.is_event(v, thr["value"], thr["direction"])
                    for v in obs_all if np.isfinite(v)])
    if yev.size and len(np.unique(yev)) < 2:
        thr = C.threshold(tg, obs_all, mode="percentile")
        log(f"  {tg}: 고정기준 단일클래스 → 분위기준 thr={thr['value']:.4g}", TASK)
    return thr


def _val_alpha(packs, pooled, thr, tg):
    """val 구간 보정확률로 비용가중 α* 산출(누수 없음)."""
    P, Y = [], []
    for p in packs:
        if not len(p["val_o"]):
            continue
        for di in range(C.HORIZON_DAYS):
            d = di + 1
            cal = {}
            for band, (lq, hq, a) in BANDS.items():
                lo, hi, _ = CF.calibrate_series(p["ot_v"], p["qv"][:, di, C.QI[lq]],
                                                p["qv"][:, di, C.QI[hq]], p["obs"][:, di],
                                                label_lag_h=d * 24, alpha=a,
                                                pooled_delta=pooled[di][band])
                cal[f"q{lq}_cal"], cal[f"q{hq}_cal"] = lo, hi
            for k in range(len(p["val_o"])):
                raw_q = {q: p["qv"][k, di, C.QI[q]] for q in C.Q}
                curve = AL.calibrated_curve(raw_q, {kk: vv[k] for kk, vv in cal.items()})
                P.append(AL.exceed_prob(curve, thr["value"], thr["direction"]))
                Y.append(AL.TH.is_event(p["obs"][k, di], thr["value"], thr["direction"]))
    return AL.cost_optimal_alpha(np.array(P), np.array(Y, float), tg)


def run(scope="all", stations=None, targets=None, smoke=False, device="cuda"):
    C.ensure_dirs()
    sidx = C.load_station_index()
    splits = json.load(open(os.path.join(C.DATA_PROC, "splits.json"), encoding="utf-8"))
    wides = {r: pd.read_parquet(os.path.join(C.DATA_PROC, f"{r}_auto_hourly_wide.parquet"))
             for r in C.RIVERS}
    for r in wides:
        wides[r]["station"] = wides[r]["station"].astype(str)
    name_of = dict(zip(sidx["station"], sidx.get("name", sidx["station"])))

    want = C.station_list(sidx, scope)
    if stations:
        sset = set(stations)
        want = [(rv, st) for rv, st in want if st in sset]
    if smoke:
        want = want[:1]
    tgs = targets or C.TARGETS
    by_river = {r: [st for rv, st in want if rv == r] for r in C.RIVERS}
    reg = REG.load()

    fc_rows, al_rows = [], []
    with gpu.VramMonitor(log_path=os.path.join(C.LOGS, f"{TASK}_vram.log")):
        for tg in tgs:
            adapter = REG.active_adapter(tg, reg)
            ctx = REG.active_context(tg, reg)
            pipe = FC.load_pipeline(adapter, device)
            log(f"[{tg}] 어댑터 로드 {os.path.relpath(adapter, C.DEEP_SEE)} ctx={ctx} | {gpu.fmt()}", TASK)
            packs_all = []
            for r in C.RIVERS:
                sts = by_river[r]
                if not sts:
                    continue
                packs = _station_packs(pipe, wides[r], sts, tg, splits, ctx=ctx, desc=f"{tg}/{r} predict")
                for p in packs:
                    p["river"] = r
                packs_all += packs
            if not packs_all:
                del pipe; _free(); continue
            pooled = {di: _pooled_deltas(packs_all, di) for di in range(C.HORIZON_DAYS)}
            thr = _resolve_threshold(tg, packs_all)
            alpha = _val_alpha(packs_all, pooled, thr, tg) if not smoke else C.ALPHA_DEFAULT
            log(f"[{tg}] pooledδ(d5)={ {b: round(pooled[4][b],3) if np.isfinite(pooled[4][b]) else None for b in pooled[4]} } α*={alpha} thr={thr['value']:.4g}({thr['direction']})", TASK)

            for p in track(packs_all, desc=f"{tg} replay", task=TASK):
                _emit_station(p, tg, thr, pooled, alpha, name_of, fc_rows, al_rows)
            gpu.check(where=f"replay {tg}")
            del pipe; _free()

    _persist(fc_rows, al_rows)
    return len(fc_rows), len(al_rows)


def _emit_station(p, tg, thr, pooled, alpha, name_of, fc_rows, al_rows):
    sd, st, river = p["sd"], p["sd"].station, p["river"]
    qt, obs_t, ot_t = p["qt"], p["obs_t"], p["ot_t"]
    N = len(p["test_o"])
    # horizon별 보정밴드(인과)
    cal_by_d = {}
    for di in range(C.HORIZON_DAYS):
        d = di + 1; cal = {}
        for band, (lq, hq, a) in BANDS.items():
            lo, hi, _ = CF.calibrate_series(ot_t, qt[:, di, C.QI[lq]], qt[:, di, C.QI[hq]],
                                            obs_t[:, di], label_lag_h=d * 24, alpha=a,
                                            pooled_delta=pooled[di][band])
            cal[f"q{lq}_cal"], cal[f"q{hq}_cal"] = lo, hi
        cal_by_d[d] = cal
    # 가드레일: 모델 day5 NSE < persistence day5 NSE → 저신뢰
    med5 = qt[:, 4, C.QI[0.5]]
    low_conf = bool(np.nan_to_num(Mx.nse(obs_t[:, 4], med5), nan=-9) <
                    np.nan_to_num(Mx.nse(obs_t[:, 4], p["pers_t"]), nan=-9))
    for k in range(N):
        asof = pd.Timestamp(ot_t[k]).isoformat()
        p_by_day = {}
        rows_k = []
        for di in range(C.HORIZON_DAYS):
            d = di + 1
            raw_q = {q: float(qt[k, di, C.QI[q]]) for q in C.Q}
            cal = {kk: float(vv[k]) for kk, vv in cal_by_d[d].items()}
            curve = AL.calibrated_curve(raw_q, cal)
            pe = AL.exceed_prob(curve, thr["value"], thr["direction"])
            p_by_day[d] = pe
            fc_rows.append(SC.forecast_row(
                st, river, asof, tg, d, raw_q, cal, raw_q[0.5],
                float(p["pers_t"][k]), float(p["clim_t"][k]), float(obs_t[k, di])))
            rows_k.append((d, pe, float(obs_t[k, di])))
        fired, lead, max_lv = AL.origin_summary(p_by_day, alpha)
        for d, pe, ov in rows_k:
            al_rows.append(SC.alert_row(
                st, river, asof, tg, d, pe, C.alert_level(pe), thr["value"], thr["direction"],
                AL.TH.is_event(ov, thr["value"], thr["direction"]), alpha,
                (np.isfinite(pe) and pe >= alpha), (None if not np.isfinite(lead) else int(lead)),
                max_lv, low_conf))


def _persist(fc_rows, al_rows):
    """부분 타깃 재실행 시 기존 저장소를 보존하도록 타깃 단위 머지(덮어쓰기 아님)."""
    fc = pd.DataFrame(fc_rows); al = pd.DataFrame(al_rows)
    if len(fc):
        tgs = set(fc["target"].unique())
        for (rv, st), g in fc.groupby(["river", "station"]):
            fp = os.path.join(C.FORECASTS, f"{rv}_{st}.parquet")
            if os.path.exists(fp):
                old = pd.read_parquet(fp)
                old = old[~old["target"].isin(tgs)]      # 이번에 계산한 타깃만 교체
                g = pd.concat([old, g], ignore_index=True)
            g.to_parquet(fp, index=False)
    if len(al):
        tgs = set(al["target"].unique())
        ap = os.path.join(C.ALERTS, "alert_log.parquet")
        if os.path.exists(ap):
            old = pd.read_parquet(ap)
            old = old[~old["target"].isin(tgs)]
            al = pd.concat([old, al], ignore_index=True)
        al.to_parquet(ap, index=False)
    log(f"저장 완료: forecasts {fc['station'].nunique() if len(fc) else 0}지점 / "
        f"alert_log 갱신({len(fc['target'].unique()) if len(fc) else 0}개 타깃 머지)", TASK)


def _free():
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
