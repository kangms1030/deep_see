# -*- coding: utf-8 -*-
"""최종 종합 평가 재산출: 5개 모델 사다리 + 정규화지표 + conformal 보정.

모델: persistence(지속성), climatology(계절-naive 월기후값), legacy(GAIN+GRU),
      chronos_zs512(zero-shot, ctx512), chronos_lora512(파인튜닝, ctx512).
지표: NSE, RSR(=RMSE/σ), PBIAS, R²(점추정) / CRPS, CRPS_skill(vs기후값),
      coverage80·90 (RAW vs CONFORMAL=CQR, 검증구간으로 보정).
동일 롤링 origin(과거 240h 위치·stride24·5일차 일평균)·관측-only.
산출: reports/tables/final_eval_metrics.csv
실행: PYTHONIOENCODING=utf-8 python -m src.eval.final_eval
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
from src.data import sources as S
from src.chronos import to_chronos as TC
from src.legacy import windows as W
from src.utils.progress import log
from src.utils import gpu

OUT = os.path.join(S.DEEP_SEE, "data_processed")
REP = os.path.join(S.DEEP_SEE, "reports", "tables")
PRED = os.path.join(S.DEEP_SEE, "reports", "predictions")
MODELS = os.path.join(S.DEEP_SEE, "models")
TASK = "final_eval"
CTX = 512; PDI = 4
Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]; QI = {q: i for i, q in enumerate(Q)}
RIVERS = ["han", "nak", "geum", "yeong"]


def _idx(times, ts):
    return int(np.searchsorted(times.values, np.datetime64(pd.Timestamp(ts))))

def nse(o, s):
    o, s = _v(o, s); d = ((o - o.mean()) ** 2).sum()
    return 1 - ((o - s) ** 2).sum() / d if d > 0 and len(o) > 2 else np.nan

def rsr(o, s):
    o, s = _v(o, s); sd = o.std()
    return np.sqrt(((o - s) ** 2).mean()) / sd if sd > 0 and len(o) else np.nan

def pbias(o, s):
    o, s = _v(o, s)
    return (o - s).sum() / o.sum() * 100 if len(o) and o.sum() != 0 else np.nan

def r2(o, s):
    o, s = _v(o, s)
    return np.corrcoef(o, s)[0, 1] ** 2 if len(o) > 2 else np.nan

def _v(o, s):
    o = np.asarray(o, float); s = np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s); return o[m], s[m]

def crps_q(o, qmat):
    m = np.isfinite(o) & np.isfinite(qmat).all(1); o = o[m]; qmat = qmat[m]
    if not len(o): return np.nan
    v = [np.mean(np.maximum(ql * (o - qmat[:, j]), (ql - 1) * (o - qmat[:, j]))) for j, ql in enumerate(Q)]
    return 2 * np.mean(v)

def crps_clim(o):
    o = o[np.isfinite(o)]
    if len(o) < 3: return np.nan
    if len(o) > 1200: o = np.random.default_rng(0).choice(o, 1200, replace=False)
    return 0.5 * np.mean(np.abs(o[:, None] - o[None, :]))

def daily5(pipe, filled, tg, origins):
    inp = TC.build_predict_inputs(filled, tg, origins, CTX, cov=None)
    ql, _ = pipe.predict_quantiles(inp, prediction_length=W.OUT_W, quantile_levels=Q)
    p = np.stack([np.asarray(q)[0] for q in ql], 0)
    return p[:, :120, :].reshape(len(origins), 5, 24, len(Q)).mean(2)  # [N,5,nq]

def obs5_of(raw_t, origins):
    od = np.full(len(origins), np.nan)
    for k, o in enumerate(origins):
        lab = raw_t[o:o + W.OUT_W][:120]
        with np.errstate(invalid="ignore"):
            od[k] = np.nanmean(lab.reshape(5, 24), axis=1)[PDI]
    return od

def cqr(cal_lo, cal_hi, cal_y, te_lo, te_hi, alpha):
    """CQR: 검증 적합도점수로 구간 보정 → test 구간 반환."""
    E = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
    E = E[np.isfinite(E)]
    if len(E) < 10:
        return te_lo, te_hi
    n = len(E); lv = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Qv = np.quantile(E, lv, method="higher")
    return te_lo - Qv, te_hi + Qv

def cov(o, lo, hi):
    m = np.isfinite(o) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((o[m] >= lo[m]) & (o[m] <= hi[m]))) if m.any() else np.nan


def main():
    from chronos import Chronos2Pipeline
    sidx = pd.read_csv(os.path.join(OUT, "station_index.csv")); sidx["station"] = sidx["station"].astype(str)
    splits = json.load(open(os.path.join(OUT, "splits.json"), encoding="utf-8"))
    reps = {r: sidx[(sidx.river == r) & sidx.is_representative]["station"].tolist() for r in RIVERS}
    rows = []
    with gpu.VramMonitor(log_path=os.path.join(S.DEEP_SEE, "logs", f"{TASK}_vram.log")):
        base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
        log(f"base 로드 | {gpu.fmt()}", TASK)
        wides = {r: pd.read_parquet(os.path.join(OUT, f"{r}_auto_hourly_wide.parquet")) for r in RIVERS}
        for r in wides: wides[r]["station"] = wides[r]["station"].astype(str)

        for tg in S.TARGETS:
            adp = Chronos2Pipeline.from_pretrained(
                os.path.join(MODELS, f"chronos_lora512_{tg}", "finetuned-ckpt"), device_map="cuda")
            for river in RIVERS:
                wide = wides[river]
                for st in reps[river]:
                    times, raw, filled, _ = TC.station_series(wide, st)
                    n = len(filled[tg]); i_tr = _idx(times, splits[st]["train_end"]); i_va = _idx(times, splits[st]["val_end"])
                    te = TC.test_origins(n, i_va, 240, W.OUT_W, 24)            # test origins
                    cal = [i_tr + 240 + k * 24 for k in range(0, (i_va - i_tr - 360) // 24 + 1)]  # 검증 origins
                    cal = [o for o in cal if o - CTX >= 0]
                    if len(te) < 10: continue
                    o5 = obs5_of(raw[tg], te)
                    rdate5 = pd.to_datetime(times.iloc[[min(o + 4 * 24, n - 1) for o in te]].values)

                    # --- baselines ---
                    persistence = np.array([np.nanmean(raw[tg][max(0, o - 24):o]) for o in te])
                    tr_months = pd.DatetimeIndex(times.iloc[:i_tr].values).month
                    clim_month = pd.Series(raw[tg][:i_tr]).groupby(tr_months.to_numpy()).mean()
                    gmean = np.nanmean(raw[tg][:i_tr])
                    climatology = np.array([clim_month.get(int(m), gmean) for m in rdate5.month])

                    # --- legacy ---
                    leg = np.full(len(te), np.nan)
                    lfp = os.path.join(PRED, f"legacy_{river}_{st}_{tg}.parquet")
                    if os.path.exists(lfp):
                        lg = pd.read_parquet(lfp); lg["origin_time"] = pd.to_datetime(lg["origin_time"])
                        mp = dict(zip(lg["origin_time"], lg["pred_d5"]))
                        ot = times.iloc[te].values
                        leg = np.array([mp.get(pd.Timestamp(t), np.nan) for t in ot])

                    # --- chronos zs & lora (test + cal) ---
                    d_te_zs = daily5(base, filled, tg, te); d_te_lo = daily5(adp, filled, tg, te)
                    d_ca_zs = daily5(base, filled, tg, cal) if len(cal) >= 10 else None
                    d_ca_lo = daily5(adp, filled, tg, cal) if len(cal) >= 10 else None
                    y_cal = obs5_of(raw[tg], cal) if len(cal) >= 10 else None

                    # point rows
                    rows.append(_pt("persistence", river, tg, st, o5, persistence))
                    rows.append(_pt("climatology", river, tg, st, o5, climatology))
                    rows.append(_pt("legacy", river, tg, st, o5, leg))
                    cc = crps_clim(o5)
                    for name, dte, dca in [("chronos_zs512", d_te_zs, d_ca_zs), ("chronos_lora512", d_te_lo, d_ca_lo)]:
                        med = dte[:, PDI, QI[0.5]]
                        row = _pt(name, river, tg, st, o5, med)
                        cm = crps_q(o5, dte[:, PDI, :]); row["crps"] = cm
                        row["crps_skill"] = 1 - cm / cc if cc and cc > 0 else np.nan
                        # RAW coverage
                        row["cov80_raw"] = cov(o5, dte[:, PDI, QI[0.1]], dte[:, PDI, QI[0.9]])
                        row["cov90_raw"] = cov(o5, dte[:, PDI, QI[0.05]], dte[:, PDI, QI[0.95]])
                        # CONFORMAL coverage (검증구간 보정)
                        if dca is not None and y_cal is not None:
                            lo80, hi80 = cqr(dca[:, PDI, QI[0.1]], dca[:, PDI, QI[0.9]], y_cal,
                                             dte[:, PDI, QI[0.1]], dte[:, PDI, QI[0.9]], 0.2)
                            lo90, hi90 = cqr(dca[:, PDI, QI[0.05]], dca[:, PDI, QI[0.95]], y_cal,
                                             dte[:, PDI, QI[0.05]], dte[:, PDI, QI[0.95]], 0.1)
                            row["cov80_conf"] = cov(o5, lo80, hi80); row["cov90_conf"] = cov(o5, lo90, hi90)
                        rows.append(row)
                    log(f"{river}/{st}/{tg} done (te={len(te)} cal={len(cal)})", TASK)
            del adp; import torch; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(os.path.join(REP, "final_eval_metrics.csv"), index=False, encoding="utf-8-sig")
    _summary(pd.DataFrame(rows))


def rate5_months(dates):
    return [d.month for d in dates]

def _pt(model, river, tg, st, o, s):
    return {"model": model, "river": river, "target": tg, "station": st,
            "nse": nse(o, s), "rsr": rsr(o, s), "pbias": pbias(o, s), "r2": r2(o, s),
            "n": int((np.isfinite(o) & np.isfinite(s)).sum())}


def _summary(df):
    print("=== 모델별 타깃 평균 NSE ===")
    piv = df.pivot_table(index="model", columns="target", values="nse").round(3)
    order = ["persistence", "climatology", "legacy", "chronos_zs512", "chronos_lora512"]
    print(piv.reindex(order).to_string())
    print("\n=== 모델별 전체 평균(클립 -1) ===")
    for m in order:
        g = df[df.model == m]
        print(f"  {m:16s} NSE={g['nse'].clip(-1).mean():.3f} median={g['nse'].median():.3f} "
              f"RSR={g['rsr'].mean():.3f} |PBIAS|={g['pbias'].abs().mean():.1f}%")
    print("\n=== Chronos 신뢰구간 RAW vs CONFORMAL (목표 0.80/0.90) ===")
    for m in ["chronos_zs512", "chronos_lora512"]:
        g = df[df.model == m]
        print(f"  {m}: cov80 {g['cov80_raw'].mean():.3f}→{g['cov80_conf'].mean():.3f} | "
              f"cov90 {g['cov90_raw'].mean():.3f}→{g['cov90_conf'].mean():.3f} | "
              f"CRPS_skill={g['crps_skill'].mean():.3f}")


if __name__ == "__main__":
    main()
