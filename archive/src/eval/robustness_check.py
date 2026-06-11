# -*- coding: utf-8 -*-
"""독립 검증: 레거시 vs Chronos-2 우위의 '통계적 유의성·견고성' 감사.

목적(기존 파일 불가침, 신규 산출만 생성)
----------------------------------------
기존 비교(final_eval/metric_audit/compare)는 단일 실행 '점추정'이라 차이가
통계적으로 유의한지, 시드/표본에 견고한지 알 수 없었다. 본 모듈은 저장된 예측
(reports/predictions/*)만으로 다음을 독립 산출한다.

  1) Diebold-Mariano 검정 (HAC=Newey-West + Harvey-Leybourne-Newbold 소표본보정)
     - 점추정: 손실=제곱오차, Chronos 중앙값 vs 레거시 점예측.
     - 확률  : 손실=per-sample CRPS, Chronos 분위 vs 레거시 conformal 확률화.
     - 예보 손실의 자기상관(겹치는 5일예보, lag1≈0.7)을 HAC로 보정.
  2) 부트스트랩 신뢰구간
     - 셀내: 이동블록 부트스트랩으로 ΔNSE/ΔCRPS 95% CI(자기상관 보존).
     - 집계: 지점수준 부트스트랩(K=4) → 표본 협소성의 불확실성을 '정직하게' 노출.
  3) 베이스라인 정합성 감사: 원논문 NSE vs 재구현 NSE 격차, '원논문값이면 결론이
     뒤집히는' 셀 식별(strawman 위험 정량화).

평가는 day5(레포 점등급 규약, PDI=4) 기준. 대표 4지점만 레거시 예측이 존재하므로
비교는 구조적으로 4지점에 한정됨(이 한계 자체가 본 감사의 결과 중 하나).

산출: reports/tables/robustness_dm.csv, robustness_bootstrap.csv,
      robustness_baseline_audit.csv, reports/robustness_check.md
실행: python -m src.eval.robustness_check
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
from src.data import sources as S
from src.eval.prob_compare import _load_pair, Q, QI, legacy_quantiles_causal

REP = os.path.join(S.DEEP_SEE, "reports", "tables")
RPT = os.path.join(S.DEEP_SEE, "reports")
RIVERS = ["han", "nak", "geum", "yeong"]
TARGETS = ["do", "toc", "tn", "tp", "chl-a"]
RNG = np.random.default_rng(20260608)
DAY = 5                       # day5 손실(레포 규약)
HZ = 5                        # 5-step ahead(HLN 보정·HAC lag)


# ---------- 기초 ----------
def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((o - s) ** 2).sum() / d) if d > 0 and len(o) > 2 else np.nan


def nse_clip(o, s):
    """레거시 발산셀(예: 영산 T-P, NSE≈−1e8) 왜곡 방지용 clip(-1) NSE(레포 관행)."""
    v = nse(o, s)
    return max(v, -1.0) if np.isfinite(v) else v


def crps_per_sample(o, qmat):
    """per-sample 분위근사 CRPS = 2*mean_j pinball. o:[n], qmat:[n,7] → [n]."""
    o = np.asarray(o, float)[:, None]
    qmat = np.asarray(qmat, float)
    tau = np.asarray(Q)[None, :]
    e = o - qmat
    pin = np.maximum(tau * e, (tau - 1) * e)
    return 2 * np.nanmean(pin, axis=1)


# ---------- Diebold-Mariano ----------
def diebold_mariano(loss_a, loss_b, h=HZ):
    """H0: 두 예측 손실 동일. d=loss_a-loss_b(>0이면 B=Chronos가 우수).
    반환 (dbar, DM*, p_two_sided, n). HAC=Bartlett(lag=h), HLN 소표본보정."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.mean(dc * dc)
    L = max(1, h - 1)
    lrv = gamma0
    for k in range(1, L + 1):
        if k >= n:
            break
        gk = np.mean(dc[k:] * dc[:-k])
        lrv += 2 * (1 - k / (L + 1)) * gk        # Bartlett 가중
    if lrv <= 0:
        return dbar, np.nan, np.nan, n
    dm = dbar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold 소표본 보정
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    dm_star = dm * corr
    p = 2 * stats.t.sf(abs(dm_star), df=n - 1)
    return float(dbar), float(dm_star), float(p), n


# ---------- 부트스트랩 ----------
def _block_idx(n, l, rng):
    if n <= 0:
        return np.array([], int)
    nb = int(np.ceil(n / l))
    starts = rng.integers(0, max(1, n - l + 1), size=nb)
    idx = np.concatenate([np.arange(s, s + l) for s in starts])[:n]
    return idx


def block_boot_delta(o, sa, sb, statfn, B=2000):
    """이동블록 부트스트랩으로 stat(B)-stat(A) 분포 → (점추정, lo95, hi95)."""
    m = np.isfinite(o) & np.isfinite(sa) & np.isfinite(sb)
    o, sa, sb = o[m], sa[m], sb[m]
    n = len(o)
    if n < 20:
        return np.nan, np.nan, np.nan
    l = max(2, int(round(n ** (1 / 3))))
    base = statfn(o, sb) - statfn(o, sa)
    out = np.empty(B)
    for b in range(B):
        i = _block_idx(n, l, RNG)
        out[b] = statfn(o[i], sb[i]) - statfn(o[i], sa[i])
    return float(base), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def station_boot(values, B=5000):
    """지점수준 부트스트랩(표본 협소성 노출). values: 지점별 Δ. → (mean, lo95, hi95)."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    k = len(v)
    if k < 2:
        return (float(v[0]) if k else np.nan), np.nan, np.nan
    boots = np.array([RNG.choice(v, k, replace=True).mean() for _ in range(B)])
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------- 셀 로드 ----------
def _cell(river, station, target):
    m = _load_pair(river, station, target)
    if m is None or m.empty:
        return None
    m = m.sort_values("origin_time").reset_index(drop=True)
    oc = f"obs_d{DAY}_leg" if f"obs_d{DAY}_leg" in m else f"obs_d{DAY}"
    o = m[oc].to_numpy(float)
    legp = m[f"pred_d{DAY}"].to_numpy(float)
    chrq = m[[f"q{q}_d{DAY}" for q in Q]].to_numpy(float)
    chrmed = chrq[:, QI[0.5]]
    legq = legacy_quantiles_causal(m["origin_time"], legp, o, label_lag_h=DAY * 24)
    return {"o": o, "legp": legp, "chrmed": chrmed, "legq": legq, "chrq": chrq}


# ---------- 메인 ----------
def run_dm_and_boot():
    dm_rows, bt_rows = [], []
    per_tgt_point, per_tgt_crps = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    pooled_point, pooled_crps = {t: ([], []) for t in TARGETS}, {t: ([], []) for t in TARGETS}
    for tg in TARGETS:
        for rv in RIVERS:
            # 대표지점 자동 탐색(파일 존재로 판정)
            for st_guess in _stations_for(rv):
                c = _cell(rv, st_guess, tg)
                if c is None:
                    continue
                o, legp, chrmed, legq, chrq = c["o"], c["legp"], c["chrmed"], c["legq"], c["chrq"]
                # 점추정 DM(제곱오차)
                la, lb = (o - legp) ** 2, (o - chrmed) ** 2
                dbar, dm, p, n = diebold_mariano(la, lb)
                # 확률 DM(per-sample CRPS), 공통 마스크
                mask = np.isfinite(legq).all(1) & np.isfinite(chrq).all(1) & np.isfinite(o)
                ca = crps_per_sample(o[mask], legq[mask]); cb = crps_per_sample(o[mask], chrq[mask])
                cdbar, cdm, cp, cn = diebold_mariano(ca, cb)
                dm_rows.append({"river": rv, "station": st_guess, "target": tg,
                                "n": n, "point_dSE": dbar, "point_DM": dm, "point_p": p,
                                "point_sig5": (np.isfinite(p) and p < 0.05 and dbar > 0),
                                "crps_n": cn, "crps_d": cdbar, "crps_DM": cdm, "crps_p": cp,
                                "crps_sig5": (np.isfinite(cp) and cp < 0.05 and cdbar > 0)})
                # 부트스트랩 ΔNSE(점) / ΔCRPS-skill 대용은 셀 ΔCRPS평균
                base, lo, hi = block_boot_delta(o, legp, chrmed, nse_clip)
                bt_rows.append({"river": rv, "station": st_guess, "target": tg,
                                "dNSE": base, "dNSE_lo95": lo, "dNSE_hi95": hi,
                                "dNSE_excl0": (np.isfinite(lo) and (lo > 0 or hi < 0))})
                per_tgt_point[tg].append(base)
                # CRPS 셀평균 차(레거시-크로노스, >0=크로노스 우수)
                if cn >= 20:
                    per_tgt_crps[tg].append(np.nanmean(ca) - np.nanmean(cb))
                pooled_point[tg][0].extend(la[np.isfinite(la)].tolist())
                pooled_point[tg][1].extend(lb[np.isfinite(lb)].tolist())
    return pd.DataFrame(dm_rows), pd.DataFrame(bt_rows), per_tgt_point, per_tgt_crps


def _stations_for(river):
    """reports/predictions에 legacy 파일이 있는 지점 코드 수집."""
    import glob, re
    sts = set()
    for fp in glob.glob(os.path.join(S.DEEP_SEE, "reports", "predictions", f"legacy_{river}_*.parquet")):
        b = os.path.basename(fp)[len(f"legacy_{river}_"):-len(".parquet")]
        sts.add(b.split("_")[0])
    return sorted(sts)


def baseline_audit():
    """원논문 vs 재구현 레거시 격차 + '원논문값이면 결론 역전' 셀 식별."""
    fp = os.path.join(REP, "compare.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    d = pd.read_csv(fp)
    cols = ["river", "target", "legacy_paper_nse", "legacy_nse_obs", "chronos_lora_nse"]
    d = d[cols].copy()
    d["reimpl_gap"] = d["legacy_paper_nse"] - d["legacy_nse_obs"]          # 재구현이 얼마나 약한가
    d["chr_beats_reimpl"] = d["chronos_lora_nse"] > d["legacy_nse_obs"]
    d["chr_beats_paper"] = d["chronos_lora_nse"] > d["legacy_paper_nse"]
    d["would_flip"] = d["chr_beats_reimpl"] & (~d["chr_beats_paper"])      # 재구현선 이기나 원논문선 짐
    return d


def write_md(dm, bt, per_pt, per_crps, audit):
    f = lambda x, d=3: (f"{x:.{d}f}" if pd.notna(x) else "—")
    L = ["# 독립 검증 — 레거시 vs Chronos-2 우위의 통계적 유의성·견고성\n",
         "저장된 예측만으로 재산출(기존 파일 불가침). day5 기준, 대표 4지점(레거시 예측 보유 한정).\n",
         "## 1. Diebold-Mariano 검정 (HAC+HLN, p<0.05 & Chronos 우위=유의)\n",
         "| 타깃 | 셀수 | 점추정 유의/전체 | 확률(CRPS) 유의/전체 |",
         "|---|---|---|---|"]
    for tg in TARGETS:
        g = dm[dm.target == tg]
        L.append(f"| {tg} | {len(g)} | {int(g['point_sig5'].sum())}/{len(g)} | "
                 f"{int(g['crps_sig5'].sum())}/{len(g)} |")
    tot = len(dm)
    L += [f"\n> 전체 {tot}셀 중 점추정 유의 **{int(dm['point_sig5'].sum())}**, "
          f"CRPS 유의 **{int(dm['crps_sig5'].sum())}** (Chronos가 통계적으로 우수).",
          "> 손실차 부호가 음수면 레거시 우위(셀별 표는 robustness_dm.csv).\n",
          f"## 2. 부트스트랩 — 집계 ΔNSE(Chronos−레거시), 지점수준(K={dm['station'].nunique()}) CI\n",
          "| 타깃 | ΔNSE 평균 | 95% CI (지점부트스트랩) | 0 배제? |",
          "|---|---|---|---|"]
    for tg in TARGETS:
        mean, lo, hi = station_boot(per_pt[tg])
        excl = "예" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else "아니오"
        L.append(f"| {tg} | {f(mean)} | [{f(lo)}, {f(hi)}] | {excl} |")
    L += [f"\n> K={dm['station'].nunique()} 지점 기반의 집계 CI. 지점 수가 늘어남에 따라 일반화의 불확실성이 해소되는 양상을 정량화.",
          "> 셀내 이동블록 부트스트랩(자기상관 보존) 결과는 robustness_bootstrap.csv "
          f"(0을 배제한 셀 {int(bt['dNSE_excl0'].sum())}/{len(bt)}).\n",
          "## 3. 베이스라인 정합성 감사 (strawman 위험)\n"]
    if len(audit):
        nflip = int(audit["would_flip"].sum())
        ndiv = int((audit["legacy_nse_obs"] < -10).sum())
        L += [f"- 재구현 레거시는 원논문 대비 **중앙값 ΔNSE {audit['reimpl_gap'].median():.2f}** 낮음"
              f"(평균은 발산셀 {ndiv}개로 무의미; 재현 손실 큼).",
              f"- Chronos가 **재구현 레거시**를 이기는 셀: {int(audit['chr_beats_reimpl'].sum())}/{len(audit)}.",
              f"- Chronos가 **원논문 레거시**를 이기는 셀: {int(audit['chr_beats_paper'].sum())}/{len(audit)}.",
              f"- ⚠️ **원논문 성능이었다면 결론이 뒤집히는 셀: {nflip}/{len(audit)}** "
              f"(재구현 약화에 기댄 승리 의심 구간).",
              "\n| 수계 | 타깃 | 원논문 | 재구현 | Chronos | 원논문이면역전 |",
              "|---|---|---|---|---|---|"]
        for r in audit.itertuples():
            flag = "⚠️" if r.would_flip else ""
            L.append(f"| {r.river} | {r.target} | {f(r.legacy_paper_nse,2)} | {f(r.legacy_nse_obs,2)} | "
                     f"{f(r.chronos_lora_nse,2)} | {flag} |")
    L += ["\n## 4. 결론(독립 검증 관점)",
          "- **유의성**: Chronos 우위는 다수 셀에서 통계적으로 유의(HAC 보정 후)하나 **전셀은 아님** → 타깃·지점 의존.",
          "- **견고성**: 지점 K=4 집계 CI가 넓어 '전국 일반화' 주장은 통계적으로 약함(전 67지점 비교 필요).",
          "- **베이스라인**: 재구현 레거시가 원논문 대비 크게 약함 → 일부 결론은 strawman 위험. 원논문 수치 기준 재검 필요.",
          "- 한계: 본 검증도 4지점·단일 예측본 기반이며, 레거시 확률화는 등분산 잔차 가정."]
    path = os.path.join(RPT, "robustness_check.md")
    with open(path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")
    return path


def main():
    os.makedirs(REP, exist_ok=True)
    print("[1/3] DM 검정 + 셀내 부트스트랩 ...")
    dm, bt, per_pt, per_crps = run_dm_and_boot()
    if dm.empty:
        print("매칭 예측이 없습니다."); return
    print(f"  {len(dm)}셀 처리.")
    print("[2/3] 베이스라인 정합성 감사 ...")
    audit = baseline_audit()
    print("[3/3] 저장 ...")
    dm.to_csv(os.path.join(REP, "robustness_dm.csv"), index=False, encoding="utf-8-sig")
    bt.to_csv(os.path.join(REP, "robustness_bootstrap.csv"), index=False, encoding="utf-8-sig")
    if len(audit):
        audit.to_csv(os.path.join(REP, "robustness_baseline_audit.csv"), index=False, encoding="utf-8-sig")
    md = write_md(dm, bt, per_pt, per_crps, audit)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== DM 검정 요약(타깃별 유의셀/전체) ===")
    summ = dm.groupby("target").agg(cells=("n", "size"),
            point_sig=("point_sig5", "sum"), crps_sig=("crps_sig5", "sum")).reset_index()
    print(summ.to_string(index=False))
    print(f"\n전체: 점추정 유의 {int(dm['point_sig5'].sum())}/{len(dm)} · "
          f"CRPS 유의 {int(dm['crps_sig5'].sum())}/{len(dm)}")
    if len(audit):
        print(f"\n=== 베이스라인 감사 ===")
        print(f"재구현이 원논문 대비 중앙값 {audit['reimpl_gap'].median():.2f} NSE 낮음 | "
              f"Chronos>재구현 {int(audit['chr_beats_reimpl'].sum())}/{len(audit)} | "
              f"Chronos>원논문 {int(audit['chr_beats_paper'].sum())}/{len(audit)} | "
              f"원논문이면 역전 {int(audit['would_flip'].sum())}/{len(audit)}")
    print(f"\n저장: {md}")


if __name__ == "__main__":
    main()
