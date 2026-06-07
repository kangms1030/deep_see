# -*- coding: utf-8 -*-
"""Stage F: 모델 개선 트랙(후보 학습 + 게이트 승급, 블루-그린).

연구 결론상 한계효용은 작으나 사용자 요청으로 병행. incumbent(chronos_lora512_*)는
미파괴, 후보는 models/cand_<tag>_<target>/ 에 저장. 승급은 val replay 게이트
(ΔCRPS↓ AND ΔNSE≥0 AND coverage 목표 근접) 통과 시에만.

  train  : python -m src.system.run_system tune --target chl-a --candidate ctx1024 --context 1024
  promote: python -m src.system.run_system promote --target chl-a --candidate ctx1024 --apply
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from src.data import sources as S
from src.chronos import to_chronos as TC
from src.legacy import windows as W
from src.system import config as C
from src.system import context as CX
from src.system import forecast as FC
from src.system import conformal as CF
from src.system import registry as REG
from src.eval import metrics as Mx
from src.utils.progress import log
from src.utils import gpu

TASK = "sys_tune"


def _cand_dir(tag, target):
    return os.path.join(C.MODELS, f"cand_{tag}_{target}")


def train_candidate(target, tag, context=512, rank=16, num_steps=1500, lora_lr=5e-5,
                    batch_size=16, device="cuda"):
    """전 지점 train으로 후보 LoRA 어댑터 학습."""
    assert tag, "--candidate 태그 필요(예: ctx1024, rank32)"
    from chronos import Chronos2Pipeline
    splits = json.load(open(os.path.join(C.DATA_PROC, "splits.json"), encoding="utf-8"))
    out_dir = _cand_dir(tag, target)
    lora_cfg = {"r": rank, "lora_alpha": 2 * rank, "lora_dropout": 0.05}
    with gpu.VramMonitor(log_path=os.path.join(C.LOGS, f"{TASK}_vram.log")):
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
        log(f"[tune {target}/{tag}] base 로드 ctx={context} rank={rank} steps={num_steps} | {gpu.fmt()}", TASK)
        fit_inputs, val_inputs = [], []
        for r in C.RIVERS:
            wide = pd.read_parquet(os.path.join(C.DATA_PROC, f"{r}_auto_hourly_wide.parquet"))
            wide["station"] = wide["station"].astype(str)
            sts = sorted(wide["station"].unique())
            fi, vi = TC.build_finetune_inputs(wide, sts, target, splits, cov_wide=None)
            fit_inputs += fi; val_inputs += vi
        log(f"[tune {target}/{tag}] fit items={len(fit_inputs)} val={len(val_inputs)}", TASK)
        ft = pipe.fit(fit_inputs, prediction_length=W.OUT_W, validation_inputs=val_inputs or None,
                      finetune_mode="lora", lora_config=lora_cfg, learning_rate=lora_lr,
                      num_steps=num_steps, batch_size=batch_size, context_length=context,
                      output_dir=out_dir)
        gpu.check(where=f"tune fit {target}/{tag}")
        log(f"[tune {target}/{tag}] 학습 완료 → {out_dir}/finetuned-ckpt | {gpu.fmt()}", TASK)
        # 후보 메타(컨텍스트 길이 등) 기록 — promote 평가에서 사용
        meta = {"target": target, "tag": tag, "context": context, "rank": rank,
                "num_steps": num_steps, "lora_lr": lora_lr}
        with open(os.path.join(out_dir, "cand_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        del ft, pipe
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    return out_dir


def _val_eval(pipe, target, predict_ctx, origin_ctx, scope="rep"):
    """val origin에서 day5 점추정·확률 지표 풀링 평가.

    origin_ctx로 origin을 생성(incumbent/candidate 공통)하고 predict_ctx로 모델에 컨텍스트를
    먹인다 → 두 모델이 동일 origin에서 공정 비교(누수 없음, 컨텍스트 길이만 다름).
    """
    sidx = C.load_station_index()
    splits = json.load(open(os.path.join(C.DATA_PROC, "splits.json"), encoding="utf-8"))
    want = C.station_list(sidx, scope)
    O, M, QM = [], [], []          # obs5, med5, qmat5
    L80, H80, L90, H90 = [], [], [], []
    for rv in C.RIVERS:
        wide = pd.read_parquet(os.path.join(C.DATA_PROC, f"{rv}_auto_hourly_wide.parquet"))
        wide["station"] = wide["station"].astype(str)
        for r2, st in [(r, s) for r, s in want if r == rv]:
            if st not in splits:
                continue
            sd = CX.StationData(wide, st)
            if np.isfinite(sd.raw[target]).sum() < 50:
                continue
            i_tr, i_va = CX.split_indices(sd.times, splits[st])
            vo = sd.val_origins(i_tr, i_va, ctx=origin_ctx)
            if len(vo) < 5:
                continue
            q = FC.predict_daily_quantiles(pipe, sd.filled, target, vo, ctx=predict_ctx)
            obs = sd.obs_daily(target, vo)
            O.append(obs[:, 4]); M.append(q[:, 4, C.QI[0.5]]); QM.append(q[:, 4, :])
            L80.append(q[:, 4, C.QI[0.1]]); H80.append(q[:, 4, C.QI[0.9]])
            L90.append(q[:, 4, C.QI[0.05]]); H90.append(q[:, 4, C.QI[0.95]])
    if not O:
        return None
    o = np.concatenate(O); med = np.concatenate(M); qm = np.concatenate(QM)
    return {"nse": Mx.nse(o, med), "crps": Mx.crps_from_quantiles(o, C.Q, qm),
            "cov80": Mx.coverage(o, np.concatenate(L80), np.concatenate(H80)),
            "cov90": Mx.coverage(o, np.concatenate(L90), np.concatenate(H90)),
            "n": int(np.isfinite(o).sum())}


def evaluate_and_promote(target, tag, apply=False, scope="rep", device="cuda"):
    """incumbent vs 후보를 val에서 비교 → 게이트 판정(+선택적 승급)."""
    cand_path = os.path.join(_cand_dir(tag, target), "finetuned-ckpt")
    if not os.path.isdir(cand_path):
        raise FileNotFoundError(f"후보 없음: {cand_path} (먼저 tune 실행)")
    meta = {}
    mp = os.path.join(_cand_dir(tag, target), "cand_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    cand_ctx = int(meta.get("context", C.CTX))
    reg = REG.load()
    inc_path = REG.active_adapter(target, reg)
    inc_ctx = REG.active_context(target, reg)
    origin_ctx = max(inc_ctx, cand_ctx)        # 공통 origin(둘 다 컨텍스트 확보)

    with gpu.VramMonitor(log_path=os.path.join(C.LOGS, f"{TASK}_vram.log")):
        inc = FC.load_pipeline(inc_path, device)
        inc_m = _val_eval(inc, target, predict_ctx=inc_ctx, origin_ctx=origin_ctx, scope=scope)
        del inc; _free()
        cnd = FC.load_pipeline(cand_path, device)
        cnd_m = _val_eval(cnd, target, predict_ctx=cand_ctx, origin_ctx=origin_ctx, scope=scope)
        del cnd; _free()

    if inc_m is None or cnd_m is None:
        log(f"[promote {target}/{tag}] 평가표본 부족 → 보류", TASK); return False
    d_nse = cnd_m["nse"] - inc_m["nse"]
    d_crps = cnd_m["crps"] - inc_m["crps"]
    cov_inc = abs(cnd_m["cov80"] - 0.8) <= abs(inc_m["cov80"] - 0.8) + 0.02
    gate = (d_crps < 0) and (d_nse >= -1e-6) and cov_inc
    log(f"[promote {target}/{tag}] incumbent NSE={inc_m['nse']:.3f} CRPS={inc_m['crps']:.3f} "
        f"cov80={inc_m['cov80']:.2f} | cand NSE={cnd_m['nse']:.3f} CRPS={cnd_m['crps']:.3f} "
        f"cov80={cnd_m['cov80']:.2f} | ΔNSE={d_nse:+.3f} ΔCRPS={d_crps:+.3f} gate={gate}", TASK)
    if gate and apply:
        REG.promote(target, cand_path, f"cand_{tag}",
                    {"val": cnd_m, "vs_incumbent": {"d_nse": d_nse, "d_crps": d_crps}},
                    context=cand_ctx, reg=reg)
        log(f"[promote {target}/{tag}] ✅ 승급 완료(ctx={cand_ctx}) → registry.json", TASK)
    elif gate:
        log(f"[promote {target}/{tag}] 게이트 통과(미적용; --apply로 승급)", TASK)
    else:
        log(f"[promote {target}/{tag}] ❌ 게이트 미달 → incumbent 유지", TASK)
    return gate


def _free():
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
