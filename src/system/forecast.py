# -*- coding: utf-8 -*-
"""예보 엔진: Chronos-2 어댑터로 origin 배치 분위예측 → 5일 일평균 분위.

- predict_daily_quantiles: filled 컨텍스트 → [N, 5일, nq] 분위(시간단위 120h를 일평균).
- 1모델 다지점 배치(predict_quantiles)로 저자원. baseline은 context.StationData가 제공.
- final_eval.daily5 / run_ctx512.eval_metrics와 동일 로직(컨텍스트 길이 파라미터화).
"""
from __future__ import annotations
import numpy as np

from src.system import config as C
from src.chronos import to_chronos as TC


def predict_daily_quantiles(pipe, filled, target, origins, ctx: int = C.CTX) -> np.ndarray:
    """origins에 대한 5일 일평균 분위 예측. 반환 shape [N, 5, nq=len(Q)]."""
    if not origins:
        return np.empty((0, C.HORIZON_DAYS, len(C.Q)))
    inp = TC.build_predict_inputs(filled, target, origins, ctx, cov=None)
    ql, _ = pipe.predict_quantiles(inp, prediction_length=C.OUT_W, quantile_levels=C.Q)
    p = np.stack([np.asarray(q)[0] for q in ql], axis=0)          # [N, OUT_W, nq]
    return p[:, :C.OUT_W, :].reshape(len(origins), C.HORIZON_DAYS, 24, len(C.Q)).mean(axis=2)


def load_pipeline(adapter_path: str, device: str = "cuda"):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(adapter_path, device_map=device)
