# -*- coding: utf-8 -*-
"""모델 레지스트리: 타깃별 활성 어댑터·버전·승급 근거(val 지표)를 registry.json으로 관리.

- 초기 incumbent = models/chronos_lora512_{tg}/finetuned-ckpt (연구 검증 모델).
- promote: 후보가 게이트 통과 시에만 활성 어댑터 교체(블루-그린, ckpt 미덮어쓰기).
- rollback: 직전 버전으로 복귀.
"""
from __future__ import annotations
import os
import json
import time

from src.system import config as C


def _default_entry(tg: str) -> dict:
    return {"adapter": os.path.join(C.MODELS, f"chronos_lora512_{tg}", "finetuned-ckpt"),
            "version": "lora512", "context": C.CTX, "val_metrics": {},
            "updated": None, "history": []}


def load() -> dict:
    if os.path.exists(C.REGISTRY_PATH):
        with open(C.REGISTRY_PATH, encoding="utf-8") as f:
            reg = json.load(f)
    else:
        reg = {}
    for tg in C.TARGETS:
        reg.setdefault(tg, _default_entry(tg))
    return reg


def save(reg: dict) -> None:
    C.ensure_dirs()
    with open(C.REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def active_adapter(target: str, reg: dict | None = None) -> str:
    reg = reg or load()
    path = reg[target]["adapter"]
    if not os.path.isdir(path):
        # 안전 폴백: 기본 incumbent
        path = _default_entry(target)["adapter"]
    return path


def active_context(target: str, reg: dict | None = None) -> int:
    reg = reg or load()
    return int(reg[target].get("context", C.CTX))


def promote(target: str, adapter_path: str, version: str, val_metrics: dict,
            context: int = C.CTX, reg: dict | None = None) -> dict:
    """후보를 활성으로 승급(이전 활성은 history에 적재). context=추론 입력 길이."""
    reg = reg or load()
    prev = {k: reg[target].get(k) for k in ("adapter", "version", "context", "val_metrics", "updated")}
    reg[target].setdefault("history", []).append(prev)
    reg[target].update({"adapter": adapter_path, "version": version, "context": int(context),
                        "val_metrics": val_metrics,
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    save(reg)
    return reg


def rollback(target: str, reg: dict | None = None) -> dict:
    reg = reg or load()
    hist = reg[target].get("history", [])
    if hist:
        prev = hist.pop()
        reg[target].update(prev)
        save(reg)
    return reg
