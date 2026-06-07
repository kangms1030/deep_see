# -*- coding: utf-8 -*-
"""GPU VRAM 모니터링 + 16GB 하드캡 가드.

RTX 5060 Ti 총 VRAM = 15.93GB. 본 프로젝트는 16GB를 넘기면 안 되므로
실사용(reserved) 안전 임계값을 14.5GB로 두고, 근접 시 경고/캐시정리/예외를 발생시킨다.
"""
from __future__ import annotations
import os
import time
import threading
import subprocess

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

GB = 1024 ** 3
SAFE_LIMIT_GB = 14.5   # reserved 메모리 안전 임계 (총 15.93GB 중)
HARD_LIMIT_GB = 15.5   # 이 이상이면 OOM 위험 → 예외


def vram_stats(device: int = 0) -> dict:
    """현재 VRAM 사용량(GB). torch 기준 + 가능하면 nvidia-smi 실제값."""
    out = {"alloc": 0.0, "reserved": 0.0, "max_reserved": 0.0, "total": 0.0, "smi_used": None}
    if _HAS_TORCH and torch.cuda.is_available():
        out["alloc"] = torch.cuda.memory_allocated(device) / GB
        out["reserved"] = torch.cuda.memory_reserved(device) / GB
        out["max_reserved"] = torch.cuda.max_memory_reserved(device) / GB
        out["total"] = torch.cuda.get_device_properties(device).total_memory / GB
    used = _nvidia_smi_used(device)
    if used is not None:
        out["smi_used"] = used
    return out


def _nvidia_smi_used(device: int = 0):
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(device)],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().splitlines()[0]) / 1024.0  # MiB->GB
    except Exception:
        return None
    return None


def fmt(device: int = 0) -> str:
    s = vram_stats(device)
    smi = f", smi_used={s['smi_used']:.2f}GB" if s["smi_used"] is not None else ""
    return (f"VRAM alloc={s['alloc']:.2f} reserved={s['reserved']:.2f} "
            f"peak={s['max_reserved']:.2f}/{s['total']:.2f}GB{smi}")


def check(device: int = 0, where: str = "") -> None:
    """안전 임계 초과 시 캐시 비우고, 하드 한도 초과 시 예외."""
    if not (_HAS_TORCH and torch.cuda.is_available()):
        return
    s = vram_stats(device)
    if s["reserved"] >= SAFE_LIMIT_GB:
        torch.cuda.empty_cache()
        s2 = vram_stats(device)
        print(f"[VRAM-WARN]{(' '+where) if where else ''} reserved {s['reserved']:.2f}GB "
              f">= {SAFE_LIMIT_GB}GB → empty_cache → {s2['reserved']:.2f}GB")
        if s2["reserved"] >= HARD_LIMIT_GB:
            raise RuntimeError(
                f"[VRAM-HARDCAP] reserved {s2['reserved']:.2f}GB >= {HARD_LIMIT_GB}GB at {where}. "
                f"batch_size/context를 줄이세요.")


def reset_peak(device: int = 0) -> None:
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


class VramMonitor(threading.Thread):
    """백그라운드로 주기 로깅하며 16GB 가드. with 블록으로 사용."""

    def __init__(self, device: int = 0, interval: float = 15.0, log_path: str | None = None):
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self.log_path = log_path
        self._stop = threading.Event()
        self.peak = 0.0

    def run(self):
        while not self._stop.wait(self.interval):
            s = vram_stats(self.device)
            self.peak = max(self.peak, s["reserved"])
            line = f"[VRAM-MON] {fmt(self.device)}"
            print(line, flush=True)
            if self.log_path:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
            if s["reserved"] >= HARD_LIMIT_GB:
                print(f"[VRAM-MON][CRITICAL] reserved {s['reserved']:.2f}GB >= {HARD_LIMIT_GB}GB", flush=True)

    def __enter__(self):
        if _HAS_TORCH and torch.cuda.is_available():
            self.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
