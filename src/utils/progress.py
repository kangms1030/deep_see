# -*- coding: utf-8 -*-
"""진행률 가시화 유틸.

- tqdm 진행바(%, 경과/ETA)를 콘솔에 출력.
- 동시에 logs/<task>.log 에 주기적으로 % 라인을 기록(백그라운드 실행 시 대기 없이 진행 확인).
"""
from __future__ import annotations
import os
import sys
import time

from tqdm.auto import tqdm

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg: str, task: str = "run") -> None:
    """콘솔 + logs/<task>.log 동시 기록."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, f"{task}.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def track(iterable, desc="work", total=None, task="run", log_every_pct=5):
    """tqdm 진행바 + 로그파일에 N% 단위로 진행상황 기록.

    백그라운드 실행에서도 logs/<task>.log 를 tail 하면 %·ETA 확인 가능.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    bar = tqdm(iterable, desc=desc, total=total, dynamic_ncols=True,
               file=sys.stdout, mininterval=0.5)
    start = time.time()
    last_logged = -1
    logpath = os.path.join(LOG_DIR, f"{task}.log")
    for i, item in enumerate(bar, 1):
        yield item
        if total:
            pct = int(i * 100 / total)
            if pct >= last_logged + log_every_pct or i == total:
                last_logged = pct
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                msg = (f"[{time.strftime('%H:%M:%S')}] {desc}: {pct:3d}% "
                       f"({i}/{total}) elapsed={elapsed:6.1f}s eta={eta:6.1f}s")
                try:
                    with open(logpath, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass
    bar.close()
