# -*- coding: utf-8 -*-
"""시스템 CLI 진입점.

  replay   : test 구간 전 지점 재생 → forecasts/alerts 저장소 생성
  score    : 저장소 집계 → scorecard(+sanity gate)
  serve    : forecast_asof 샘플 JSON 생성(프론트 계약 데모)
  viz      : 대표지점 경보 타임라인 PNG
  tune     : 모델 개선 후보 학습(Stage F, run_tune 위임)
  promote  : 후보 게이트 판정 → registry 승급

실행: PYTHONIOENCODING=utf-8 python -m src.system.run_system <cmd> [opts]
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser(prog="run_system")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("replay")
    pr.add_argument("--scope", default="all", choices=["all", "rep"])
    pr.add_argument("--stations", default=None, help="쉼표구분 지점코드 필터")
    pr.add_argument("--targets", default=None, help="쉼표구분 타깃 필터")
    pr.add_argument("--smoke", action="store_true")
    pr.add_argument("--device", default="cuda")

    sub.add_parser("score")

    sv = sub.add_parser("serve")
    sv.add_argument("--stations", default=None)
    sv.add_argument("--asof", default=None)

    vz = sub.add_parser("viz")
    vz.add_argument("--targets", default=None)

    tu = sub.add_parser("tune")
    tu.add_argument("--target", required=True)
    tu.add_argument("--candidate", default=None, help="후보 태그(예: ctx1024, rank32)")
    tu.add_argument("--context", type=int, default=512)
    tu.add_argument("--rank", type=int, default=16)
    tu.add_argument("--num-steps", type=int, default=1500)
    tu.add_argument("--lora-lr", type=float, default=5e-5)
    tu.add_argument("--batch-size", type=int, default=16)

    pm = sub.add_parser("promote")
    pm.add_argument("--target", required=True)
    pm.add_argument("--candidate", required=True, help="models/cand_<tag>_<target> 의 <tag>")
    pm.add_argument("--apply", action="store_true", help="게이트 통과 시 실제 승급")

    args = ap.parse_args()
    if args.cmd == "replay":
        from src.system import replay
        sts = args.stations.split(",") if args.stations else None
        tgs = args.targets.split(",") if args.targets else None
        n_fc, n_al = replay.run(scope=args.scope, stations=sts, targets=tgs,
                                smoke=args.smoke, device=args.device)
        print(f"[replay] forecast rows={n_fc}, alert rows={n_al}")
    elif args.cmd == "score":
        from src.system import scorecard
        scorecard.run()
    elif args.cmd == "serve":
        from src.system import serve
        sts = args.stations.split(",") if args.stations else None
        paths = serve.dump_samples(stations=sts, asof=args.asof)
        for p in paths:
            print("sample:", p)
    elif args.cmd == "viz":
        from src.system import viz
        tgs = args.targets.split(",") if args.targets else None
        outs = viz.plot_representatives(tgs)
        print(f"[viz] {len(outs)} PNG 생성")
    elif args.cmd == "tune":
        from src.chronos import run_tune
        run_tune.train_candidate(target=args.target, tag=args.candidate, context=args.context,
                                 rank=args.rank, num_steps=args.num_steps,
                                 lora_lr=args.lora_lr, batch_size=args.batch_size)
    elif args.cmd == "promote":
        from src.chronos import run_tune
        run_tune.evaluate_and_promote(target=args.target, tag=args.candidate, apply=args.apply)


if __name__ == "__main__":
    main()
