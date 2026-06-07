# `src/system` — Chronos-2 동적 수질오염 예측·경보 백엔드

연구(Phase 1~10)에서 검증된 **Chronos-2 LoRA@512 + conformal 보정 분위예측**을 운영형
예보·경보 파이프라인으로 조립한 모듈. 추가 데이터 없이 **보유 test 구간 replay**로
구동·검증한다. 설계 근거는 `deep_see/SYSTEM_DESIGN.md`, 성능 근거는 `deep_see/FINAL_REPORT.md`.

## 파이프라인
```
인입(wide parquet) → context(512h as-of 버퍼) → forecast(분위예측, 1~5일)
  → conformal(온라인 CQR 보정) → alerting(초과확률·등급·리드·비용가중α·가드레일)
  → 저장소(forecasts/alerts) → scorecard(검증·모니터링) / serve(API JSON) / viz(PNG)
```

## 모듈
| 파일 | 역할 |
|---|---|
| `config.py` | 분위·임계·보정창·경보등급·비용비·SLO·경로·지점목록(단일 출처) |
| `schemas.py` | ForecastRecord/AlertRecord/Scorecard + API JSON(프론트 계약) |
| `registry.py` | 타깃별 활성 어댑터·버전 관리, 블루-그린 promote/rollback |
| `context.py` | `StationData`: as-of 컨텍스트·test/val origin·관측·persistence·climatology |
| `forecast.py` | `predict_daily_quantiles`: origin 배치 분위예측(120h→5일 일평균) |
| `conformal.py` | `calibrate_series`: 인과적 온라인 CQR(실현 origin만, 풀링 폴백) |
| `alerting.py` | 초과확률·다horizon·리드타임·등급·비용가중 α |
| `replay.py` | 전 지점 백테스트 오케스트레이터 → forecasts/alerts 저장 |
| `scorecard.py` | 저장소 독립 집계(점추정·확률·경보)·sanity gate |
| `serve.py` | `forecast_asof(station, asof)` → JSON, `dump_samples` |
| `viz.py` | 경보 타임라인 PNG(시연 시안) |
| `run_system.py` | CLI: replay / score / serve / viz / tune / promote |

## 실행 (conda deep_see, PYTHONIOENCODING=utf-8)
```bash
python -m src.system.run_system replay --scope all          # 전 67지점 백테스트
python -m src.system.run_system replay --smoke --targets do,tp   # 스모크
python -m src.system.run_system score                       # 스코어카드+sanity
python -m src.system.run_system serve --stations S01001     # API 샘플 JSON
python -m src.system.run_system viz                         # 대표지점 타임라인 PNG
python -m src.system.run_system tune --target chl-a --candidate ctx1024 --context 1024
python -m src.system.run_system promote --target chl-a --candidate ctx1024 --apply
```

## 산출물 `deep_see/system_out/`
- `forecasts/{river}_{station}.parquet` — origin·타깃·horizon별 원분위(q0.05~q0.95)+보정밴드(q*_cal)+median+persistence+climatology+obs.
- `alerts/alert_log.parquet` — origin·타깃·horizon별 p_exceed·level·threshold·is_event·fired·lead_day·max_level·low_confidence·alpha.
- `scorecard/{target}.csv`, `summary.csv`, `by_station.csv`, `scorecard.md`.
- `api_samples/forecast_*.json` — 프론트 계약 데모.
- `figures/system_timeline_*.png`, `registry.json`.

## 설계 결정(연구 결론 반영)
- **stricter origins**: 컨텍스트가 test에 완전히 포함되도록 origin을 `i_va+512`부터 → 누수 0.
- **conformal 인과성**: horizon d 라벨은 origin+d·24h에 확정 → 그 시점 이후 표본만 보정에 사용.
- **가드레일**: persistence/climatology 동반 산출, 모델이 persistence 미달인 (지점,타깃)은 `low_confidence`.
- **타깃 운용**: DO·T-N 점추정+경보, Chl-a·T-P 경보 중심(`config.TARGET_MODE`).
- **모델 개선**: incumbent 미파괴, val replay 게이트(ΔCRPS↓·ΔNSE≥0·coverage 근접) 통과 시에만 승급.
