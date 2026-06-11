# Archive — 과거 버전 및 완료된 실험 코드

> 이 폴더는 deep_see 프로젝트의 과거 버전 파일, 완료된 실험 코드, 중간 결과물을 보관합니다.
> 현재 활성 코드는 `deep_see/src/`에 있습니다. 여기의 코드는 **참조 전용**입니다.

## 보관 일시
2026-06-11

## 파일 목록

### src/chronos/ — 완료된 실험 스크립트
| 파일 | 원래 역할 | 보관 사유 |
|---|---|---|
| `run_context_sweep.py` | 컨텍스트 길이 스윕 실험 (240→1024) | 실험 완료, 최적값(512) 확정 |
| `run_cov_ablation.py` | 수문 공변량 ablation 실험 | 실험 완료, 결과 FINAL_REPORT에 반영 |
| `run_ctx512.py` | 컨텍스트 512 전용 실행기 | run_chronos.py --context 512로 대체 |
| `run_tune.py` | LoRA 튜닝 초기 버전 | run_chronos.py --mode lora로 대체 |

### src/eval/ — 통합 전 개별 비교 스크립트
| 파일 | 원래 역할 | 보관 사유 |
|---|---|---|
| `compare.py` | NSE 중심 레거시 vs Chronos 비교 | unified_compare.py로 대체 예정 |
| `prob_compare.py` | 확률 비교 (conformal 확률화) | unified_compare.py에 통합 예정 |
| `final_compare.py` | 5모델 최종 비교 | unified_compare.py에 통합 예정 |
| `final_eval.py` | 5모델 최종 평가 | unified_compare.py에 통합 예정 |
| `metric_audit.py` | 지표 감사 | 일회성 감사, 완료 |
| `high_intensity_audit.py` | 고강도 감사 | 일회성 감사, 완료 |
| `robustness_check.py` | 강건성 검사 | 일회성 검사, 완료 |

### reports/ — 과거 버전 리포트
| 파일 | 원래 역할 | 보관 사유 |
|---|---|---|
| `compare.md` | NSE 중심 비교 리포트 | unified_compare.md로 대체 예정 |
| `prob_compare.md` | 확률 비교 리포트 | unified_compare.md에 통합 예정 |
| `phase_logs/` | 단계별 실행 로그 (8개 md) | 과정 기록 보존 |

### models_baseline_nocov/ — 초기 LoRA 모델 (공변량 미사용)
- 5타깃별 LoRA 체크포인트 (rank 기본, step 1000)
- 현재 models/ 의 ctx512 LoRA로 대체됨

### reports_baseline_nocov/ — 초기 실험 결과
- 공변량 미사용 시의 예측 parquet, 비교 테이블, 경보 결과
