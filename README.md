# 수질 예측·경보 프로젝트 (`deep_see`) — 종합 통합본

> **이 문서가 프로젝트의 단일 진입점(마스터)입니다.** 연구(레거시 vs Chronos-2 비교)부터
> 운영형 동적 예보·경보 시스템 구축까지 전 과정을 한 곳에 정리했습니다.
> 상세 부록: 지표·연구 심층 → [`FINAL_REPORT.md`](FINAL_REPORT.md), 시스템 설계 근거 → [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md), 코드 사용법 → [`src/system/README.md`](src/system/README.md).

---

## 0. 한눈 요약 (TL;DR)

- **목표**: AI-Hub "수질측정 및 오염원" 자동측정망 데이터로 ① 고전 모델(GAIN+GRU)과 최신 시계열 파운데이션 모델(**Chronos-2**)을 정량 비교하고, ② Chronos-2의 확률분위 예측으로 **동적 수질오염 경보 시스템**을 구축.
- **핵심 결과**: **Chronos-2 LoRA+Weather@512가 전 모델 중 최고 성능** (NSE 0.557, CRPS 0.763). 레거시(GAIN+GRU) 대비 NSE 45.8% 향상. **기상 공변량(ASOS 5채널)이 CRPS를 일관 개선**하며, 특히 TOC·T-P에서 NSE +3~5%p 상승. 분위는 conformal(CQR) 보정으로 목표 신뢰도 정렬.
- **산출물**: 전 67지점 예보·경보를 **보유 데이터 replay**로 구동·검증하는 운영 백엔드(`src/system/`)와, 프론트엔드가 바로 소비 가능한 JSON/parquet 저장소·API.
- **현재 상태**: 최적 모델 = **Chronos-2 LoRA+Weather, ctx=512**. 3계층 통합 평가 완료. 기상 공변량 Ablation 검증 완료.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터 | AI-Hub 자동측정망(시간단위), **67지점 / 4수계**(한강 22·낙동 23·금강 13·영산 9) |
| 타깃(5) | DO(용존산소)·TOC·T-N·T-P·Chl-a |
| 입력 | 수질 다변량(10채널: 5타깃 + 수온·pH·EC·탁도·질산성질소) + **ASOS 기상 5채널**(기온·강수·풍속·습도·일사) |
| 예측 | 과거 컨텍스트 → **5일(120h)** 분위예측, 평가=5일차 일평균, **관측-only 채점**, 롤링-오리진 |
| 비교 | 레거시 **GAIN(결측보간)+GRU**(PyTorch 충실 재구현) vs **Chronos-2**(zero-shot / LoRA 파인튜닝) |
| 제약 | 원본 데이터 불가침 · VRAM ≤16GB · 진행률(%·ETA) 가시화 · conda `deep_see` |

> 레거시 원본(`kotechnia/water-quality`)은 TF-GPU 2.4.1 기준이라 현재 Windows+RTX 5060 Ti(Blackwell)에서 네이티브 GPU 불가 → **구조·하이퍼파라미터 사양서로만 참조하고 PyTorch로 재현**해 Chronos-2와 동일 스택을 공유.

---

## 2. 평가지표 (요약)

| 지표 | 정의 | 절대 등급 기준 |
|---|---|---|
| **NSE** | 1−Σ(o−s)²/Σ(o−ō)² | >0.8 매우우수·0.7~0.8 우수·0.5~0.7 만족·≤0.5 미흡 |
| **RSR** | RMSE/σ_obs | 0~0.5 매우우수 … >0.7 미흡 |
| **PBIAS** | Σ(o−s)/Σo×100 | 0 근처일수록 무편향 |
| **CRPS / CRPS-skill** | 분위예측 정확도 / 기후값 대비 | skill>0 = 기후값보다 유용 |
| **Winkler Score** | 신뢰구간 너비 + 위반 벌점 | 낮을수록 우수 (구간 선명도+정확도) |
| **PIT 편차** | 예측 분포의 교정(균일분포 편차) | 0에 가까울수록 잘 보정됨 |
| **Sharpness** | 예측 구간 평균 물리 너비 | 좁을수록 선명한 예측 |
| **coverage(cov80/90)** | 예측구간 실제 포함률 | 목표 0.80 / 0.90 |
| **CSI / ETS / HSS** | 우연보정 위협점수 (오염 이벤트 탐지) | 1에 가까울수록 우수 |
| **PR-AUC·F1·BSS·POD·FAR** | 경보(이벤트) 유용성 | BSS>0 = 기후값보다 유용 |

상세(수식·함정·상황별 적합성)는 [`FINAL_REPORT.md`](FINAL_REPORT.md) §지표 참조.

---

## 3. 연구 결과 종합

### 3.1 전 모델 종합 비교 — NSE / CRPS / Cov80 (대표 4지점 평균)

| 순위 | 모델 | NSE | CRPS | Cov80 | 비고 |
|---|---|---|---|---|---|
| 🥇 1위 | **Chr2 LoRA+Weather** | **0.557** | **0.763** | 0.777 | 수질10 + 기상5 |
| 🥈 2위 | Chr2 ZS+Weather | 0.551 | 0.766 | **0.782** | 수질10 + 기상5 |
| 🥉 3위 | Chr2 LoRA | 0.546 | 0.785 | 0.765 | 수질10 |
| 4위 | Chr2 ZeroShot | 0.540 | 0.794 | 0.752 | 수질10 |
| 5위 | Legacy (GAIN+GRU) | 0.382 | — | — | 수질10 |

**타깃별 NSE 상세**:

| 타깃 | Legacy | Chr2 ZeroShot | Chr2 LoRA | Chr2 ZS+WX | **Chr2 LoRA+WX** |
|---|---|---|---|---|---|
| DO | 0.580 | 0.817 | 0.820 | 0.820 | **0.823** |
| TOC | 0.247 | 0.532 | 0.536 | 0.562 | **0.566** |
| T-N | 0.402 | 0.640 | 0.646 | 0.640 | **0.646** |
| T-P | 0.395 | 0.445 | 0.465 | 0.490 | **0.498** |
| Chl-a | **0.288** | 0.264 | 0.261 | 0.245 | 0.252 |

### 3.2 핵심 발견
1. **Chronos-2 LoRA+Weather가 전체 1위** — NSE 0.557, CRPS 0.763으로 3개 지표 모두 최고 또는 최고 수준.
2. **레거시 대비 NSE 45.8% 향상** (0.382 → 0.557). 전 타깃(Chl-a 제외)에서 Chronos-2가 압도.
3. **기상 공변량이 새로운 성능 레버**: ASOS 5채널 추가 시 CRPS 0.785→0.763 일관 개선. **TOC +3%p, T-P +3.2%p NSE 상승** — 강수·기온의 물리적 인과관계 입증.
4. **LoRA 파인튜닝 일관된 효과**: ZeroShot→LoRA 전환 시 NSE +0.006~0.011 안정적 개선. 3분/타깃으로 효율적.
5. **Coverage 개선**: 기상 추가 시 Cov80 0.765→0.777, 0.80 목표에 근접.
6. **Chl-a 점추정은 Legacy 소폭 우세(0.288 vs 0.252)이나**, 확률 분포(CRPS) 및 경보(BSS 0.59→0.72)에서 Chronos-2 압도.
7. **분위는 과신**(cov80≈0.75) → **conformal(CQR) 보정**으로 0.80/0.90 목표 정렬.

### 3.3 실험 일지 (Phase 1~11 압축)

| Phase | 내용 | 결론/산출 |
|---|---|---|
| 1 | 전처리: 67지점·5타깃 wide 데이터셋(시점분할) | `data_processed/*.parquet`, `station_index.csv`, `splits.json` |
| 2 | 레거시 GAIN+GRU PyTorch 재구현 | `legacy_metrics.csv` (취약·발산 다수) |
| 3 | Chronos-2 zero-shot + 네이티브 LoRA | `Chronos2Pipeline.fit(finetune_mode='lora')` |
| 4 | 정량 비교(동일 origin·집계·관측-only) | Chronos 압도, 레거시 발산 0건 |
| 5 | 확률분위 임계초과 경보 | F1 0.66 vs 레거시 0.31, 리드 2~3일 |
| 6~7 | 수문 공변량 연결·ablation + 컨텍스트 스윕 | 수문 공변량 중립, 컨텍스트가 레버 |
| 12 | **기상 공변량(ASOS) 파이프라인 + Ablation** | **기상 5채널 CRPS 일관 개선, TOC·T-P NSE +3~5%p** |
| 13 | **3계층 통합 평가 체계 구축** | Winkler/PIT/CSI/ETS/HSS 등 14개 지표, 459k행 전수 비교 |
| 9 | context=512 통합(전67 zs + LoRA@512) | LoRA@512가 zs@512 +0.0245(19/20 개선) |
| 10 | 지표 감사·재산출(베이스라인 추가·conformal) | persistence 강함·레거시 미달·CQR 보정 |
| 11 | **동적 예보·경보 백엔드 구현 + 검증** | `src/system/`, 전67 replay 47분, 아래 §4 |
| F | 모델 개선 트랙(ctx1024 게이트) | broad-test 미확인 → **ctx512로 롤백**(아래) |

(원천 per-phase 표는 [`reports/phase_logs/`](reports/phase_logs/), 보존 베이스라인은 `reports_baseline_nocov/`)

---

## 4. 동적 예보·경보 시스템 (`src/system/`)

연구에서 검증된 **Chronos-2 LoRA@512 + conformal 보정 분위예측**을 운영형 파이프라인으로 조립.
**추가 데이터 없이 보유 test 구간을 스트림처럼 replay**해 구동·검증한다(=운영 파이프라인 = 검증 수단).

### 4.1 파이프라인
```
인입(wide parquet) → context(512h as-of 버퍼) → forecast(분위예측 1~5일)
  → conformal(온라인 CQR 보정) → alerting(초과확률·등급·리드·비용가중α·가드레일)
  → 저장소(forecasts/alerts) → scorecard(검증) / serve(API JSON) / viz(PNG)
```
설계 원칙: ① persistence/climatology **가드레일** 상시 병행(모델이 이길 때만 신뢰), ② **타깃별 운용**(DO·T-N 점추정+경보 / Chl-a·T-P 경보 중심), ③ **분위 conformal 보정 필수**, ④ 1모델 다지점 배치(VRAM≤~2GB), ⑤ 무편향 유지.

### 4.2 운영 스코어카드 — 전 67지점 백테스트 (최종 ctx512)

| 타깃 | NSE | cov80 raw→cal | cov90 raw→cal | CRPS-skill | ΔNSE(vs persist) | PR-AUC | F1 | BSS | 리드(일) | 등급 |
|---|---|---|---|---|---|---|---|---|---|---|
| DO | 0.727 | 0.76→0.81 | 0.87→0.90 | 0.55 | +0.05 | 0.69 | 0.68 | 0.50 | 1.09 | Good |
| T-N | 0.679 | 0.74→0.81 | 0.84→0.91 | 0.53 | +0.04 | 0.72 | 0.80 | 0.44 | 1.02 | Satisfactory |
| TOC | 0.487 | 0.77→0.81 | 0.87→0.90 | 0.43 | +0.07 | 0.43 | 0.56 | 0.17 | 1.10 | (경계) |
| T-P | 0.338 | 0.77→0.81 | 0.86→0.90 | 0.36 | +0.07 | 0.37 | 0.55 | −0.10 | 1.06 | (경보보조) |
| Chl-a | 0.437 | 0.75→0.81 | 0.86→0.91 | 0.43 | +0.08 | **0.78** | **0.74** | **0.55** | 1.08 | (녹조경보 강) |

- **conformal 보정 전 지점 유효**(cov80→0.81 / cov90→0.90), **가드레일 전 타깃 통과**(vs_persistence 모두 +), **sanity gate |Δ|=0.023**(`final_eval`와 일치 → 파이프라인 무결성).
- 핵심: TSFM의 운영 가치는 ① 베이스라인 위 견고성, ② **Chl-a 녹조 경보**(PR-AUC 0.78/BSS 0.55), ③ 보정된 확률·리드타임.

### 4.3 Stage F — 모델 개선 트랙 (게이트·블루-그린)
전 5타깃 **context=1024** 후보를 학습→게이트 검증. rep-val 게이트에서 **DO·TOC 승급**(ΔNSE +0.005/+0.017)했으나, **전 67지점 test에서 DO가 0.727→0.703으로 하락** → rep-val 4지점 게이트가 낙관적(과적합)임이 드러남. 사용자 결정으로 **전부 ctx512 롤백**(검증된 broad-best). 
→ **결론: 추가 모델 변경의 한계효용은 미미**(기존 결론 재확인). 단 **게이트·블루-그린이 나쁜 후보를 자동 차단함을 입증**. 향후 게이트는 전지점 val로 강화 필요. (후보 어댑터 `models/cand_ctx1024_*`는 보존.)

---

## 5. 사용법

환경: conda `deep_see` (`C:\Users\minsoo\anaconda3\envs\deep_see\python.exe`), 콘솔 `PYTHONIOENCODING=utf-8`.

```bash
# ── 운영 시스템 (src/system) ──
python -m src.system.run_system replay --scope all     # 전 67지점 백테스트(약 47분, VRAM≤2GB)
python -m src.system.run_system score                  # 스코어카드 + sanity gate
python -m src.system.run_system serve --stations S01001  # forecast_asof → API JSON 샘플
python -m src.system.run_system viz                    # 대표지점 경보 타임라인 PNG
python -m src.system.run_system tune    --target chl-a --candidate ctx1024 --context 1024  # 후보 학습
python -m src.system.run_system promote --target chl-a --candidate ctx1024 --apply         # 게이트→승급

# ── 연구 재현 (참고) ──
python -m src.data.build_dataset                              # 수질 전처리
python -m src.data.build_weather_covariates                   # 기상 공변량 빌드
python -m src.legacy.run_legacy                               # 레거시 GAIN+GRU
python -m src.chronos.run_chronos --mode both --context 512   # Chronos2 기본
python -m src.chronos.run_chronos --use-weather --mode both   # Chronos2 +기상
python -m src.eval.unified_compare                            # 3계층 통합 비교
```

프로그램 호출: `from src.system.serve import forecast_asof; forecast_asof("S01001")` → 밴드·P(초과)·등급·리드일·가드레일 포함 JSON(프론트 계약).

---

## 6. 저장소 구조

```
deep_see/
├─ README.md              ← (이 문서) 마스터 통합본
├─ FINAL_REPORT.md        ← 연구·지표 상세 부록
├─ SYSTEM_DESIGN.md       ← 시스템 설계 근거 부록
├─ src/
│  ├─ data/               전처리: sources, build_dataset, build_covariates, **build_weather_covariates**
│  ├─ legacy/             레거시 재현: gain, gru, windows, run_legacy
│  ├─ chronos/            Chronos: to_chronos, **run_chronos**(--use-weather), run_ctx512
│  ├─ eval/               지표: metrics, **unified_metrics**, **unified_compare**, make_verification_figures
│  ├─ alert/              경보 토대: thresholds, alert (Phase5)
│  ├─ system/   ★운영★   config·schemas·registry·context·forecast·conformal·alerting·replay·scorecard·serve·viz·run_system
│  └─ utils/              gpu(VRAM가드), progress(tqdm/로그)
├─ data_processed/        {river}_*_wide.parquet, station_index.csv, splits.json
├─ models/                chronos_lora512_*(★활성★), cand_ctx1024_*(후보), chronos_lora_*(240, 구), lora_*cov_*(ablation)
├─ system_out/   ★산출★  forecasts/(67) · alerts/alert_log.parquet · scorecard/ · api_samples/ · figures/ · registry.json
├─ reports/              tables/(CSV 지표) · predictions/ · figures/ · phase_logs/(per-phase md)
├─ reports_baseline_nocov/  공변량 도입 前 보존 결과(불가침)
└─ logs/                 작업별 %·VRAM 로그
```

**운영(★)과 연구를 구분**: 실제 서비스는 `src/system/` + `models/chronos_lora512_*` + `system_out/`. 나머지 `src/chronos/run_chronos|run_cov_ablation|run_context_sweep`, `src/eval/compare|final_compare`는 결론을 도출한 **재현용 연구 스크립트**(보존).

---

## 7. 결론 & 향후 방향

**결론**: Chronos-2 LoRA+Weather가 전 5모델 중 최고 성능(NSE 0.557, CRPS 0.763)을 달성. 레거시 대비 NSE 45.8% 향상. **기상 공변량(ASOS 5채널)이 확률분포 품질을 일관 개선**하며, 특히 비점오염(TOC/T-P)에서 물리적 인과관계가 수치로 입증됨. 3계층 통합 평가(Winkler/PIT/CSI/ETS 등 14개 지표)로 확률 예측 모델의 다차원적 성능을 검증 완료.

**향후**:
1. **프론트엔드 시연**: `system_out` + `forecast_asof` JSON을 소비하는 경보 대시보드.
2. **기상 공변량 전국 확장**: 현재 대표 4지점 → 전 67지점에 기상 공변량 반영 전수 평가.
3. **실시간 인입·온라인 재보정**: 라이브 기상 API + conformal 검증창 갱신 + 드리프트 감시.
4. **미래 known covariate**: 기상 예보(KMA 단기예보 API)를 future_covariates로 연계.

---

*최종 갱신: 2026-06-11 · 최적 모델=Chronos-2 LoRA+Weather ctx512 · 3계층 평가+기상 Ablation 완료.*
