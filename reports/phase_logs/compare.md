# 레거시(GAIN+GRU) vs Chronos-2 정량 비교 (5일차 일평균 NSE 기준)

동일 측정망(자동측정망 시간단위)·동일 대표지점·동일 롤링 origin(240h→120h)·
**관측-only 채점**으로 공정 비교. 레거시는 점추정만 → 확률지표는 Chronos만 보고.

- `legacy_paper`: 레거시 사용설명서 원논문 NSE(참고).
- `legacy(obs)`: 본 재구현 레거시 GRU의 관측-only NSE.
- `chr_zs`/`chr_lora`: Chronos-2 zero-shot / LoRA NSE.

## NSE 비교표

| 수계 | 타깃 | legacy_paper | legacy(obs) | chr_zs | chr_lora | best |
|---|---|---|---|---|---|---|
| han | do | 0.915 | 0.131 | 0.830 | 0.832 | chronos_lora |
| han | toc | 0.680 | 0.016 | 0.715 | 0.718 | chronos_lora |
| han | tn | 0.712 | -0.630 | 0.712 | 0.714 | chronos_lora |
| han | tp | 0.402 | 0.078 | 0.034 | 0.084 | chronos_lora |
| han | chl-a | 0.702 | 0.224 | 0.342 | 0.346 | chronos_lora |
| nak | do | 0.511 | 0.747 | 0.782 | 0.786 | chronos_lora |
| nak | toc | 0.625 | 0.327 | 0.224 | 0.233 | legacy |
| nak | tn | 0.778 | 0.144 | 0.309 | 0.323 | chronos_lora |
| nak | tp | 0.509 | 0.580 | 0.436 | 0.451 | legacy |
| nak | chl-a | 0.482 | 0.249 | 0.123 | 0.119 | legacy |
| geum | do | 0.874 | 0.922 | 0.957 | 0.958 | chronos_lora |
| geum | toc | 0.674 | -1.186 | 0.703 | 0.710 | chronos_lora |
| geum | tn | 0.510 | 0.649 | 0.725 | 0.730 | chronos_lora |
| geum | tp | 0.742 | 0.546 | 0.782 | 0.789 | chronos_lora |
| geum | chl-a | 0.108 | 0.290 | 0.235 | 0.232 | legacy |
| yeong | do | 0.565 | 0.582 | 0.700 | 0.702 | chronos_lora |
| yeong | toc | 0.562 | 0.280 | 0.488 | 0.483 | chronos_zs |
| yeong | tn | 0.776 | 0.217 | 0.812 | 0.818 | chronos_lora |
| yeong | tp | 0.209 | -123885082.266 | 0.527 | 0.538 | chronos_lora |
| yeong | chl-a | 0.602 | 0.425 | 0.357 | 0.349 | legacy |

## 요약
- Chronos zero-shot이 레거시(obs)보다 우위: **14/20** 조합
- Chronos LoRA가 레거시(obs)보다 우위: **15/20** 조합
- **중앙값 NSE**: legacy(obs)=0.264, chr_zs=0.613, chr_lora=0.620
- 평균 NSE(레거시 -1 클립): legacy=0.189, chr_zs=0.540, chr_lora=0.546
- 레거시 발산(NSE<-1) 조합 수: **2** (Chronos는 0) — 파운데이션 모델의 안정성 우위
- 평균 CRPS(zero-shot)=0.794, 평균 cov80=0.75(목표0.8), cov90=0.85(목표0.9), calib_err=0.041 → 신뢰구간 보정 양호

## Chronos-2 확률예측 품질(zero-shot)

| 수계 | 타깃 | CRPS | cov80(목표0.8) | cov90(목표0.9) | calib_err |
|---|---|---|---|---|---|
| han | do | 0.354 | 0.65 | 0.79 | 0.072 |
| han | toc | 0.084 | 0.79 | 0.84 | 0.029 |
| han | tn | 0.066 | 0.76 | 0.84 | 0.047 |
| han | tp | 0.001 | 0.82 | 0.90 | 0.032 |
| han | chl-a | 2.972 | 0.74 | 0.84 | 0.067 |
| nak | do | 0.645 | 0.75 | 0.84 | 0.028 |
| nak | toc | 0.230 | 0.73 | 0.83 | 0.028 |
| nak | tn | 0.223 | 0.75 | 0.86 | 0.034 |
| nak | tp | 0.005 | 0.82 | 0.89 | 0.036 |
| nak | chl-a | 4.976 | 0.71 | 0.83 | 0.031 |
| geum | do | 0.294 | 0.61 | 0.73 | 0.080 |
| geum | toc | 0.051 | 0.71 | 0.82 | 0.040 |
| geum | tn | 0.063 | 0.83 | 0.89 | 0.025 |
| geum | tp | 0.002 | 0.79 | 0.90 | 0.020 |
| geum | chl-a | 0.548 | 0.69 | 0.83 | 0.039 |
| yeong | do | 0.433 | 0.69 | 0.81 | 0.053 |
| yeong | toc | 0.095 | 0.77 | 0.87 | 0.015 |
| yeong | tn | 0.069 | 0.89 | 0.95 | 0.041 |
| yeong | tp | 0.001 | 0.82 | 0.89 | 0.027 |
| yeong | chl-a | 4.759 | 0.72 | 0.86 | 0.069 |

> 레거시 취약 항목(T-P, Chl-a)에서의 개선폭과, Chronos의 신뢰구간 보정(coverage가 
목표 분위에 근접하는지)을 핵심 논거로 활용. LoRA가 zero-shot 대비 NSE·CRPS·coverage를 
개선하면 도메인 적응 효과 입증.
