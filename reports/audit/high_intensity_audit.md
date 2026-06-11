# High-Intensity Audit: deep_see Operational Forecasts

## Executive Summary
- 운영 경로의 Chronos 입력은 외생 공변량 없이 생성된다. 따라서 현재 운영 시스템은 causal forcing 모델이 아니라 수질 다변량 history 기반 예측 시스템이다.
- 저장 forecast 기준 raw quantile crossing은 거의 없지만, conformal 보정 후 80/90% band crossing이 발생한다. 이는 보정폭이 음수가 될 수 있는 구현 때문에 interval integrity가 깨지는 치명적 운영 리스크다.
- 예측폭과 실제 오차의 상관은 낮거나 음수인 경우가 많다. uncertainty가 event/error에 맞춰 동적으로 커진다는 근거가 약하다.
- 극단/급변 subset에서 NSE, coverage, alert skill이 평균 regime 대비 크게 악화되는 타깃이 있다. 평균 성능만으로 운영 가능성을 주장하기 어렵다.

## Key Quantitative Findings
| target | raw crossing | cal80 crossing | d5 width-error corr | d5 high-error width ratio | worst-5% station NSE | stations below persistence |
|---|---:|---:|---:|---:|---:|---:|
| chl-a | 0.000 | 0.018 | 0.612 | 4.338 | -0.104 | 1 |
| do | 0.000 | 0.025 | 0.435 | 1.860 | -0.006 | 7 |
| tn | 0.000 | 0.017 | 0.432 | 2.100 | 0.139 | 8 |
| toc | 0.000 | 0.028 | 0.409 | 2.387 | 0.020 | 21 |
| tp | 0.000 | 0.029 | 0.467 | 3.365 | -0.430 | 11 |

## Extreme/Event Reliability
| target | subset | n | NSE model | NSE persistence | cov80 raw | cov80 cal | pred/persist corr | pred_delta/obs_delta corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chl-a | actual_top5pct | 941 | -0.131 | -0.230 | 0.667 | 0.723 | 0.950 | -0.046 |
| chl-a | abs_delta_top5pct | 925 | 0.248 | 0.164 | 0.596 | 0.659 | 0.959 | -0.094 |
| do | actual_top5pct | 1232 | -2.507 | -2.742 | 0.682 | 0.763 | 0.933 | -0.033 |
| do | abs_delta_top5pct | 1210 | 0.269 | 0.174 | 0.618 | 0.660 | 0.957 | -0.057 |
| tn | actual_top5pct | 862 | 0.406 | 0.369 | 0.734 | 0.820 | 0.970 | -0.093 |
| tn | abs_delta_top5pct | 840 | 0.650 | 0.657 | 0.479 | 0.581 | 0.990 | -0.100 |
| toc | actual_top5pct | 1197 | -1.121 | -1.110 | 0.681 | 0.718 | 0.861 | -0.006 |
| toc | abs_delta_top5pct | 1168 | 0.202 | 0.180 | 0.535 | 0.585 | 0.940 | -0.025 |
| tp | actual_top5pct | 849 | -1.070 | -1.228 | 0.624 | 0.673 | 0.927 | -0.043 |
| tp | abs_delta_top5pct | 826 | -0.146 | -0.286 | 0.485 | 0.551 | 0.941 | -0.046 |

## Alert Skill Subsets
| target | subset | n | event rate | PR-AUC | F1 | BSS |
|---|---|---:|---:|---:|---:|---:|
| chl-a | all_horizons | 107090 | 0.236 | 0.778 | 0.740 | 0.549 |
| do | all_horizons | 2185 | 0.000 | nan | 0.000 | nan |
| tn | all_horizons | 102715 | 0.294 | 0.702 | 0.796 | 0.442 |
| toc | all_horizons | 144055 | 0.031 | 0.429 | 0.559 | 0.170 |
| tp | all_horizons | 102715 | 0.047 | 0.439 | 0.550 | -0.104 |

## Covariate Ablation Summary
| metric | n | mean delta | median delta | improved count |
|---|---:|---:|---:|---:|
| nse | 4 | -0.0061 | -0.0055 | 1 |
| crps | 4 | 0.0002 | 0.0002 | 2 |
| cov80 | 4 | 0.0256 | 0.0290 | 3 |

## Verdict
현재 증거 기준 Chronos2는 causal environmental forecasting system이 아니라 probabilistic autoregressive model에 가깝다. 운영 배포는 제한적 보조 의사결정/대시보드 수준에서만 가능하며, 고위험 폭우·오염·극단상황 자동 경보 시스템으로는 아직 부적합하다.
