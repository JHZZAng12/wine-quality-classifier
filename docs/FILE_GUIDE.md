# 파일별 역할과 읽는 순서

어떤 파일부터 열어야 할지 헷갈릴 때 이 문서부터 확인해 주세요. 두 프로젝트는 같은 폴더 구조를 사용합니다.

처음 읽는다면 **README → 분석 보고서 → 실습 노트북** 순서가 좋습니다. 바로 예측해 보고 싶다면 `examples/sample_input.json`의 값을 바꾼 뒤 `python src/predict.py --input examples/sample_input.json`을 실행해 보세요.

## 처음 볼 파일과 실행 환경

| 파일 | 하는 일 |
| --- | --- |
| [README.md](../README.md) | 프로젝트 소개와 실행 방법을 가장 먼저 읽는 문서입니다. |
| [pytest.ini](../pytest.ini) | 테스트 폴더와 소스코드 경로를 pytest에 알려 줍니다. |
| [requirements.txt](../requirements.txt) | 학습·예측·테스트에 직접 필요한 패키지 버전입니다. |
| [requirements-lock.txt](../requirements-lock.txt) | 실험 환경에 설치된 전체 패키지 버전 기록입니다. |
| [.gitattributes](../.gitattributes) | Git에서 줄바꿈을 자동 변환하지 않도록 해 원래 평가 파일의 형식을 유지합니다. |
| [.gitignore](../.gitignore) | 원본 데이터, 가상환경, 캐시 등을 GitHub 업로드에서 제외합니다. |
| [LICENSE](../LICENSE) | 직접 작성한 코드에 적용한 MIT 라이선스입니다. 데이터 이용 조건은 출처 문서에서 따로 확인합니다. |

## 코드

| 파일 | 하는 일 |
| --- | --- |
| [src/train.py](../src/train.py) | 데이터 다운로드·정리부터 학습, 모델 선택, 평가, 결과 저장까지 실행합니다. |
| [src/predict.py](../src/predict.py) | JSON 입력을 확인한 뒤 저장된 모델로 예측합니다. |
| [src/reporting.py](../src/reporting.py) | 한국어 표·그래프 이름과 분석 보고서의 문장을 만듭니다. |
| [src/paths.py](../src/paths.py) | 데이터·모델·평가 결과가 저장되는 위치를 관리합니다. |

## 실습 노트북

| 파일 | 하는 일 |
| --- | --- |
| [notebooks/analysis.ipynb](../notebooks/analysis.ipynb) | 설명과 실행 결과를 보면서 분석을 단계별로 따라가는 실습 파일입니다. |

## 설명 문서

| 파일 | 하는 일 |
| --- | --- |
| [docs/evaluation.md](../docs/evaluation.md) | 모델 비교, 성능 수치, 오답과 한계를 문장으로 설명한 보고서입니다. |
| [docs/FEATURES.md](../docs/FEATURES.md) | 입력 항목의 영문 이름과 한국어 뜻, 입력할 때 확인할 점을 정리했습니다. |
| [docs/DATA_SOURCES.md](../docs/DATA_SOURCES.md) | 원본 데이터의 출처·저작자·이용 조건을 확인하는 문서입니다. |
| [docs/FILE_GUIDE.md](../docs/FILE_GUIDE.md) | 지금 보고 있는 파일 안내입니다. 파일별 역할과 읽는 순서를 찾을 수 있습니다. |

## 학습 모델

| 파일 | 하는 일 |
| --- | --- |
| [models/model.joblib](../models/model.joblib) | 전처리 과정, 학습 모델, 임계값을 함께 저장한 파일입니다. |

## 입력 예시

| 파일 | 하는 일 |
| --- | --- |
| [examples/sample_input.json](../examples/sample_input.json) | 입력 형식을 익히거나 값을 바꿔 예측할 때 사용하는 예제입니다. |

## 평가 수치와 기록

| 파일 | 하는 일 |
| --- | --- |
| [reports/metrics.json](../reports/metrics.json) | 최종 성능, 선택 모델, 임계값, 데이터 점검 결과와 실행 버전입니다. |
| [reports/model_comparison.csv](../reports/model_comparison.csv) | 같은 분할에서 비교한 후보 모델의 학습·검증·테스트 점수입니다. |
| [reports/validation_thresholds.csv](../reports/validation_thresholds.csv) | 검증 자료에서 임계값을 바꿨을 때의 성능 기록입니다. |
| [reports/misclassified.csv](../reports/misclassified.csv) | 테스트에서 틀린 사례의 원본 행 번호, 입력, 정답, 예측과 확률입니다. |
| [reports/split_manifest.csv](../reports/split_manifest.csv) | 원본의 각 기록을 학습·검증·테스트 중 어디에 썼는지 기록했습니다. |
| [reports/feature_importance.csv](../reports/feature_importance.csv) | 검증 자료에서 항목별 순열 중요도를 측정한 수치입니다. |
| [reports/feature_summary.csv](../reports/feature_summary.csv) | 학습 자료의 수치형 입력에 대한 평균·표준편차·분위수 등입니다. |

## 그래프

| 파일 | 하는 일 |
| --- | --- |
| [figures/class_balance.png](../figures/class_balance.png) | 학습 자료에서 두 정답 유형의 개수를 비교한 그림입니다. |
| [figures/feature_distributions.png](../figures/feature_distributions.png) | 학습 자료의 입력값이 어떤 범위에 모여 있는지 보여 줍니다. |
| [figures/confusion_matrix.png](../figures/confusion_matrix.png) | 실제 결과와 예측을 비교해 맞힌 경우와 틀린 경우를 보여 줍니다. |
| [figures/precision_recall.png](../figures/precision_recall.png) | 임계값에 따라 정밀도와 재현율이 달라지는 모습을 보여 줍니다. |
| [figures/feature_importance.png](../figures/feature_importance.png) | 모델이 주로 참고한 항목의 중요도를 비교한 그림입니다. |

## 테스트

| 파일 | 하는 일 |
| --- | --- |
| [tests/test_project.py](../tests/test_project.py) | 예측 결과·입력 검증·분할 간 중복·학습 전처리 범위를 확인합니다. |

## 실행하면서 만들어지는 폴더

`data/`에는 UCI에서 내려받은 원본 CSV가 들어갑니다. 가상환경과 캐시처럼 GitHub에는 올리지 않습니다. `python src/train.py`를 실행하면 그래프는 `figures/`, 평가 기록은 `reports/`, 모델은 `models/`, 보고서는 `docs/`에 저장됩니다.
