# 소믈리에가 되어라! 성분표로 알아보는 와인 품질

이 프로젝트에서는 알코올 도수, 산도, 당도 같은 성분으로 레드 와인의 품질을 구분해 봤습니다. 와인마다 주어진 11개 성분을 모델에 입력하고, 품질 점수가 **7점 이상이면 우수**, 나머지는 일반으로 분류했습니다. 여기서 7점은 실습을 위해 정한 기준입니다.

## 결과부터 살펴보면

272개의 와인으로 확인했을 때 정확도는 87.50%였습니다. 다만 일반 와인이 많은 데이터여서 이 숫자만 보기는 어렵습니다. 실제 우수 와인을 찾아낸 비율은 59.46%, 우수하다고 추천한 와인이 실제로 우수했던 비율은 53.66%였습니다.

검증 자료에서 고른 모델은 **의사결정나무 · 깊이 5**, 최종 판단에 쓴 임계값은 **0.30**입니다. 임계값은 예측 확률이 얼마 이상일 때 우수 와인으로 판단할지를 정하는 기준입니다.

| 정확도 | 정밀도 | 재현율 | F1 | AP |
| --- | --- | --- | --- | --- |
| 87.50% | 53.66% | 59.46% | 0.5641 | 0.4453 |

![실제 결과와 예측 결과를 비교한 혼동 행렬](figures/confusion_matrix.png)

실험 조건과 수치의 해석, 틀린 사례는 **[분석 보고서](docs/evaluation.md)**에 자세히 적었습니다.

## 이 과정에서 다룬 내용

성분마다 숫자 범위가 다르다는 점에서 시작했습니다. `StandardScaler`로 범위를 맞추기 전후에 로지스틱 회귀(`LogisticRegression`)의 결과가 어떻게 달라지는지 비교했습니다. 의사결정나무(`DecisionTreeClassifier`)도 함께 학습하고, 정확도뿐 아니라 정밀도·재현율·F1과 혼동 행렬로 틀린 예측까지 살펴봤습니다.

학습·검증·테스트 자료는 따로 나눴습니다. 학습 자료로 모델을 만들고 검증 자료로 모델과 임계값을 고른 뒤, 테스트 자료는 마지막 확인에만 사용했습니다. 일반 사례가 많다는 점을 고려해, 모델을 고를 때는 정확도 대신 AP를 기준으로 삼았습니다.

## 파일을 역할별로 나눴습니다

```text
프로젝트/
├── README.md                # 소개와 실행 방법
├── src/                     # 학습·예측·보고서 생성 코드
├── notebooks/               # 단계별 실습 노트북
├── docs/                    # 분석 보고서와 설명 문서
├── models/                  # 학습된 모델
├── examples/                # 예측 입력 예시
├── reports/                 # 평가 수치와 오답 CSV
├── figures/                 # 분석 그래프
├── tests/                   # 실행 검증 코드
├── pytest.ini               # 테스트 설정
├── requirements.txt         # 필요한 패키지
├── requirements-lock.txt    # 전체 패키지 버전
├── .gitattributes           # 파일의 줄바꿈 형식 유지
├── .gitignore               # Git에 올리지 않을 항목
└── LICENSE                  # 코드 이용 조건
```

**[파일별 역할과 읽는 순서](docs/FILE_GUIDE.md)**에서 각 파일의 용도를 확인할 수 있습니다. 원본 데이터는 학습할 때 `data/`에 내려받으며, GitHub에는 올리지 않습니다.

## 자료를 읽는 순서

1. **[분석 보고서](docs/evaluation.md)**에서 전체 흐름과 결과를 먼저 읽어 보세요.
2. **[실습 노트북](notebooks/analysis.ipynb)**에서는 데이터를 불러오는 단계부터 예측까지 직접 따라갈 수 있습니다. 실행 결과도 함께 저장했습니다.
3. **[입력 항목 안내](docs/FEATURES.md)**와 **[데이터 출처](docs/DATA_SOURCES.md)**에서 각 열의 뜻과 이용 조건을 확인할 수 있습니다.

## 내 데이터로 예측해 보기

Python **3.12**를 기준으로 실행했습니다. 먼저 이 저장소를 내려받고, 파일이 들어 있는 폴더에서 가상환경을 만들어 주세요.

```bash
python -m venv .venv
```

Windows PowerShell에서는 다음 명령으로 가상환경을 켭니다.

```powershell
.venv\Scripts\Activate.ps1
```

macOS나 Linux에서는 `source .venv/bin/activate`를 실행하면 됩니다. 필요한 패키지를 설치한 뒤 예제 입력으로 예측해 보세요.

```bash
python -m pip install -r requirements.txt
python src/predict.py --input examples/sample_input.json
```

`examples/sample_input.json`을 복사해서 성분 값을 바꾸면 다른 입력도 시험해 볼 수 있습니다. 학습한 모델이 `models/model.joblib`에 들어 있어서 **예측만 할 때는 데이터를 다시 내려받지 않아도 됩니다.** 입력 항목 이름은 예제와 같아야 합니다. 빠진 항목이나 잘못된 숫자가 있으면 어떤 부분을 고쳐야 하는지 안내합니다.

출력의 `label`은 예측 결과이고, `positive_probability`는 우수 와인일 것으로 모델이 계산한 확률입니다. `outside_training_range`에는 학습 때 보았던 수치 범위를 벗어난 항목이 표시됩니다. 확률은 별도로 보정하지 않은 값이므로 확정적인 판단으로 받아들이지는 마세요.

## 처음부터 다시 학습해 보기

```bash
python src/train.py
python -m pytest -q
```

재학습할 때는 공식 UCI 사이트에서 데이터를 받기 때문에 인터넷 연결이 필요합니다. 보고서·차트·모델 파일도 함께 다시 만듭니다. 노트북으로 따라가려면 `python -m pip install notebook ipykernel`을 추가로 실행해 주세요.

그래프를 새로 그릴 때는 맑은 고딕, AppleGothic, 나눔고딕 또는 Noto Sans CJK KR 중 설치된 글꼴을 사용합니다. 글자가 네모로 보인다면 한글 글꼴을 설치한 뒤 다시 실행해 주세요. 저장소에 올린 그래프는 한글 글꼴로 생성했습니다.

## 파일을 찾을 때

| 보고 싶은 내용 | 파일 |
| --- | --- |
| 분석 흐름과 결과 해석 | `docs/evaluation.md`, `notebooks/analysis.ipynb` |
| 학습 코드와 한국어 보고서 생성 코드 | `src/train.py`, `src/reporting.py` |
| 입력값으로 예측하는 코드 | `src/predict.py`, `examples/sample_input.json` |
| 학습된 모델 | `models/model.joblib` |
| 성능 수치와 모델 비교 | `reports/metrics.json`, `reports/model_comparison.csv` |
| 틀린 사례와 데이터 분할 기록 | `reports/misclassified.csv`, `reports/split_manifest.csv` |
| 입력 항목의 통계와 중요도 | `reports/feature_summary.csv`, `reports/feature_importance.csv`, `*.png` |
| 임계값을 바꿔 본 기록 | `reports/validation_thresholds.csv` |
| 실행 검증과 패키지 버전 | `tests/test_project.py`, `requirements.txt`, `requirements-lock.txt` |

CSV·JSON의 영문 열 이름은 코드에서 읽는 이름이므로 그대로 두었습니다. 보고서와 노트북의 표에는 한국어 설명을 붙였습니다. 코드는 MIT 라이선스이며, 데이터와 그 파생 결과는 [출처 안내](docs/DATA_SOURCES.md)의 CC BY 4.0 조건을 따릅니다. `joblib` 파일은 신뢰할 수 있는 출처의 것만 불러와 주세요.
