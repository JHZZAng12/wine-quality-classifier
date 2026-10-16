# 소믈리에가 되어라! 화학 성분 기반 와인 품질 판별

Python · scikit-learn · 이진 분류 · 재현 가능한 학습 파이프라인

11개 물리화학 성분으로 레드 와인이 우수한 품질인지 예측합니다. `quality >= 7`을 우수(1), 나머지를 일반(0)으로 정의합니다.

## 먼저 볼 자료
- **[성능 평가서 및 오답 노트](evaluation.md)**: 실제 실험 결과, 모델 비교, 혼동 행렬, 한계
- **[분석 노트북](analysis.ipynb)**: 데이터 확인부터 재학습, 성능 읽기, 예측까지
- **[데이터 출처](DATA_SOURCES.md)**: 원저작자, UCI 다운로드, CC BY 4.0

## 실험 결과
선택 모델: **tree_depth_5** · 임계값 **0.30** · 테스트 272건

| 정확도 | 정밀도 | 재현율 | F1 | AP |
| --- | --- | --- | --- | --- |
| 87.50% | 53.66% | 59.46% | 0.5641 | 0.4453 |

![혼동 행렬](confusion_matrix.png)

## 학습 목표와 구현
StandardScaler 전후 LogisticRegression 비교, DecisionTreeClassifier, 정확도·정밀도·재현율·F1·혼동 행렬을 학습합니다.
학습/검증/테스트를 분리하고, 전처리는 학습 데이터로만 적합합니다. 정확도 외에 불균형에 적합한 AP를 사용하며, 모델과 임계값은 검증 세트에서 선택합니다.

## 빠른 실행
Python **3.12** 기준입니다. 저장소 루트에서 실행합니다.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py --input sample_input.json
python train.py
python -m pytest -q
```

`model.joblib`이 포함되어 있어 예측은 데이터 다운로드 없이 실행됩니다. 재학습 시 UCI 데이터에 접근할 인터넷 연결이 필요합니다. 모델에는 전처리와 임계값이 함께 들어 있습니다. Jupyter에서 노트북을 실행하려면 `python -m pip install notebook ipykernel`을 추가로 실행합니다.

## 입력과 출력
`sample_input.json`을 복사해 값을 바꿉니다. 입력 열 이름은 예제와 정확히 일치해야 하며, 추가/누락 열·비수치·음수·NaN은 거부합니다. 범주 코드는 문자열로 정규화하고, 학습에 없던 범주는 OneHotEncoder가 처리합니다. 학습 수치 범위를 벗어나는 열은 출력에 표시합니다.
출력에는 `label`, `prediction`, `positive_probability`, `threshold`, `outside_training_range`가 포함됩니다. 확률은 보정되지 않은 모델 출력입니다.

## 파일 안내
| 파일 | 용도 |
| --- | --- |
| `train.py`, `predict.py` | 데이터 다운로드·학습·평가 / JSON 입력 추론 |
| `analysis.ipynb` | 단계별 실습 |
| `model.joblib`, `sample_input.json` | 학습된 모델 / 입력 예시 |
| `evaluation.md`, `metrics.json` | 해설 / 기계가 읽는 실험 결과 |
| `model_comparison.csv`, `validation_thresholds.csv` | 후보 비교 / 검증 임계값 실험 |
| `misclassified.csv`, `split_manifest.csv` | 오답 전체 / 분할 재현 |
| `feature_importance.csv`, `feature_summary.csv`, `*.png` | 중요도·통계·시각화 |
| `test_project.py` | 추론·누수 방지·입력 검증 테스트 |
| `requirements.txt`, `requirements-lock.txt` | 직접 의존성 / 실험 패키지 버전 |

저장 모델은 이 저장소의 신뢰할 수 있는 파일만 로드하세요. joblib은 Python 객체를 역직렬화합니다. 코드 라이선스는 MIT이며 원본 데이터와 그 파생 데이터는 별도의 CC BY 4.0 출처 고지를 따릅니다.
