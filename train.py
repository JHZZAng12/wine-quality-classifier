# Reproduce data audit, training, validation selection, and held-out evaluation.
from pathlib import Path
import hashlib
import io
import json
import platform
import warnings
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TASK = 'wine'
HERE = Path(__file__).resolve().parent
SEED = 42
WINE = TASK == 'wine'
URL = ('https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip' if WINE else
       'https://archive.ics.uci.edu/static/public/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip')
FILENAME = 'winequality-red.csv' if WINE else 'online_shoppers_intention.csv'
CATEGORICAL = [] if WINE else ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

def load_data():
    cache = HERE / 'data' / FILENAME
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL, timeout=90) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                member = next(n for n in archive.namelist() if n.endswith(FILENAME))
                cache.write_bytes(archive.read(member))
    df = pd.read_csv(cache, sep=';' if WINE else ',')
    target = 'quality' if WINE else 'Revenue'
    features = [c for c in df.columns if c != target and c != 'PageValues']
    raw_rows = len(df)
    # Remove repeats before splitting; exclude contradictory labels for identical inputs.
    df = df.drop_duplicates(subset=features + [target])
    duplicates_removed = raw_rows - len(df)
    conflicting = df.duplicated(subset=features, keep=False)
    conflicting_removed = int(conflicting.sum())
    df = df.loc[~conflicting].copy()
    X = df[features].copy()
    if WINE:
        y = (df[target] >= 7).astype(int)
    else:
        y = df[target].astype(str).str.lower().map({'true': 1, 'false': 0})
        if y.isna().any():
            raise ValueError('Revenue must contain only True/False.')
        for c in CATEGORICAL:
            X[c] = X[c].astype(str)
    audit = dict(raw_rows=raw_rows, rows=len(X), duplicates_removed=duplicates_removed,
        conflicting_rows_removed=conflicting_removed, missing_cells=int(X.isna().sum().sum()),
        positive_rate=float(y.mean()), features=features,
        source_url=URL, raw_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    return X, y, audit

def split_data(X, y):
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=.25, stratify=ydev, random_state=SEED)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest

def make_pipeline(X, classifier, scale=True):
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    transformers = [('numeric', Pipeline(numeric_steps), numeric)]
    if CATEGORICAL:
        transformers.append(('category', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL))
    return Pipeline([('preprocess', ColumnTransformer(transformers)), ('classifier', classifier)])

def score(y, probability, threshold=.5):
    prediction = (probability >= threshold).astype(int)
    return dict(accuracy=float(accuracy_score(y, prediction)),
        precision=float(precision_score(y, prediction, zero_division=0)),
        recall=float(recall_score(y, prediction, zero_division=0)),
        f1=float(f1_score(y, prediction, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probability)),
        average_precision=float(average_precision_score(y, probability)))

def markdown_table(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)

def train():
    X, y, audit = load_data()
    Xtr, Xv, Xt, ytr, yv, yt = split_data(X, y)
    # EDA uses training rows only, so model design cannot inspect test distributions.
    Xtr.select_dtypes(include='number').describe().to_csv(HERE / 'feature_summary.csv')
    Xtr.select_dtypes(include='number').hist(figsize=(13, 9), bins=25, color='#77334a' if WINE else '#187d92')
    plt.suptitle('Training feature distributions', y=1.01)
    plt.tight_layout(); plt.savefig(HERE / 'feature_distributions.png', dpi=130, bbox_inches='tight'); plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    ytr.value_counts().sort_index().plot.bar(ax=ax, color=['#8295a3', '#b34262'])
    ax.set(xticklabels=['Negative (0)', 'Positive (1)'], ylabel='Training rows', title='Training class balance')
    plt.tight_layout(); fig.savefig(HERE / 'class_balance.png', dpi=130); plt.close(fig)

    candidates = {
        'dummy': make_pipeline(Xtr, DummyClassifier(strategy='prior'), False),
        'logistic_unscaled': make_pipeline(Xtr, LogisticRegression(solver='lbfgs', max_iter=20000, random_state=SEED), False),
        'logistic_scaled': make_pipeline(Xtr, LogisticRegression(solver='lbfgs', max_iter=20000, random_state=SEED), True),
        **{f'tree_depth_{d}': make_pipeline(Xtr, DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=SEED), False) for d in [3, 5, 8]},
        'tree_unrestricted_diagnostic': make_pipeline(Xtr, DecisionTreeClassifier(random_state=SEED), False),
    }
    if not WINE:
        candidates['random_forest'] = make_pipeline(Xtr, RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=SEED, n_jobs=-1), False)
    rows = []
    for name, model in candidates.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(Xtr, ytr)
        record = {'model': name, 'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught)}
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        for label, xx, yy in [('train', Xtr, ytr), ('validation', Xv, yv)]:
            record.update({f'{label}_{k}': v for k, v in score(yy, model.predict_proba(xx)[:, 1]).items()})
        rows.append(record)
    comparison = pd.DataFrame(rows)
    eligible = comparison[~comparison.model.isin(['dummy', 'tree_unrestricted_diagnostic'])]
    best_name = eligible.sort_values('validation_average_precision', ascending=False, kind='stable').iloc[0]['model']
    best = candidates[best_name]
    vp = best.predict_proba(Xv)[:, 1]
    thresholds = pd.DataFrame([{'threshold': float(t), **score(yv, vp, t)} for t in np.arange(.1, .91, .05)])
    threshold = float(thresholds.sort_values(['f1', 'threshold'], ascending=[False, False]).iloc[0]['threshold'])
    thresholds.to_csv(HERE / 'validation_thresholds.csv', index=False)
    # Model and threshold are frozen before any test evaluation.
    for i, name in enumerate(comparison.model):
        for k, value in score(yt, candidates[name].predict_proba(Xt)[:, 1]).items():
            comparison.loc[i, f'test_{k}'] = value
    comparison.to_csv(HERE / 'model_comparison.csv', index=False)
    probability = best.predict_proba(Xt)[:, 1]
    prediction = (probability >= threshold).astype(int)
    cm = confusion_matrix(yt, prediction, labels=[0, 1])
    result = dict(task=TASK, selected_model=best_name, threshold=threshold,
        selection_metric='validation_average_precision', seed=SEED,
        split_sizes=dict(train=len(Xtr), validation=len(Xv), test=len(Xt)),
        test=score(yt, probability, threshold), confusion_matrix=cm.tolist(),
        classification_report=classification_report(yt, prediction, output_dict=True, zero_division=0),
        audit=audit, convergence_warnings=comparison.loc[comparison.convergence_warning, 'model'].tolist(),
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
            pandas=pd.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__, joblib=joblib.__version__))
    (HERE / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'source_row': X.index, 'split': ['train' if idx in Xtr.index else 'validation' if idx in Xv.index else 'test' for idx in X.index], 'target': y}).to_csv(HERE / 'split_manifest.csv', index=False)
    test_rows = Xt.copy()
    test_rows.insert(0, 'source_row', Xt.index)
    test_rows['actual'] = yt
    test_rows['probability'] = probability
    test_rows['predicted'] = prediction
    test_rows['error_type'] = np.where((yt == 0) & (prediction == 1), 'FP', np.where((yt == 1) & (prediction == 0), 'FN', 'correct'))
    errors = test_rows[test_rows.actual != test_rows.predicted].copy()
    errors['confidence'] = np.maximum(errors.probability, 1-errors.probability)
    errors = errors.sort_values('confidence', ascending=False)
    errors.to_csv(HERE / 'misclassified.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{best_name} | threshold={threshold:.2f}')
    fig.tight_layout(); fig.savefig(HERE / 'confusion_matrix.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(yt, probability, ax=ax, name=best_name)
    ax.axhline(yt.mean(), linestyle='--', color='gray', label='Positive prevalence')
    ax.legend(); fig.tight_layout(); fig.savefig(HERE / 'precision_recall.png', dpi=140); plt.close(fig)
    importance = permutation_importance(best, Xv, yv, scoring='average_precision', n_repeats=8, random_state=SEED, n_jobs=1)
    ranking = pd.DataFrame(dict(feature=X.columns, mean_ap_decrease=importance.importances_mean,
        std_ap_decrease=importance.importances_std)).sort_values('mean_ap_decrease', ascending=False)
    ranking.to_csv(HERE / 'feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    top = ranking.head(10).iloc[::-1]
    ax.barh(top.feature, top.mean_ap_decrease, xerr=top.std_ap_decrease, color='#187d92')
    ax.set(xlabel='Validation average precision decrease (8 permutations)', title='Permutation feature importance')
    fig.tight_layout(); fig.savefig(HERE / 'feature_importance.png', dpi=140); plt.close(fig)
    bundle = dict(model=best, threshold=threshold, features=list(X.columns), categorical=CATEGORICAL,
        task=TASK, versions=result['versions'], numeric_ranges={c: [float(Xtr[c].min()), float(Xtr[c].max())] for c in X.columns if c not in CATEGORICAL})
    joblib.dump(bundle, HERE / 'model.joblib', compress=3)
    sample = Xtr.iloc[0].to_dict()
    (HERE / 'sample_input.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(result, comparison, errors, ranking)
    print(json.dumps({k: result[k] for k in ['selected_model', 'threshold', 'split_sizes', 'test']}, indent=2))
    return result

def write_report(r, comparison, errors, ranking):
    a, m = r['audit'], r['test']
    tn, fp, fn, tp = np.array(r['confusion_matrix']).ravel()
    label = '우수 와인(원점수 7 이상)' if WINE else '구매(Revenue=True)'
    table = comparison[['model', 'train_f1', 'validation_f1', 'validation_average_precision', 'test_accuracy', 'test_f1']]
    overfit = comparison[comparison.model == 'tree_unrestricted_diagnostic'].iloc[0]
    insights = ('알코올·산도 등의 중요도는 이 표본에서의 예측 연관성입니다. 성분을 인위적으로 바꿨을 때 품질이 개선된다는 인과 결론은 낼 수 없습니다.' if WINE else
        '상위 특성에 따라 상품 탐색 동선, 상세 페이지 정보, 재방문 경험을 점검하는 실험 가설을 세울 수 있습니다. 체류 시간을 늘리면 구매가 늘어난다는 인과 결론은 아닙니다. 쿠폰 비용·구매 이익 데이터가 없으므로 매출 개선액이나 ROI를 추정하지 않았습니다.')
    if not WINE:
        insights += '\n\n- `Month`: 계절·캠페인 시기별 구매 패턴의 차이를 점검하되, 월이 중요하다는 결과는 시간 변화에 취약하다는 신호일 수도 있습니다. 다음 기간 검증을 우선합니다.\n- `ProductRelated_Duration`: 상품 탐색이 길어지는 구간의 상세 정보·검색 품질을 점검하고 개선안을 A/B 테스트합니다. 중요도만으로 관계의 방향을 해석하지 않습니다.\n- `ExitRates`, `VisitorType`: 이탈이 많은 페이지와 신규·재방문 동선의 차이를 추가 분석합니다. 현재 중요도는 특정 집단의 전환율이 더 높다는 결론을 제공하지 않습니다.'
    note = '' if WINE else '''
## 예측 시점과 누수 검토
`PageValues`는 거래 가치에서 파생된 지표여서 구매 전 사용 가능성을 입증할 수 없습니다. 기본 모델에서 제외했습니다. 이는 명백한 누수라고 단정한 것이 아니라 보수적인 가용성 판단입니다.
체류 시간, BounceRates, ExitRates도 완성된 세션의 집계입니다. 이 프로젝트는 과거 세션의 구매 여부 분류 실습이며, 실시간 전환 예측 성능을 입증하지 않습니다. 실서비스에서는 구매 이전 시점의 이벤트 스냅샷과 시계열 홀드아웃으로 다시 평가해야 합니다. 월·접속 환경 코드에는 원핫 인코딩을 적용합니다.
'''
    report = f'''# {'와인 품질' if WINE else '온라인 쇼핑몰 구매 전환'} 성능 평가서

## 문제와 데이터
양성은 **{label}**입니다. 원본 {a['raw_rows']:,}행에서 입력·타깃이 동일한 중복 {a['duplicates_removed']:,}행, 동일 입력의 상충 타깃 {a['conflicting_rows_removed']:,}행을 제거해 {a['rows']:,}행을 사용했습니다. 결측 셀은 {a['missing_cells']}개, 양성 비율은 {a['positive_rate']:.2%}입니다. 원본 인덱스와 분할은 `split_manifest.csv`, 원본 파일 SHA-256은 `metrics.json`에 기록했습니다.

## 평가 설계
`random_state=42`로 양성 비율을 유지하여 학습/검증/테스트를 60/20/20으로 분리했습니다. 실제 건수는 {r['split_sizes']}입니다. 스케일러·결측 대체·인코더는 Pipeline 안에서 학습 세트로만 적합합니다. EDA도 학습 세트만 사용합니다.
모델은 검증 **Average Precision(AP)** 최대값으로 선택하고, 선택 모델의 임계값은 검증 F1이 최대인 값(0.10~0.90, 간격 0.05)으로 정했습니다. 동률이면 큰 임계값을 사용합니다. AP는 불균형 데이터의 순위 품질을 측정하며, 사다리꼴 PR-AUC와 계산이 다릅니다.
최종 선택은 **{r['selected_model']}**, 임계값은 **{r['threshold']:.2f}**입니다. 테스트 결과는 선택에 사용하지 않았고, 검증 중요도 평가를 위해 최종 모델을 학습+검증으로 재적합하지 않았습니다. 무제한 트리는 과적합 관찰용으로 선택 대상에서 제외했습니다.

## 모델 비교 — 공통 임계값 0.50
{markdown_table(table)}

모든 후보는 같은 분할을 사용합니다. 아래 최종 성능은 조정 임계값을 사용하므로 위 표의 0.50 결과와 다를 수 있습니다. 스케일링 비교는 같은 LogisticRegression 설정에서 수행합니다. 표준화는 각 특성을 학습 평균 0·표준편차 1로 바꾸며, 정규화 제약과 최적화에 영향을 줍니다. 트리는 분기 순서를 사용하므로 스케일링을 생략했습니다.
수렴 경고가 나온 후보: **{', '.join(r['convergence_warnings']) or '없음'}**. 경고가 있는 비표준화 LogisticRegression은 최적화가 완료되지 않은 비교 실험이며, 그 성능을 수렴한 모델의 확정적 한계로 해석하지 않습니다. `model_comparison.csv`의 convergence_warning 열에도 기록했습니다.

## 최종 테스트 성능
| 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} |

혼동 행렬의 행은 실제, 열은 예측이며 순서는 음성(0), 양성(1)입니다: **TN={tn}, FP={fp}, FN={fn}, TP={tp}**.
다수 클래스 기준 모델의 테스트 정확도는 **{comparison.loc[comparison.model == 'dummy', 'test_accuracy'].iloc[0]:.4f}**입니다. 최종 정확도가 이보다 낮아도 양성 탐지 F1과 재현율이 개선될 수 있습니다. 이 선택은 매출이나 비용 최적화를 의미하지 않습니다.
정밀도 TP/(TP+FP)는 양성 예측의 적중률, 재현율 TP/(TP+FN)는 실제 양성을 찾아낸 비율입니다. FP는 {'일반 와인을 우수로 잘못 추천' if WINE else '구매하지 않을 고객에게 캠페인 비용을 사용'}하는 오류, FN은 {'우수 와인을 놓치는' if WINE else '잠재 구매자를 놓치는'} 오류입니다.

![혼동 행렬](confusion_matrix.png)
![정밀도 재현율 곡선](precision_recall.png)

## 과적합과 오답 노트
무제한 트리의 학습 F1={overfit['train_f1']:.4f}, 검증 F1={overfit['validation_f1']:.4f}입니다. 깊이 3·5·8 및 최소 잎 표본 10으로 제한한 트리와 비교해 일반화 차이를 확인합니다.
테스트 오답 {len(errors)}건 전체를 `misclassified.csv`에 저장했습니다. FP/FN, 실제값, 예측값, 확률, 원본 행 번호와 입력 성분/행동을 확인할 수 있습니다. 확신이 큰 오답 5건은 다음과 같습니다.

{markdown_table(errors[['source_row', 'actual', 'predicted', 'probability', 'error_type']].head(5))}

오답은 추가 확인 대상이며, 관측하지 않은 원인을 단정하지 않습니다. 확률은 보정하지 않은 모델 출력입니다. 이번 테스트로 수정 방향을 정했다면 다음 버전에는 새로운 홀드아웃이 필요합니다.

## 주요 특성과 활용
검증 세트 원본 열을 하나씩 섞었을 때 AP 하락량을 8회 측정했습니다. 음수는 중요도가 안정적이지 않을 수 있음을 뜻하며, 상관된 열은 중요도를 나눠 가질 수 있습니다.

{markdown_table(ranking.head(5))}

![특성 중요도](feature_importance.png)
{insights}
{note}
## 한계와 후속 실험
단일 무작위 분할이므로 성능 변동과 신뢰구간을 추정하지 않았습니다. 교차검증·반복 분할, 확률 보정, 비용 기반 임계값 선택을 후속 실험으로 권합니다. {'포르투갈 레드 Vinho Verde 표본에 국한되며 다른 산지·화이트 와인·실제 전문가 평가로의 일반화는 검증하지 않았습니다. 7점 기준은 이 프로젝트의 교육용 정의입니다.' if WINE else '마케팅 실행 전 시간순 검증과 A/B 테스트가 필요합니다. 데이터에는 명확한 이벤트 시점이 없어서 이 저장소에서는 시간순 검증을 수행하지 않았습니다.'}

## 출처와 재현
데이터 출처·저작자·라이선스는 [DATA_SOURCES.md](DATA_SOURCES.md)에 있습니다. `python train.py`로 전체 결과를 재생성합니다. Python 및 라이브러리 버전은 `metrics.json`, 설치 목록은 `requirements-lock.txt`에서 확인합니다.
'''
    (HERE / 'evaluation.md').write_text(report, encoding='utf-8')

if __name__ == '__main__':
    train()
