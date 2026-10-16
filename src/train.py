# 데이터 정리부터 학습, 검증, 최종 평가까지 한 번에 실행합니다.
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
from reporting import FEATURE_LABELS, MODEL_LABELS, configure_plot_font
configure_plot_font()
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
from paths import ROOT, DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, EXAMPLES_DIR, DOCS_DIR, ensure_output_dirs
HERE = ROOT
SEED = 42
WINE = TASK == 'wine'
URL = ('https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip' if WINE else
       'https://archive.ics.uci.edu/static/public/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset.zip')
FILENAME = 'winequality-red.csv' if WINE else 'online_shoppers_intention.csv'
CATEGORICAL = [] if WINE else ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

def load_data():
    cache = DATA_DIR / FILENAME
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
    # 같은 기록이 학습과 테스트에 나눠 들어가지 않도록 중복을 먼저 정리합니다.
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
            raise ValueError('Revenue에는 True 또는 False만 들어 있어야 합니다.')
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
    ensure_output_dirs()
    X, y, audit = load_data()
    Xtr, Xv, Xt, ytr, yv, yt = split_data(X, y)
    # 분포를 살펴볼 때도 학습 자료만 사용합니다.
    Xtr.select_dtypes(include='number').describe().to_csv(REPORTS_DIR / 'feature_summary.csv')
    Xtr.select_dtypes(include='number').rename(columns=FEATURE_LABELS).hist(figsize=(13, 9), bins=25, color='#77334a' if WINE else '#187d92')
    plt.suptitle('학습 자료의 입력값은 어떻게 분포할까요?', y=1.01)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=130, bbox_inches='tight'); plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    ytr.value_counts().sort_index().plot.bar(ax=ax, color=['#8295a3', '#b34262'])
    ax.set(xticklabels=['일반 (0)', '우수 (1)'] if WINE else ['미구매 (0)', '구매 (1)'], ylabel='학습 자료 수', title='찾으려는 대상이 얼마나 있을까요?')
    plt.tight_layout(); fig.savefig(FIGURES_DIR / 'class_balance.png', dpi=130); plt.close(fig)

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
    thresholds.to_csv(REPORTS_DIR / 'validation_thresholds.csv', index=False)
    # 모델과 임계값을 정한 뒤에야 테스트 성능을 확인합니다.
    for i, name in enumerate(comparison.model):
        for k, value in score(yt, candidates[name].predict_proba(Xt)[:, 1]).items():
            comparison.loc[i, f'test_{k}'] = value
    comparison.to_csv(REPORTS_DIR / 'model_comparison.csv', index=False)
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
    (REPORTS_DIR / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'source_row': X.index, 'split': ['train' if idx in Xtr.index else 'validation' if idx in Xv.index else 'test' for idx in X.index], 'target': y}).to_csv(REPORTS_DIR / 'split_manifest.csv', index=False)
    test_rows = Xt.copy()
    test_rows.insert(0, 'source_row', Xt.index)
    test_rows['actual'] = yt
    test_rows['probability'] = probability
    test_rows['predicted'] = prediction
    test_rows['error_type'] = np.where((yt == 0) & (prediction == 1), 'FP', np.where((yt == 1) & (prediction == 0), 'FN', 'correct'))
    errors = test_rows[test_rows.actual != test_rows.predicted].copy()
    errors['confidence'] = np.maximum(errors.probability, 1-errors.probability)
    errors = errors.sort_values('confidence', ascending=False)
    errors.to_csv(REPORTS_DIR / 'misclassified.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['일반', '우수'] if WINE else ['미구매', '구매']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set(title=f'{MODEL_LABELS[best_name]} · 임계값 {threshold:.2f}', xlabel='모델의 예측', ylabel='실제 결과')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(yt, probability, ax=ax, name=MODEL_LABELS[best_name])
    ax.set(xlabel='재현율 · 실제 대상을 찾아낸 비율', ylabel='정밀도 · 대상이라고 예측한 결과의 적중률', title='판단 기준을 바꾸면 결과도 달라집니다')
    ax.axhline(yt.mean(), linestyle='--', color='gray', label='테스트 자료에서 대상이 차지하는 비율')
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / 'precision_recall.png', dpi=140); plt.close(fig)
    importance = permutation_importance(best, Xv, yv, scoring='average_precision', n_repeats=8, random_state=SEED, n_jobs=1)
    ranking = pd.DataFrame(dict(feature=X.columns, mean_ap_decrease=importance.importances_mean,
        std_ap_decrease=importance.importances_std)).sort_values('mean_ap_decrease', ascending=False)
    ranking.to_csv(REPORTS_DIR / 'feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    top = ranking.head(10).iloc[::-1]
    ax.barh(top.feature.map(lambda c: FEATURE_LABELS.get(c, c)), top.mean_ap_decrease, xerr=top.std_ap_decrease, color='#187d92')
    ax.set(xlabel='항목을 섞었을 때 줄어든 검증 AP · 8회 평균', title='모델이 주로 참고한 항목')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'feature_importance.png', dpi=140); plt.close(fig)
    bundle = dict(model=best, threshold=threshold, features=list(X.columns), categorical=CATEGORICAL,
        task=TASK, versions=result['versions'], numeric_ranges={c: [float(Xtr[c].min()), float(Xtr[c].max())] for c in X.columns if c not in CATEGORICAL})
    joblib.dump(bundle, MODELS_DIR / 'model.joblib', compress=3)
    sample = Xtr.iloc[0].to_dict()
    (EXAMPLES_DIR / 'sample_input.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(result, comparison, errors, ranking)
    print(f'학습과 평가를 마쳤습니다. 선택한 모델: {MODEL_LABELS[best_name]} / 임계값: {threshold:.2f}')
    print('자세한 수치와 해석은 docs/evaluation.md에서 확인할 수 있습니다.')
    return result

def write_report(r, comparison, errors, ranking):
    from reporting import write_report as write_korean_report
    write_korean_report(DOCS_DIR, r, comparison, errors, ranking)

if __name__ == '__main__':
    train()
