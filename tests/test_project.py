import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from predict import predict
from train import load_data, split_data, CATEGORICAL
from paths import REPORTS_DIR, MODELS_DIR, EXAMPLES_DIR

@pytest.fixture(scope='module')
def bundle():
    return joblib.load(MODELS_DIR/'model.joblib')

@pytest.fixture
def sample():
    return json.loads((EXAMPLES_DIR/'sample_input.json').read_text(encoding='utf-8'))

def test_inference_matches_saved_pipeline(bundle, sample):
    result = predict(sample, bundle)
    row = dict(sample)
    for c in CATEGORICAL:
        row[c] = str(row[c])
    p = bundle['model'].predict_proba(pd.DataFrame([row], columns=bundle['features']))[0,1]
    assert result['positive_probability'] == pytest.approx(p)
    assert result['prediction'] == int(p >= bundle['threshold'])

@pytest.mark.parametrize('kind', ['missing', 'extra', 'nan', 'negative', 'text'])
def test_bad_input_rejected(bundle, sample, kind):
    numeric = next(c for c in bundle['features'] if c not in CATEGORICAL)
    if kind == 'missing': sample.pop(numeric)
    elif kind == 'extra': sample['unknown'] = 3
    elif kind == 'nan': sample[numeric] = float('nan')
    elif kind == 'negative': sample[numeric] = -1
    else: sample[numeric] = 'not a number'
    with pytest.raises(ValueError): predict(sample, bundle)

def test_no_feature_overlap_between_splits():
    X, y, _ = load_data()
    a, b, c, *_ = split_data(X, y)
    sets = [set(pd.util.hash_pandas_object(xx, index=False)) for xx in [a,b,c]]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]

def test_confusion_counts_and_manifest():
    r = json.loads((REPORTS_DIR/'metrics.json').read_text(encoding='utf-8'))
    manifest = pd.read_csv(REPORTS_DIR/'split_manifest.csv')
    assert manifest.source_row.is_unique
    assert int(np.array(r['confusion_matrix']).sum()) == r['split_sizes']['test']
    assert manifest.groupby('split').size().to_dict() == r['split_sizes']

def test_preprocessing_fits_training_only(bundle):
    X,y,_ = load_data()
    Xtr,*_ = split_data(X,y)
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    pre = bundle['model'].named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(pre.named_steps['impute'].statistics_, Xtr[numeric].median())
    if 'scale' in pre.named_steps:
        np.testing.assert_allclose(pre.named_steps['scale'].mean_, Xtr[numeric].mean())
