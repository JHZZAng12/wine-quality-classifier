"""JSON 파일의 입력값을 확인한 뒤 저장된 모델로 예측합니다."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from paths import MODELS_DIR, EXAMPLES_DIR

def predict(payload, bundle):
    if not isinstance(payload, dict):
        raise ValueError('입력은 항목 이름과 값이 짝을 이루는 JSON 객체여야 합니다.')
    expected = set(bundle['features'])
    if set(payload) != expected:
        raise ValueError(f'빠진 항목: {sorted(expected-set(payload))}; 예제에 없는 항목: {sorted(set(payload)-expected)}')
    row = dict(payload)
    outside = []
    for c in bundle['features']:
        if row[c] is None:
            raise ValueError(f'{c}: 값이 비어 있습니다. 예제처럼 값을 입력해 주세요.')
        if c in bundle['categorical']:
            if not isinstance(row[c], (str, int, bool)):
                raise ValueError(f'{c}: 문자열, 정수 또는 참·거짓으로 범주를 입력해 주세요.')
            row[c] = str(row[c])
        else:
            if isinstance(row[c], bool) or not isinstance(row[c], (int, float)):
                raise ValueError(f'{c}: 숫자로 입력해 주세요.')
            row[c] = float(row[c])
            if not np.isfinite(row[c]) or row[c] < 0:
                raise ValueError(f'{c}: 0 이상의 유한한 숫자를 입력해 주세요.')
            lo, hi = bundle['numeric_ranges'][c]
            if row[c] < lo or row[c] > hi:
                outside.append(c)
    probability = float(bundle['model'].predict_proba(pd.DataFrame([row], columns=bundle['features']))[0, 1])
    positive = probability >= bundle['threshold']
    label = ('우수' if positive else '일반') if bundle['task'] == 'wine' else ('구매 예상' if positive else '미구매 예상')
    return dict(label=label, prediction=int(positive), positive_probability=probability,
        threshold=bundle['threshold'], outside_training_range=outside)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=EXAMPLES_DIR/'sample_input.json')
    args = parser.parse_args()
    bundle = joblib.load(MODELS_DIR/'model.joblib')
    try:
        output = predict(json.loads(args.input.read_text(encoding='utf-8-sig')), bundle)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
