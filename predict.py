"""Validate a JSON feature row and apply the saved preprocessing/model pipeline."""
import argparse
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def predict(payload, bundle):
    if not isinstance(payload, dict):
        raise ValueError('Input must be a JSON object with feature names.')
    expected = set(bundle['features'])
    if set(payload) != expected:
        raise ValueError(f'Missing: {sorted(expected-set(payload))}; unexpected: {sorted(set(payload)-expected)}')
    row = dict(payload)
    outside = []
    for c in bundle['features']:
        if row[c] is None:
            raise ValueError(f'{c}: null is not accepted in inference input.')
        if c in bundle['categorical']:
            if not isinstance(row[c], (str, int, bool)):
                raise ValueError(f'{c}: expected string/integer/boolean category.')
            row[c] = str(row[c])
        else:
            if isinstance(row[c], bool) or not isinstance(row[c], (int, float)):
                raise ValueError(f'{c}: expected numeric value.')
            row[c] = float(row[c])
            if not np.isfinite(row[c]) or row[c] < 0:
                raise ValueError(f'{c}: expected finite nonnegative number.')
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
    parser.add_argument('--input', type=Path, default=HERE/'sample_input.json')
    args = parser.parse_args()
    bundle = joblib.load(HERE/'model.joblib')
    try:
        output = predict(json.loads(args.input.read_text(encoding='utf-8-sig')), bundle)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
