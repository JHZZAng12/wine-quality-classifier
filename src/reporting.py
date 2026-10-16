"""분석 결과를 읽기 쉬운 한국어 보고서로 정리합니다."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    'fixed acidity': '고정 산도', 'volatile acidity': '휘발성 산도',
    'citric acid': '구연산', 'residual sugar': '잔당', 'chlorides': '염화물',
    'free sulfur dioxide': '유리 이산화황', 'total sulfur dioxide': '총 이산화황',
    'density': '밀도', 'pH': 'pH', 'sulphates': '황산염', 'alcohol': '알코올 도수',
    'Administrative': '관리 페이지 방문 수', 'Administrative_Duration': '관리 페이지 체류 시간',
    'Informational': '정보 페이지 방문 수', 'Informational_Duration': '정보 페이지 체류 시간',
    'ProductRelated': '상품 페이지 방문 수', 'ProductRelated_Duration': '상품 페이지 체류 시간',
    'BounceRates': '반송률', 'ExitRates': '이탈률', 'SpecialDay': '특별한 날과의 근접도',
    'Month': '방문 월', 'OperatingSystems': '운영체제', 'Browser': '브라우저',
    'Region': '지역', 'TrafficType': '유입 경로', 'VisitorType': '방문자 유형', 'Weekend': '주말 여부',
}
MODEL_LABELS = {
    'dummy': '다수 클래스만 예측하는 기준 모델',
    'logistic_unscaled': '로지스틱 회귀 · 표준화 전',
    'logistic_scaled': '로지스틱 회귀 · 표준화 후',
    'tree_depth_3': '의사결정나무 · 깊이 3', 'tree_depth_5': '의사결정나무 · 깊이 5',
    'tree_depth_8': '의사결정나무 · 깊이 8',
    'tree_unrestricted_diagnostic': '깊이 제한 없는 나무 · 비교용',
    'random_forest': '랜덤 포레스트',
}
METRIC_LABELS = {'accuracy':'정확도', 'precision':'정밀도', 'recall':'재현율', 'f1':'F1',
    'roc_auc':'ROC-AUC', 'average_precision':'AP'}

def configure_plot_font():
    """설치된 한글 글꼴을 사용합니다. 운영체제에 따라 글꼴 이름은 달라집니다."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    else:
        import warnings
        warnings.warn('한글 글꼴을 찾지 못했습니다. 그래프의 글자가 깨지면 나눔고딕 또는 Noto Sans CJK KR을 설치해 주세요.')
    plt.rcParams['axes.unicode_minus'] = False

def markdown_table(frame):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)

def comparison_for_readers(frame):
    table = frame[['model','train_f1','validation_f1','validation_average_precision','test_accuracy','test_f1']].copy()
    table['model'] = table.model.map(MODEL_LABELS)
    return table.rename(columns={'model':'모델', 'train_f1':'학습 F1', 'validation_f1':'검증 F1',
        'validation_average_precision':'검증 AP', 'test_accuracy':'테스트 정확도', 'test_f1':'테스트 F1'})

def importance_for_readers(ranking):
    table = ranking.copy()
    table['feature'] = table.feature.map(lambda name: f'{FEATURE_LABELS.get(name, name)} ({name})')
    return table.rename(columns={'feature':'입력 항목', 'mean_ap_decrease':'AP 감소량 평균', 'std_ap_decrease':'AP 감소량 표준편차'})

def errors_for_readers(errors, wine):
    table = errors[['source_row','actual','predicted','probability','error_type']].copy()
    labels = {0:'일반',1:'우수'} if wine else {0:'미구매',1:'구매'}
    for column in ['actual','predicted']:
        table[column] = table[column].map(labels)
    table['error_type'] = table.error_type.map({'FP':'잘못 추천 (FP)' if wine else '구매로 잘못 예측 (FP)',
        'FN':'우수 와인을 놓침 (FN)' if wine else '구매 고객을 놓침 (FN)'})
    return table.rename(columns={'source_row':'원본 행 번호', 'actual':'실제 결과', 'predicted':'예측 결과',
        'probability':'우수일 확률' if wine else '구매 확률', 'error_type':'틀린 유형'})

def write_report(here, r, comparison, errors, ranking):
    here = Path(here)
    wine = r['task'] == 'wine'
    a, m, split = r['audit'], r['test'], r['split_sizes']
    tn, fp, fn, tp = np.array(r['confusion_matrix']).ravel()
    selected = MODEL_LABELS[r['selected_model']]
    overfit = comparison[comparison.model == 'tree_unrestricted_diagnostic'].iloc[0]
    base = comparison[comparison.model == 'dummy'].iloc[0]
    scaled = comparison[comparison.model == 'logistic_scaled'].iloc[0]
    raw = comparison[comparison.model == 'logistic_unscaled'].iloc[0]
    first = ('와인을 직접 맛보지 않고도 성분표로 품질을 어느 정도 구분할 수 있는지 살펴봤습니다. 알코올 도수와 산도 등 11개 성분을 입력하고, 품질 점수가 **7점 이상이면 우수**, 그보다 낮으면 일반으로 나눴습니다. 7점은 이 실습에서 정한 기준입니다.' if wine else
        '쇼핑몰에 들어온 사람이 실제로 구매할지, 방문 기록에서 단서를 찾을 수 있는지 살펴봤습니다. 페이지를 얼마나 봤는지, 얼마나 머물렀는지, 어떤 환경에서 접속했는지를 입력하고 **구매 여부**를 예측했습니다. 이 보고서의 결과는 이미 끝난 방문 기록을 분석한 결과입니다.')
    outcome = (f'최종 모델은 **{selected}**였습니다. 한 번도 학습에 쓰지 않은 와인 {split["test"]:,}개 중 {tn+tp:,}개를 맞혀 정확도는 **{m["accuracy"]:.2%}**였습니다. 다만 우수 와인을 실제로 얼마나 찾아냈는지까지 보면, 재현율은 **{m["recall"]:.2%}**였습니다.' if wine else
        f'최종 모델은 **{selected}**였습니다. 테스트에서 실제 구매 고객 {fn+tp:,}명 중 {tp:,}명을 찾아 재현율은 **{m["recall"]:.2%}**였습니다. 반면 구매할 것으로 예측한 {fp+tp:,}명 중 실제 구매자는 {tp:,}명이라, 정밀도는 **{m["precision"]:.2%}**였습니다. 구매 고객을 더 찾는 대신 잘못 짚는 경우도 많은 결과입니다.')
    positive = '우수 와인' if wine else '구매한 고객'
    data_note = ('품종을 구분하는 `sklearn.datasets.load_wine` 대신 UCI의 레드 와인 품질 데이터를 사용했습니다.' if wine else
        '`Revenue=True`를 구매로 표시했습니다. 접속 환경 코드와 방문 월 등은 숫자 크기에 의미가 없어서 원핫 인코딩으로 바꿨습니다. `PageValues`를 제외한 이유는 아래에서 따로 설명합니다.')
    conflict_note = (f'입력이 같은데 정답이 서로 다른 {a["conflicting_rows_removed"]:,}행도 제외했습니다.' if a['conflicting_rows_removed'] else
        '같은 입력에 서로 다른 정답이 붙어 있는 사례는 중복 정리 후 발견되지 않았습니다.')
    warning = ('이번 실험에서 수렴 경고는 나오지 않았습니다.' if not r.get('convergence_warnings') else
        '표준화하지 않은 로지스틱 회귀에서는 수렴 경고가 나왔습니다. 반복 계산이 충분히 마무리되지 않았다는 뜻입니다. 따라서 이 점수만으로 모델 자체의 한계를 단정하기는 어렵습니다. 경고 여부는 `reports/model_comparison.csv`에도 남겼습니다.')
    scale_note = (f'로지스틱 회귀의 검증 F1은 표준화 전 {raw["validation_f1"]:.4f}, 표준화 후 {scaled["validation_f1"]:.4f}였습니다. 다만 검증 AP는 {raw["validation_average_precision"]:.4f}에서 {scaled["validation_average_precision"]:.4f}로 달라졌습니다. 이번처럼 지표에 따라 변화 방향이 다를 수 있으므로 표준화가 모든 점수를 높인다고 해석하면 안 됩니다.' if wine else
        f'로지스틱 회귀의 검증 AP는 표준화 전 {raw["validation_average_precision"]:.4f}, 표준화 후 {scaled["validation_average_precision"]:.4f}였습니다. 수치 범위를 맞춘 뒤 구매 고객의 순서를 구분하는 성능은 개선됐습니다. 다만 아래 수렴 경고도 함께 고려해야 합니다.')
    baseline_note = (f'모든 와인을 일반으로만 분류해도 정확도는 {base["test_accuracy"]:.2%}입니다. 일반 와인이 훨씬 많기 때문입니다. 그래서 {m["accuracy"]:.2%}라는 정확도와 함께 우수 와인에 대한 정밀도·재현율을 살펴봤습니다.' if wine else
        f'모두 미구매로 예측하면 정확도는 {base["test_accuracy"]:.2%}로, 최종 모델의 {m["accuracy"]:.2%}보다 높습니다. 하지만 그렇게 하면 구매 고객을 한 명도 찾지 못합니다. 이번에는 검증 F1을 기준으로 구매를 판단하는 문턱을 낮춰 재현율을 높였습니다. 어느 쪽이 사업에 유리한지는 캠페인 비용과 구매 이익까지 알아야 판단할 수 있습니다.')
    counts = (f'우수 와인 {tp+fn}개 가운데 {tp}개를 찾았고, {fn}개는 놓쳤습니다. 일반 와인 {tn+fp}개 중에서는 {fp}개를 우수하다고 잘못 추천했습니다.' if wine else
        f'실제 구매 고객 {tp+fn}명 중 {tp}명을 찾았고, {fn}명은 놓쳤습니다. 구매하지 않은 고객 {tn+fp:,}명 중에서는 {fp}명을 구매할 것으로 잘못 예측했습니다.')
    implications = ('''알코올 도수, 황산염, pH가 상위에 나타났습니다. 이 항목을 섞었을 때 모델 성능이 크게 떨어졌다는 뜻입니다. 다만 이 결과만으로 “도수를 높이면 품질이 좋아진다”거나 “황산염을 늘리면 된다”고 말할 수는 없습니다. 성분 사이의 관계와 관능 평가를 더 살펴봐야 합니다.''' if wine else
        '''방문 월, 상품 페이지 체류 시간, 이탈률이 상위에 나타났습니다. 이를 바탕으로 다음 분석과 실험을 생각해 볼 수 있습니다.

- **방문 월:** 계절이나 캠페인 시기에 따라 패턴이 달라졌을 수 있습니다. 먼저 다른 기간에도 성능이 유지되는지 확인하는 편이 좋겠습니다.
- **상품 페이지 체류 시간:** 오래 머문 이유가 관심인지, 정보를 찾기 어려워서인지 나눠 살펴볼 수 있습니다. 상품 설명이나 검색 기능을 개선한 뒤 A/B 테스트로 구매 변화를 확인해 볼 수 있습니다.
- **이탈률과 방문자 유형:** 고객이 어느 페이지에서 떠나는지, 신규 방문자와 재방문자가 어떤 동선을 거치는지 비교해 볼 수 있습니다.

중요도만으로 어느 집단이 더 잘 구매하는지, 체류 시간을 늘리면 구매가 늘어나는지까지 알 수는 없습니다. 쿠폰 비용과 구매 이익도 없어 예상 매출이나 투자수익률(ROI)은 계산하지 않았습니다.''')
    timing = '' if wine else '''
## 실제 서비스에 쓰기 전에 확인할 점

`PageValues`는 거래 가치에서 계산되는 지표입니다. 구매 전에 같은 값을 알 수 있는지 확인할 수 없어 이번 입력에서는 제외했습니다. 무조건 데이터 누수라고 단정하기보다는, 예측 시점에 사용할 수 있는지 불확실한 정보를 보수적으로 뺀 것입니다.

체류 시간·반송률·이탈률도 이번 파일에서는 방문이 끝난 뒤의 집계입니다. 따라서 지금 결과를 그대로 “방문 중에 구매를 예측하는 성능”이라고 소개하면 곤란합니다. 실제 서비스에서는 구매 전에 알 수 있었던 정보만 모은 뒤, 이후 기간의 데이터로 다시 평가해야 합니다.
'''
    limitations = ('이번 데이터는 포르투갈의 레드 Vinho Verde 와인입니다. 다른 산지나 화이트 와인에서도 같은 결과가 나오는지는 확인하지 않았습니다. 성분 분석을 바탕으로 한 보조 판단이므로 실제 시음 평가를 대신한다고 보기는 어렵습니다.' if wine else
        '데이터에 정확한 이벤트 시각이 없어 시간순 평가를 하지 못했습니다. 마케팅에 적용하기 전에는 이후 기간에서의 검증과 A/B 테스트가 필요합니다.')
    report = f'''# {'성분표로 읽어 본 와인 품질' if wine else '방문 기록에서 찾아본 구매 신호'}

{first}

{outcome}

## 먼저 데이터를 정리했습니다

{data_note} 원본은 {a['raw_rows']:,}행이었습니다. 입력과 정답이 똑같이 반복된 {a['duplicates_removed']:,}행을 정리한 뒤 {a['rows']:,}행으로 실험했습니다. 같은 기록이 학습과 테스트에 나눠 들어가 점수가 부풀려지는 일을 줄이기 위해서입니다. {conflict_note}

비어 있는 값은 {a['missing_cells']}개였고, 전체에서 {positive}의 비율은 **{a['positive_rate']:.2%}**였습니다. 우리가 찾고 싶은 대상이 적다는 점을 염두에 두고 결과를 읽어야 합니다.

## 배울 자료와 시험 볼 자료를 나눴습니다

데이터는 학습 60%, 검증 20%, 테스트 20%로 나눴습니다. 학습 자료로 모델을 만들고, 검증 자료로 모델과 판단 기준을 골랐습니다. 테스트 자료는 이 선택이 모두 끝난 뒤 최종 성능을 확인하는 데만 썼습니다.

| 용도 | 기록 수 |
| --- | --- |
| 모델 학습 | {split['train']:,} |
| 모델·임계값 선택을 위한 검증 | {split['validation']:,} |
| 최종 성능 확인 | {split['test']:,} |

각 묶음의 정답 비율이 비슷하도록 층화 분할을 했고, `random_state=42`로 같은 분할을 다시 만들 수 있게 했습니다. 평균·표준편차나 결측값을 채울 기준은 **학습 자료에서만** 구했습니다. 데이터 분포를 그릴 때도 학습 자료만 사용했습니다. 이 과정을 `Pipeline`으로 묶어 테스트 정보가 학습에 섞이지 않게 했습니다.

## 여러 모델을 같은 조건에서 비교했습니다

아래 표는 모두 예측 확률 **0.50 이상**을 {'우수' if wine else '구매'}로 판단했을 때의 결과입니다. F1과 AP는 0~1 사이의 값이며, 클수록 좋습니다.

{markdown_table(comparison_for_readers(comparison))}

### 수치 범위를 맞춘 이유

체류 시간이나 성분 농도처럼 입력마다 숫자 크기가 다르면 로지스틱 회귀의 학습과 규제에 영향을 줄 수 있습니다. `StandardScaler`는 학습 자료의 평균과 표준편차를 기준으로 이 범위를 맞춰 줍니다. 의사결정나무는 값을 기준으로 가지를 나누기 때문에 별도로 표준화하지 않았습니다.

{scale_note}

{warning}

### 최종 모델과 판단 기준

일반 사례가 많은 데이터라 정확도만 보고 고르지 않았습니다. 검증 자료에서 **AP(Average Precision)**가 가장 높은 모델을 골랐습니다. AP는 우리가 찾는 대상을 예측 확률의 앞쪽에 얼마나 잘 모았는지 살펴보는 지표입니다. 여기서 쓴 AP는 PR 곡선의 넓이를 사다리꼴로 계산한 값과는 계산 방식이 다릅니다.

선택한 모델은 **{selected}**(`{r['selected_model']}`)입니다. 그다음 확률이 얼마 이상일 때 {'우수' if wine else '구매'}로 판단할지, 즉 **임계값**을 검증 자료에서 조정했습니다. 0.10부터 0.90까지 0.05 간격으로 비교해 F1이 가장 높은 **{r['threshold']:.2f}**를 골랐습니다. 동률이면 더 큰 값을 사용했습니다.

깊이 제한 없는 나무는 과적합을 관찰하는 비교용으로만 사용했습니다. 또 검증 자료에서 특성 중요도를 평가하려고, 마지막에 학습·검증 자료를 합쳐 재학습하지는 않았습니다.

## 처음 보는 데이터에서는 이 정도로 맞혔습니다

다음은 선택한 모델에 조정한 임계값 **{r['threshold']:.2f}**를 적용한 결과입니다. 앞의 비교 표는 0.50을 사용했으므로 수치가 다를 수 있습니다.

| 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- |
| {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} |

{baseline_note}

{counts}

**정밀도**는 “{'우수하다고 추천한 와인' if wine else '구매할 것으로 짚은 고객'} 중 실제로 맞은 비율”, **재현율**은 “{'실제 우수 와인' if wine else '실제 구매 고객'} 중 찾아낸 비율”입니다. 각각 TP/(TP+FP), TP/(TP+FN)으로 계산합니다.

![실제 결과와 예측 결과를 비교한 혼동 행렬](../figures/confusion_matrix.png)

행은 실제 결과, 열은 모델의 예측입니다. {'일반' if wine else '미구매'}을 0, {'우수' if wine else '구매'}를 1로 두었을 때 TN={tn}, FP={fp}, FN={fn}, TP={tp}입니다.

![판단 기준에 따라 달라지는 정밀도와 재현율](../figures/precision_recall.png)

## 틀린 사례도 따로 살펴봤습니다

테스트에서 틀린 {len(errors):,}건을 `reports/misclassified.csv`에 모았습니다. 원본 행 번호와 입력값, 실제 결과, 예측 결과, 확률을 함께 볼 수 있습니다. 아래는 모델이 0.50에서 가장 멀리 떨어진 확률을 내놓고도 틀린 사례 5건입니다.

{markdown_table(errors_for_readers(errors.head(5), wine))}

{'FP는 일반 와인을 우수하다고 잘못 추천한 경우, FN은 우수 와인을 놓친 경우입니다.' if wine else 'FP가 많으면 구매하지 않을 고객에게 캠페인 비용을 쓸 수 있고, FN이 많으면 관심을 가져볼 구매 고객을 놓칠 수 있습니다.'} 표의 확률은 별도로 보정하지 않은 값입니다. 특히 나무 모델의 0이나 1을 절대적인 확신으로 받아들이지는 않는 편이 좋겠습니다.

오답의 원인은 이 표만으로 단정하지 않았습니다. 테스트 오답을 보고 다음 모델을 개선한다면, 그 모델은 새 테스트 자료로 다시 확인해야 합니다.

## 학습 점수만 높으면 생기는 문제

깊이를 제한하지 않은 나무는 학습 F1이 **{overfit['train_f1']:.4f}**였지만, 검증 F1은 **{overfit['validation_f1']:.4f}**였습니다. 배운 자료를 잘 외우더라도 새로운 자료에서는 잘 맞히지 못할 수 있다는 예입니다. 그래서 깊이를 3·5·8로 제한하고, 마지막 잎에 들어가는 학습 표본도 최소 10개가 되도록 해 비교했습니다.

## 모델이 주로 참고한 항목

입력 항목을 하나씩 무작위로 섞은 뒤 검증 AP가 얼마나 떨어지는지 8번씩 확인했습니다. 이를 **순열 중요도(permutation importance)**라고 합니다. 감소량이 크면, 이 모델이 그 항목에 많이 의존했다는 뜻입니다.

{markdown_table(importance_for_readers(ranking.head(5)))}

![입력 항목을 섞었을 때 성능이 얼마나 떨어졌는지 비교](../figures/feature_importance.png)

{implications}

감소량이 음수인 항목은 이번 실험에서 중요도가 뚜렷하지 않을 수 있습니다. 서로 비슷한 정보를 담은 항목끼리는 중요도를 나눠 가질 수도 있습니다.
{timing}
## 여기까지 확인했고, 다음에는 더 확인하고 싶습니다

{limitations}

이번에는 한 번의 무작위 분할로 실험해서 분할을 바꾸면 점수가 얼마나 달라지는지, 신뢰구간은 어느 정도인지까지 계산하지 않았습니다. 다음에는 교차검증이나 반복 분할로 결과의 안정성을 확인하고, 예측 확률 보정과 오류 비용을 반영한 임계값 선택도 비교해 보고 싶습니다.

## 같은 결과를 다시 확인하려면

저장소에서 `python src/train.py`를 실행하면 학습과 보고서를 다시 만들 수 있습니다. 데이터 출처와 이용 조건은 [데이터 출처 안내](DATA_SOURCES.md)에 정리했습니다. 파일별로는 다음 기록을 남겼습니다.

- `reports/split_manifest.csv`: 원본의 어느 행을 학습·검증·테스트에 썼는지
- `reports/metrics.json`: 평가 수치, 실행 버전, 원본 파일의 SHA-256 확인값
- `requirements-lock.txt`: 실험 환경의 패키지 버전

CSV와 JSON의 영문 열 이름은 코드에서 그대로 읽을 수 있도록 유지했습니다. 이 보고서와 노트북에서는 한국어 이름을 덧붙여 설명했습니다.
'''
    (here/'evaluation.md').write_text(report, encoding='utf-8')
