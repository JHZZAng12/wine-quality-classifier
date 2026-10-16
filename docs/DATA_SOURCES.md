# 사용한 데이터와 출처

이번 실습에서는 UCI의 **Wine Quality** 데이터 중 레드 와인 파일(`winequality-red.csv`)을 사용했습니다. 포르투갈 Vinho Verde 와인의 성분과 품질 점수가 담겨 있습니다. 품종을 구분하는 `sklearn.datasets.load_wine`과는 다른 데이터입니다.

원문 인용 정보는 다음과 같습니다.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. [DOI: 10.24432/C56S3T](https://doi.org/10.24432/C56S3T).

[UCI 데이터 소개 페이지](https://archive.ics.uci.edu/dataset/186/wine+quality)

원본에서 중복 기록을 정리한 뒤, 품질 점수 7점 이상을 우수(1), 그보다 낮은 점수를 일반(0)으로 바꿔 사용했습니다.

## 데이터를 다시 받는 방법

원본 CSV는 저장소에 올리지 않았습니다. `python src/train.py`를 실행하면 공식 UCI ZIP 파일에서 필요한 CSV를 내려받아 `data/`에 보관합니다. 그다음 실행부터는 받아 둔 파일을 사용합니다.

## 이용 조건

데이터의 라이선스는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다. 사용할 때는 원저작자와 출처를 밝히고, 데이터를 바꿨다면 그 사실도 알려야 합니다. 이 프로젝트에서는 위에 적은 정리와 변환을 거쳤으며, 보고서·통계·오답 CSV에도 원본 데이터에서 만든 결과가 들어 있습니다. 원저작자가 이 프로젝트나 모델을 보증한다는 뜻은 아닙니다.

## 구현하면서 참고한 문서

- [scikit-learn: 학습 과정에서 데이터를 다룰 때 주의할 점](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: 순열 중요도로 모델 해석하기](https://scikit-learn.org/stable/modules/permutation_importance.html)
