# 데이터 출처

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

https://archive.ics.uci.edu/dataset/186/wine+quality

레드 와인 CSV만 사용합니다. `sklearn.datasets.load_wine`은 품종 데이터이므로 품질 분류에 사용하지 않습니다.

데이터 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원본 CSV는 저장소에서 배포하지 않고 `train.py`가 공식 UCI ZIP에서 내려받아 `data/`에 캐시합니다. 원본 데이터를 중복 제거하고 타깃을 변환했습니다. 보고서·집계·오답 CSV에는 해당 데이터의 파생 결과가 포함됩니다. 저작자에게 프로젝트의 보증을 암시하지 않습니다.

구현 참고: [scikit-learn 데이터 누수 방지](https://scikit-learn.org/stable/common_pitfalls.html), [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
