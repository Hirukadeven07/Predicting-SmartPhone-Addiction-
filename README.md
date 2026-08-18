# Predicting Smartphone Addiction

Solution for the Kaggle Playground Series S6E8 competition: predicting the probability that a
person is addicted to their smartphone from a set of behavioral and usage features.

**Competition:** [playground-series-s6e8](https://www.kaggle.com/competitions/playground-series-s6e8)
**Metric:** ROC-AUC

## EDA

`notebooks/eda.ipynb` covers missingness patterns, target balance, feature distributions by
class, and correlations (matplotlib/seaborn) — the findings that motivated the feature
engineering below.

`notebooks/eda_plotly.ipynb` is a companion notebook with interactive Plotly versions of the
same key visuals (hover tooltips, a feature-picker dropdown, a brushable parallel-coordinates
view), kept separate so the static notebook doesn't get weighed down by the larger Plotly
outputs.

## Approach

The dataset contains ~700K training rows with 12 features (screen time, social media/gaming
hours, sleep, notifications, app opens, gender, stress level, academic/work impact) and heavy
missingness (5-20% per column). The pipeline:

- **Feature engineering** — missingness indicators per column, usage ratios (social/gaming/work
  time relative to total screen time), weekday vs. weekend deltas, notification/app-open rates,
  and a few interaction and polynomial terms.
- **Models** — LightGBM and XGBoost gradient-boosted trees, each trained with 5-fold stratified
  cross-validation and early stopping, using native categorical feature support.
- **Ensembling** — out-of-fold predictions from both models are combined via a logistic
  regression meta-model trained in logit space.

## Results

| Model | OOF AUC |
|---|---|
| LightGBM | 0.9637 |
| XGBoost | 0.9641 |
| **Stacked ensemble** | **0.9642** |

## Usage

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn plotly kaleido

python scripts/train_lgbm.py       # single LightGBM baseline
python scripts/train_ensemble.py   # LightGBM + XGBoost stacked ensemble
```

Both scripts expect `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv` (from
the Kaggle competition page) and write a submission CSV to `submissions/`.
