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

## Model training notebook

`notebooks/model_training.ipynb` walks through the full modeling pipeline behind
`scripts/train_ensemble.py` step by step and cell by cell: loading the data, an illustrative
train/validation hold-out split followed by the 5-fold stratified CV actually used, each block
of feature engineering applied and previewed individually, categorical encoding, per-fold
LightGBM/XGBoost training, OOF evaluation, logistic-regression stacking, feature importance,
and writing the final submission. Same hyperparameters and random seed as the script, so its
OOF AUCs match the results table below (small differences are normal — LightGBM/XGBoost are
multi-threaded and not bit-for-bit deterministic). Training all 10 fold/model fits takes
roughly 30 minutes on a typical laptop CPU.

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
