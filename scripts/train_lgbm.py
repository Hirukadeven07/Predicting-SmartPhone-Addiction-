"""
Predicting Smartphone Addiction - Playground Series S6E8
Baseline+ model: LightGBM with 5-fold stratified CV, missing-value indicators,
categorical encoding, and a handful of engineered ratio features.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time

RANDOM_STATE = 42
N_SPLITS = 5

DATA_DIR = "data"
OUT_DIR = "submissions"

NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


def engineer_features(df):
    df = df.copy()

    # missingness indicators (missingness itself can be informative)
    for c in NUM_COLS + CAT_COLS:
        df[f"{c}_missing"] = df[c].isnull().astype(np.int8)

    # ratio / interaction features (guard div-by-zero with np.where)
    eps = 1e-3
    df["social_to_screen_ratio"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_to_screen_ratio"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + eps)
    df["work_to_screen_ratio"] = df["work_study_hours"] / (df["daily_screen_time_hours"] + eps)
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"] / (df["sleep_hours"] + eps)
    df["weekend_vs_weekday_screen"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    df["opens_per_notification"] = df["app_opens_per_day"] / (df["notifications_per_day"] + eps)
    df["notifications_per_hour"] = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["leisure_hours"] = df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
    df["leisure_to_work_ratio"] = df["leisure_hours"] / (df["work_study_hours"] + eps)
    df["total_accounted_hours"] = (
        df["social_media_hours"].fillna(0)
        + df["gaming_hours"].fillna(0)
        + df["work_study_hours"].fillna(0)
        + df["sleep_hours"].fillna(0)
    )
    df["free_hours_24"] = 24 - df["total_accounted_hours"]

    # count of missing fields per row (data-quality / engagement signal)
    df["n_missing"] = df[[c for c in NUM_COLS + CAT_COLS]].isnull().sum(axis=1)

    return df


def encode_categoricals(train, test):
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        # align categories between train/test
        cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
        train[c] = train[c].cat.set_categories(cats)
        test[c] = test[c].cat.set_categories(cats)
    return train, test


def main():
    t0 = time.time()
    train, test = load_data()
    print(f"Loaded train {train.shape}, test {test.shape}")

    y = train["addicted_label"].values
    train_ids = train["id"].values
    test_ids = test["id"].values

    train_feat = engineer_features(train.drop(columns=["id", "addicted_label"]))
    test_feat = engineer_features(test.drop(columns=["id"]))

    train_feat, test_feat = encode_categoricals(train_feat, test_feat)

    feature_cols = [c for c in train_feat.columns]
    print(f"Using {len(feature_cols)} features")

    cat_features = [c for c in CAT_COLS]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(train_feat))
    test_preds = np.zeros(len(test_feat))
    fold_scores = []

    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_feat, y)):
        X_tr, X_va = train_feat.iloc[tr_idx], train_feat.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )

        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        oof_preds[va_idx] = va_pred
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1}/{N_SPLITS} AUC: {fold_auc:.5f} (best_iter={model.best_iteration_})")

        test_preds += model.predict_proba(test_feat, num_iteration=model.best_iteration_)[:, 1] / N_SPLITS

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOOF AUC: {overall_auc:.5f}  (fold mean {np.mean(fold_scores):.5f} +- {np.std(fold_scores):.5f})")

    # feature importance from last fold model (quick look)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 features (last fold):")
    print(importances.head(15))

    submission = pd.DataFrame({"id": test_ids, "addicted_label": test_preds})
    out_path = f"{OUT_DIR}/submission_lgbm.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
