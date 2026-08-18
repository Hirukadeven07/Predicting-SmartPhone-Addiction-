"""
Predicting Smartphone Addiction - Playground Series S6E8
Ensemble model: LightGBM + XGBoost, 5-fold Stratified CV each,
combined via logistic-regression stacking in logit space.
(CatBoost was dropped -- too slow on this machine for the time budget.)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import time
import warnings
warnings.filterwarnings("ignore")

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
    eps = 1e-3

    for c in NUM_COLS + CAT_COLS:
        df[f"{c}_missing"] = df[c].isnull().astype(np.int8)

    df["social_to_screen_ratio"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_to_screen_ratio"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + eps)
    df["work_to_screen_ratio"] = df["work_study_hours"] / (df["daily_screen_time_hours"] + eps)
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"] / (df["sleep_hours"] + eps)
    df["weekend_vs_weekday_screen"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    df["weekend_screen_ratio"] = df["weekend_screen_time"] / (df["daily_screen_time_hours"] + eps)
    df["opens_per_notification"] = df["app_opens_per_day"] / (df["notifications_per_day"] + eps)
    df["notifications_per_hour"] = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["opens_per_hour"] = df["app_opens_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["leisure_hours"] = df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
    df["leisure_to_work_ratio"] = df["leisure_hours"] / (df["work_study_hours"] + eps)
    df["total_accounted_hours"] = (
        df["social_media_hours"].fillna(0)
        + df["gaming_hours"].fillna(0)
        + df["work_study_hours"].fillna(0)
        + df["sleep_hours"].fillna(0)
    )
    df["free_hours_24"] = 24 - df["total_accounted_hours"]
    df["sleep_deficit"] = 8 - df["sleep_hours"]
    df["screen_per_age"] = df["daily_screen_time_hours"] / (df["age"] + eps)
    df["notif_x_opens"] = df["notifications_per_day"] * df["app_opens_per_day"]
    df["screen_x_social"] = df["daily_screen_time_hours"] * df["social_media_hours"]
    df["screen_sq"] = df["daily_screen_time_hours"] ** 2
    df["social_sq"] = df["social_media_hours"] ** 2

    df["n_missing"] = df[[c for c in NUM_COLS + CAT_COLS]].isnull().sum(axis=1)

    df["stress_academic_combo"] = (
        df["stress_level"].astype(str) + "_" + df["academic_work_impact"].astype(str)
    )

    return df


def prep_categorical(train_feat, test_feat, cat_cols):
    tr, te = train_feat.copy(), test_feat.copy()
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
        cats = pd.api.types.union_categoricals([tr[c], te[c]]).categories
        tr[c] = tr[c].cat.set_categories(cats)
        te[c] = te[c].cat.set_categories(cats)
    return tr, te


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    t0 = time.time()
    train, test = load_data()
    print(f"Loaded train {train.shape}, test {test.shape}", flush=True)

    y = train["addicted_label"].values
    test_ids = test["id"].values

    train_feat = engineer_features(train.drop(columns=["id", "addicted_label"]))
    test_feat = engineer_features(test.drop(columns=["id"]))

    all_cat_cols = CAT_COLS + ["stress_academic_combo"]

    # both LightGBM and XGBoost (hist, enable_categorical) use the same
    # pandas 'category' dtype prep
    prep_tr, prep_te = prep_categorical(train_feat, test_feat, all_cat_cols)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_feat, y))

    n_train, n_test = len(train_feat), len(test_feat)
    oof = {m: np.zeros(n_train) for m in ["lgbm", "xgb"]}
    test_pred = {m: np.zeros(n_test) for m in ["lgbm", "xgb"]}

    lgbm_params = dict(
        objective="binary", metric="auc", boosting_type="gbdt",
        n_estimators=2500, learning_rate=0.035, num_leaves=63,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

    xgb_params = dict(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=2500, learning_rate=0.035, max_depth=7,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
        enable_categorical=True, random_state=RANDOM_STATE, n_jobs=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===", flush=True)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # LightGBM
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            prep_tr.iloc[tr_idx], y_tr,
            eval_set=[(prep_tr.iloc[va_idx], y_va)],
            eval_metric="auc", categorical_feature=all_cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof["lgbm"][va_idx] = model.predict_proba(prep_tr.iloc[va_idx], num_iteration=model.best_iteration_)[:, 1]
        test_pred["lgbm"] += model.predict_proba(prep_te, num_iteration=model.best_iteration_)[:, 1] / N_SPLITS
        auc_l = roc_auc_score(y_va, oof["lgbm"][va_idx])
        print(f"  LightGBM AUC: {auc_l:.5f} (best_iter={model.best_iteration_})", flush=True)

        # XGBoost
        xmodel = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=100)
        xmodel.fit(
            prep_tr.iloc[tr_idx], y_tr,
            eval_set=[(prep_tr.iloc[va_idx], y_va)],
            verbose=False,
        )
        oof["xgb"][va_idx] = xmodel.predict_proba(prep_tr.iloc[va_idx])[:, 1]
        test_pred["xgb"] += xmodel.predict_proba(prep_te)[:, 1] / N_SPLITS
        auc_x = roc_auc_score(y_va, oof["xgb"][va_idx])
        print(f"  XGBoost  AUC: {auc_x:.5f} (best_iter={xmodel.best_iteration})", flush=True)

    print("\n=== Individual model OOF AUCs ===", flush=True)
    for m in ["lgbm", "xgb"]:
        print(f"  {m}: {roc_auc_score(y, oof[m]):.5f}", flush=True)

    # Stack in logit space via logistic regression meta-model
    stack_oof = np.column_stack([logit(oof["lgbm"]), logit(oof["xgb"])])
    stack_test = np.column_stack([logit(test_pred["lgbm"]), logit(test_pred["xgb"])])

    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(stack_oof, y)
    oof_blend = meta.predict_proba(stack_oof)[:, 1]
    blend_auc = roc_auc_score(y, oof_blend)
    print(f"\nStacked blend OOF AUC: {blend_auc:.5f}", flush=True)
    print(f"Meta weights: {meta.coef_}, intercept: {meta.intercept_}", flush=True)

    # Simple average as a sanity comparison
    simple_avg = (oof["lgbm"] + oof["xgb"]) / 2
    print(f"Simple average OOF AUC: {roc_auc_score(y, simple_avg):.5f}", flush=True)

    test_blend = meta.predict_proba(stack_test)[:, 1]

    submission = pd.DataFrame({"id": test_ids, "addicted_label": test_blend})
    out_path = f"{OUT_DIR}/submission_ensemble.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
