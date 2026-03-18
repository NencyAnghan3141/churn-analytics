import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = INPUT_CSV = os.path.join("data","processed","telco_clean_for_ml.csv")

ID_FALLBACK_CSV_CANDIDATES = [
    "data/processed/telco_clean_for_bi.csv",
    "data/raw/telco.csv",
    "data/raw/telco_churn.csv",
]

OUTPUT_PRED_CSV = "data/processed/telco_with_churn_probability.csv"
OUTPUT_LR_MODEL_FILE = "data/processed/churn_model_logreg.joblib"
OUTPUT_RF_MODEL_FILE = "data/processed/churn_model_random_forest.joblib"
OUTPUT_MODEL_COMPARISON_CSV = "data/processed/model_comparison.csv"

TARGET_COL = "Churn_Flag"
ID_COL = "customerID"

HIGH_RISK_TH = 0.70
MED_RISK_TH = 0.40

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================
def find_existing_file(candidates: list[str]) -> str | None:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def make_risk_level(p: float) -> str:
    if p >= HIGH_RISK_TH:
        return "High"
    elif p >= MED_RISK_TH:
        return "Medium"
    return "Low"


def safe_fillna(df_: pd.DataFrame) -> pd.DataFrame:
    df_ = df_.replace([np.inf, -np.inf], np.nan)
    return df_.fillna(0)


def evaluate_model(name: str, y_true, y_pred, y_prob) -> dict:
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n===== {name.upper()} EVALUATION =====")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {"Model": name, "Accuracy": acc, "ROC_AUC": auc}


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_CSV)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found in {INPUT_CSV}. "
        f"Available columns: {list(df.columns)}"
    )

customer_ids = df[ID_COL].copy() if ID_COL in df.columns else None

if df[TARGET_COL].dtype == "object":
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
df[TARGET_COL] = df[TARGET_COL].astype(int)

X = df.drop(columns=[TARGET_COL], errors="ignore")
y = df[TARGET_COL]

if ID_COL in X.columns:
    X = X.drop(columns=[ID_COL])

X = safe_fillna(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ============================================================
# 1) LOGISTIC REGRESSION
# ============================================================
lr_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                max_iter=10000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

lr_cv_auc = cross_val_score(lr_model, X, y, cv=cv, scoring="roc_auc")
print("\nLogistic Regression CV AUC:", np.round(lr_cv_auc, 4))
print("Mean CV AUC:", round(lr_cv_auc.mean(), 4))

joblib.dump(lr_model, OUTPUT_LR_MODEL_FILE)
print(f"\n✅ Saved LR model file: {OUTPUT_LR_MODEL_FILE}")

# ============================================================
# 2) RANDOM FOREST
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

rf_cv_auc = cross_val_score(rf_model, X, y, cv=cv, scoring="roc_auc")
print("\nRandom Forest CV AUC:", np.round(rf_cv_auc, 4))
print("Mean CV AUC:", round(rf_cv_auc.mean(), 4))

joblib.dump(rf_model, OUTPUT_RF_MODEL_FILE)
print(f"\n✅ Saved RF model file: {OUTPUT_RF_MODEL_FILE}")

# ============================================================
# FEATURE IMPORTANCE EXPORT (Random Forest)
# ============================================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("Feature importance exported successfully")


# ============================================================
# 3) DECISION TREE
# ============================================================
dt_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred, dt_prob)

dt_cv_auc = cross_val_score(dt_model, X, y, cv=cv, scoring="roc_auc")
print("\nDecision Tree CV AUC:", np.round(dt_cv_auc, 4))
print("Mean CV AUC:", round(dt_cv_auc.mean(), 4))

joblib.dump(dt_model, "data/processed/churn_model_decision_tree.joblib")
print("✅ Saved Decision Tree model file: data/processed/churn_model_decision_tree.joblib")

# ============================================================
# MODEL COMPARISON EXPORT (for Power BI)
# ============================================================
comparison_df = pd.DataFrame([
    {
        "Model": lr_metrics["Model"],
        "Accuracy": round(lr_metrics["Accuracy"], 4),
        "ROC_AUC": round(lr_metrics["ROC_AUC"], 4),
        "CV_ROC_AUC_Mean": round(lr_cv_auc.mean(), 4),
    },
    {
        "Model": rf_metrics["Model"],
        "Accuracy": round(rf_metrics["Accuracy"], 4),
        "ROC_AUC": round(rf_metrics["ROC_AUC"], 4),
        "CV_ROC_AUC_Mean": round(rf_cv_auc.mean(), 4),
    },
    {
        "Model": dt_metrics["Model"],
        "Accuracy": round(dt_metrics["Accuracy"], 4),
        "ROC_AUC": round(dt_metrics["ROC_AUC"], 4),
        "CV_ROC_AUC_Mean": round(dt_cv_auc.mean(), 4),
    },
])

comparison_df.to_csv(OUTPUT_MODEL_COMPARISON_CSV, index=False)
print(f"\n✅ Saved model comparison CSV: {OUTPUT_MODEL_COMPARISON_CSV}")
print(comparison_df)

# ============================================================
# PREDICT FOR ALL ROWS (BEST MODEL ONLY)
# ============================================================
best_model_name = "Random Forest" if rf_metrics["ROC_AUC"] >= lr_metrics["ROC_AUC"] else "Logistic Regression"
best_model = rf_model if best_model_name == "Random Forest" else lr_model

print(f"\n🏆 Best model selected by ROC_AUC: {best_model_name}")

all_prob = best_model.predict_proba(X)[:, 1]

if customer_ids is None:
    id_source = find_existing_file(ID_FALLBACK_CSV_CANDIDATES)
    if id_source:
        df_id = pd.read_csv(id_source)
        if ID_COL in df_id.columns and len(df_id) == len(df):
            customer_ids = df_id[ID_COL].copy()
            print(f"ℹ️ customerID fetched from: {id_source}")
        else:
            print(f"⚠️ Could not use {id_source} for customerID")

if customer_ids is None:
    print("⚠️ customerID not found. Creating fallback IDs.")
    customer_ids = pd.Series(range(len(df)), name=ID_COL)

out = pd.DataFrame(
    {
        ID_COL: customer_ids,
        "churn_probability": np.round(all_prob, 4),
    }
)

out["risk_level"] = out["churn_probability"].apply(make_risk_level)
out["model_used"] = best_model_name

out.to_csv(OUTPUT_PRED_CSV, index=False)

print(f"\n✅ Saved predictions CSV for Power BI: {OUTPUT_PRED_CSV}")
print("✅ Columns exported:", list(out.columns))