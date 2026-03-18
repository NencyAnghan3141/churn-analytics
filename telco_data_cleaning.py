"""
telco_data_cleaning.py

Step-by-step data cleaning script for the Telco Customer Churn dataset.

This script will:

1. Load the RAW dataset from:  data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
2. Clean and prepare a Power BI–friendly file:
       data/processed/telco_clean_for_bi.csv
3. Prepare a Machine Learning–ready file:
       data/processed/telco_clean_for_ml.csv

Run this script using:
    python telco_data_cleaning.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# LOAD RAW DATA
# -----------------------------
def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    print(f"📥 Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"✅ Data loaded. Shape: {df.shape}")
    return df


# -----------------------------
# BASIC INSPECTION
# -----------------------------
def basic_inspection(df: pd.DataFrame) -> None:
    """Print basic dataset info for understanding."""
    print("\n🔎 BASIC INFO")
    print(df.info())

    print("\n📊 FIRST 5 ROWS")
    print(df.head())

    print("\n📦 MISSING VALUES PER COLUMN")
    print(df.isnull().sum())


# -----------------------------
# CLEANING FOR POWER BI OUTPUT
# -----------------------------
def clean_for_bi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset for Power BI (human readable categories).
    Steps:
    - strip whitespace
    - convert TotalCharges to numeric
    - fill missing values
    - create churn flag
    """
    df_bi = df.copy()

    # 1. Strip whitespace
    print("\n🧹 Stripping whitespace...")
    str_cols = df_bi.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df_bi[col] = df_bi[col].astype(str).str.strip()

    # 2. Convert TotalCharges to numeric
    print("🔢 Converting TotalCharges to numeric...")
    if "TotalCharges" in df_bi.columns:
        df_bi["TotalCharges"] = pd.to_numeric(df_bi["TotalCharges"], errors="coerce")

        missing_before = df_bi["TotalCharges"].isna().sum()
        print(f"   Missing TotalCharges before fill: {missing_before}")

        if missing_before > 0:
            median_total = df_bi["TotalCharges"].median()
            df_bi["TotalCharges"].fillna(median_total, inplace=True)
            print(f"   Filled with median: {median_total}")

    # 3. Create churn flag
    if "Churn" in df_bi.columns:
        print("🏷️ Creating Churn_Flag column...")
        df_bi["Churn_Flag"] = df_bi["Churn"].map({"Yes": 1, "No": 0})

    return df_bi


# -----------------------------
# CLEANING FOR ML OUTPUT
# -----------------------------
def clean_for_ml(df_bi: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready dataset:
    - drop customerID
    - encode categorical columns
    - ensure Churn_Flag exists
    """
    df_ml = df_bi.copy()

    # 1. Drop customerID
    if "customerID" in df_ml.columns:
        print("\n🧬 Dropping customerID for ML...")
        df_ml.drop(columns=["customerID"], inplace=True)

    # 2. Ensure Churn_Flag exists
    if "Churn_Flag" not in df_ml.columns and "Churn" in df_ml.columns:
        print("⚠️ Creating Churn_Flag from Churn column...")
        df_ml["Churn_Flag"] = df_ml["Churn"].map({"Yes": 1, "No": 0})

    # 3. Drop original Churn column
    if "Churn" in df_ml.columns:
        print("🧽 Dropping original 'Churn'...")
        df_ml.drop(columns=["Churn"], inplace=True)

    # 4. One-hot encode categorical columns
    cat_cols = df_ml.select_dtypes(include=["object"]).columns.tolist()
    print(f"📁 Encoding categorical columns: {cat_cols}")

    df_ml_encoded = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    print(f"✅ ML DataFrame shape after encoding: {df_ml_encoded.shape}")

    return df_ml_encoded


# -----------------------------
# SAVE DATAFRAME
# -----------------------------
def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path} (shape: {df.shape})")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    project_root = Path(__file__).resolve().parent

    raw_path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_bi_path = project_root / "data" / "processed" / "telco_clean_for_bi.csv"
    processed_ml_path = project_root / "data" / "processed" / "telco_clean_for_ml.csv"

    # 1. Load raw data
    df_raw = load_raw_data(raw_path)

    # 2. Basic checks
    basic_inspection(df_raw)

    # 3. Clean for Power BI
    print("\n===== CLEANING FOR POWER BI =====")
    df_bi = clean_for_bi(df_raw)
    save_dataframe(df_bi, processed_bi_path)

    # 4. Clean for Machine Learning
    print("\n===== CLEANING FOR MACHINE LEARNING =====")
    df_ml = clean_for_ml(df_bi)
    save_dataframe(df_ml, processed_ml_path)

    print("\n🎉 DONE!")
    print(f"➡ Use for Power BI: {processed_bi_path}")
    print(f"➡ Use for ML Model: {processed_ml_path}")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    main()
