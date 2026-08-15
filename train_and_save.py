"""
Run this once (locally, where train.csv / test.csv live) to produce the
artifacts the FastAPI app serves from: model.pkl, scaler.pkl,
feature_columns.json, and defaults.json.

    python train_and_save.py

Mirrors the pipeline built in notebooks/eda_and_modeling.ipynb, using
Gradient Boosting (the winning model from the GridSearchCV comparison).
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import clean_and_encode, align_to_training_columns, NOMINAL_COLS

RAW_TRAIN_PATH = "train.csv"
RAW_TEST_PATH = "test.csv"


def main():
    train = pd.read_csv(RAW_TRAIN_PATH)
    test = pd.read_csv(RAW_TEST_PATH)

    train = train.drop("Id", axis=1)
    test_ids = test["Id"]
    test = test.drop("Id", axis=1)

    # save raw-column defaults (median for numeric, mode for categorical)
    # BEFORE any fills — this is what the API falls back on for fields
    # a caller doesn't supply.
    raw_features = train.drop(columns=["SalePrice"])
    defaults = {}
    for col in raw_features.columns:
        if raw_features[col].dtype == "object":
            defaults[col] = raw_features[col].mode(dropna=True).iloc[0] if not raw_features[col].mode(dropna=True).empty else None
        else:
            defaults[col] = float(raw_features[col].median(skipna=True))

    with open("defaults.json", "w") as f:
        json.dump(defaults, f, indent=2)

    # combine train + test so one-hot categories line up, same as the notebook
    train["dataset"] = "train"
    test["dataset"] = "test"
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # generic leftover-NaN fill (test-only quirks), matching the notebook
    for col in combined.select_dtypes(include="number").columns:
        if combined[col].isnull().sum() > 0:
            combined[col] = combined[col].fillna(combined[col].median())
    for col in combined.select_dtypes(include="object").columns:
        if col not in ("dataset",) and combined[col].isnull().sum() > 0:
            combined[col] = combined[col].fillna(combined[col].mode()[0])

    combined = clean_and_encode(combined)
    combined = pd.get_dummies(combined, columns=NOMINAL_COLS, drop_first=True)

    train_final = combined[combined["dataset"] == "train"].drop(columns=["dataset"])

    X = train_final.drop("SalePrice", axis=1)
    y = train_final["SalePrice"]

    feature_columns = list(X.columns)
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train)

    scaler = StandardScaler()
    scaler.fit(X_train)  # kept for parity / optional use by linear models
    joblib.dump(scaler, "scaler.pkl")

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train_log)
    joblib.dump(model, "model.pkl")

    print("Saved model.pkl, scaler.pkl, feature_columns.json, defaults.json")


if __name__ == "__main__":
    main()
