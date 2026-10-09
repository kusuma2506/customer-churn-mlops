"""
Practical 2: Train and evaluate a Customer Churn classification model.
Expected dataset path in the GitHub repository: data/customer_churn.csv.csv
Run: python train_model.py
"""
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = Path("data/customer_churn.csv.csv")
MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "customer_churn_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["CustomerID"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Check the file path and filename."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN] + [c for c in DROP_COLUMNS if c in df.columns])

    # Convert target labels such as Yes/No to numeric labels when needed.
    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    if y.nunique() < 2:
        raise ValueError("The Churn target must contain at least two classes.")

    # Remove columns that are completely empty; the imputers cannot learn from them.
    X = X.dropna(axis=1, how="all")
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        ))
    ])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset_path": str(DATA_PATH),
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "target_column": TARGET_COLUMN,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "target_classes": [str(value) for value in sorted(pd.Series(y).unique().tolist())]
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed successfully.")
    print(f"Rows: {metrics['rows']}")
    print(f"Features: {metrics['features']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
