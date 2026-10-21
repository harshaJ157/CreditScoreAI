from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_score_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Credit_Score"
FEATURES = [
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
]
NUMERIC_FEATURES = [
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]
CATEGORICAL_FEATURES = [
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your CSV at data/train.csv. It must contain Credit_Score "
            "and the eight documented feature columns."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    required = [TARGET] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nCheck data/README.md and make sure the CSV has the expected column names."
        )

    df = df[required].copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""]
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .astype(object)
            .where(df[col].notna(), float("nan"))
        )
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[col] = df[col].replace({"": float("nan"), "nan": float("nan")})

    if df[TARGET].nunique() < 2:
        raise ValueError("Credit_Score must contain at least two target categories.")

    X = df[FEATURES]
    y = df[TARGET]
    X = X.astype(object).where(X.notna(), float("nan"))
    y = y.astype(object).where(y.notna(), float("nan"))
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1
        ),
    }

    best_name, best_model, best_accuracy = None, None, -1
    results = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=sorted(y.unique())
            ).tolist(),
            "labels": sorted(y.unique()),
        }
        if accuracy > best_accuracy:
            best_name, best_model, best_accuracy = name, pipeline, accuracy

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "features": FEATURES,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "model_name": best_name,
        "dataset_rows": int(len(df)),
    }
    joblib.dump(artifact, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "best_accuracy": round(float(best_accuracy), 4),
            "rows": int(len(df)),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "results": results,
        }, f, indent=2)

    print(f"Rows used: {len(df)}")
    print(f"Best model: {best_name}")
    print(f"Hold-out accuracy: {best_accuracy:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("Reminder: this is a dataset-based category estimate, not an official CIBIL score.")

if __name__ == "__main__":
    main()
