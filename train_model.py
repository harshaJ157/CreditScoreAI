
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load training data
df = pd.read_csv("data/train.csv", low_memory=False)

target = "Credit_Score"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

# Remove rows where the target is missing
df = df.dropna(subset=[target]).copy()

# Exclude identifiers and fields that are not useful predictors
exclude_columns = [
    target,
    "ID",
    "Customer_ID",
    "Name",
    "SSN",
]

X = df.drop(
    columns=[c for c in exclude_columns if c in df.columns]
)
y = df[target].astype(str).str.strip()

# Remove rows with invalid target labels
valid_labels = {"Poor", "Standard", "Good"}
valid_rows = y.isin(valid_labels)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Treat every feature as numeric or categorical
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# Keep a holdout set for an initial evaluation
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training " \
"model...")
model.fit(X_train, y_train)

predictions = model.predict(X_valid)

print("\nValidation accuracy:")
print(f"{accuracy_score(y_valid, predictions):.2%}")

print("\nClassification report:")
print(classification_report(y_valid, predictions, zero_division=0))

# Save the trained pipeline
Path("models").mkdir(exist_ok=True)
joblib.dump(
    {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "labels": sorted(y.unique().tolist()),
    },
    "models/credit_score_model.joblib",
)

print("\nModel saved to models/credit_score_model.joblib")
