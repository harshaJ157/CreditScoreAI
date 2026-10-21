import pandas as pd

NUMERIC_FIELDS = {
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
}

def parse_numeric(value):
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def profile_to_frame(profile: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        value = profile.get(col, "")
        if col in NUMERIC_FIELDS:
            value = parse_numeric(value)
        row[col] = value
    return pd.DataFrame([row], columns=feature_columns)
