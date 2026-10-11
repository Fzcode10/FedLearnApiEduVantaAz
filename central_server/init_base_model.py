"""
init_base_model.py
==================
Loads real student records from  central_server/student_data.csv,
encodes all categorical columns, trains an initial SGDRegressor, and exports:
  - weights/global_weights_v1.npz   (coef + intercept + scaler params)
  - weights/feature_metadata.json   (ordered feature column list + encodings)

Run once before starting server_api.py:
    python init_base_model.py

CSV column -> internal feature name mapping is defined in CSV_TO_FEATURE_MAP.
Extra CSV columns (Preferred_Language, Educational_Stage, etc.) are ignored.
Binary CSV columns that use "Yes"/"No" are converted to 1/0 automatically.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DATA_FILE     = os.path.join(BASE_DIR, "student_data.csv")
WEIGHTS_FILE  = os.path.join(WEIGHTS_DIR, "global_weights_v1.npz")
METADATA_FILE = os.path.join(WEIGHTS_DIR, "feature_metadata.json")

# ── CSV column -> internal feature name ────────────────────────────────────────
# Maps each CSV header to our canonical feature name used everywhere in the system.
CSV_TO_FEATURE_MAP = {
    "Current_CGPA":                     "current_cgpa",
    "Matriculation_10th_Percentage":    "matric_10th",
    "Intermediate_12th_Percentage":     "inter_12th",
    "Attendance_Percentage":            "attendance_pct",
    "Daily_Study_Time_Hours":           "daily_study_hours",
    "Backlogs_Count":                   "backlogs",
    "Assignment_Submission_Rate":       "assignment_rate",
    "Stress_Level":                     "stress_level",
    "Self_Confidence":                  "self_confidence",
    "Time_Management_Skill":            "time_mgmt",
    "Exam_Anxiety_Level":               "exam_anxiety",
    "Motivation_Level":                 "motivation_level",
    "Sleep_Hours":                      "sleep_hours",
    "Sleep_Quality":                    "sleep_quality",
    "Screen_Time_Hours":                "screen_time_hours",
    "Social_Media_Usage":               "social_media_usage",
    "Physical_Health_Fitness":          "physical_health",
    "Diet_Nutrition_Quality":           "diet_quality",
    "Extracurricular_Involvement_Hours":"extracurricular_hours",
    "Financial_Support_Level":          "financial_support",
    "Parent_Education_Level":           "parent_education",
    "State_Region":                     "region_state",
    "Family_Income_Bracket":            "family_income",
    "Family_Size":                      "family_size",
    "Internet_Access_At_Home":          "internet_access",
    "Scholarship_Received":             "scholarship",
    "Age":                              "age",
    "Gender":                           "gender",
    "Region_Type":                      "region_type",
    "Part_Time_Job_Status":             "part_time_job",
    # Target
    "Predicted_Next_CGPA":              "next_semester_cgpa",
}

# CSV columns that will be intentionally ignored (not part of our feature schema)
IGNORED_CSV_COLUMNS = {
    "Preferred_Language",
    "Educational_Stage",
    "Part_Time_Job_Hours",
    "Higher_Study_Or_Job",
    "Performance_Category",
}

# ── categorical encodings ──────────────────────────────────────────────────────
CATEGORICAL_MAPS = {
    "parent_education": {"Illiterate": 0, "School": 1, "Graduate": 2, "PostGraduate": 3},
    "region_state":     {"Bihar": 0, "UP": 1, "Rajasthan": 2, "Kerala": 3, "West Bengal": 4, "Other": 5},
    "family_income":    {"Low": 0, "Medium": 1, "High": 2},
    "gender":           {"Male": 0, "Female": 1, "Other": 2},
    "region_type":      {"Rural": 0, "Urban": 1},
}

# Binary columns stored as "Yes"/"No" in the CSV  -> 1/0
BINARY_YES_NO_COLS = {"internet_access", "scholarship", "part_time_job"}

# ── ordered feature column list ────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Academic
    "current_cgpa", "matric_10th", "inter_12th", "attendance_pct",
    "daily_study_hours", "backlogs", "assignment_rate",
    # Mental / Behavioral
    "stress_level", "self_confidence", "time_mgmt", "exam_anxiety", "motivation_level",
    # Lifestyle
    "sleep_hours", "sleep_quality", "screen_time_hours", "social_media_usage",
    "physical_health", "diet_quality", "extracurricular_hours",
    # Demographic
    "financial_support", "parent_education", "region_state", "family_income",
    "family_size", "internet_access", "scholarship",
    # Personal
    "age", "gender", "region_type", "part_time_job",
]


# ── data loading & preprocessing ───────────────────────────────────────────────
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load student_data.csv, rename columns to internal feature names,
    encode categorical and binary columns, and drop unused columns.

    Returns a clean DataFrame with FEATURE_COLUMNS + next_semester_cgpa.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"student_data.csv not found at: {csv_path}\n"
            "Place the CSV file inside central_server/ and re-run."
        )

    df_raw = pd.read_csv(csv_path)
    print(f"[init] Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns from {csv_path}")

    # Rename only the columns we know about
    df_raw = df_raw.rename(columns=CSV_TO_FEATURE_MAP)

    # Keep only internal feature columns + target
    keep_cols = FEATURE_COLUMNS + ["next_semester_cgpa"]
    missing_cols = [c for c in keep_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns after rename: {missing_cols}\n"
            f"Available columns: {df_raw.columns.tolist()}"
        )

    df = df_raw[keep_cols].copy()

    # Encode categorical string columns
    for col, mapping in CATEGORICAL_MAPS.items():
        if col in df.columns:
            unknown = set(df[col].dropna().unique()) - set(mapping.keys())
            if unknown:
                raise ValueError(
                    f"Unknown categories in '{col}': {unknown}\n"
                    f"Expected one of: {list(mapping.keys())}"
                )
            df[col] = df[col].map(mapping)

    # Convert Yes/No binary columns -> 1/0
    for col in BINARY_YES_NO_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaN after encoding
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"[init] Dropped {dropped} rows with NaN values after encoding.")

    print(f"[init] Clean dataset: {len(df)} rows, {len(FEATURE_COLUMNS)} features.")
    return df


# ── baseline training ──────────────────────────────────────────────────────────
def train_baseline(df: pd.DataFrame):
    """
    Fit StandardScaler + SGDRegressor on the cleaned dataset.
    Returns (coef, intercept, scaler_mean, scaler_scale).
    """
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["next_semester_cgpa"].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.001,
        max_iter=1000,
        tol=1e-4,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42,
    )
    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)
    print(f"[init] Baseline R² on training set: {r2:.4f}")
    return model.coef_, model.intercept_, scaler.mean_, scaler.scale_


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"[init] Loading real student data from: {DATA_FILE}")
    df = load_and_preprocess(DATA_FILE)

    print("[init] Training baseline SGDRegressor ...")
    coef, intercept, scaler_mean, scaler_scale = train_baseline(df)

    # Save weights
    np.savez(
        WEIGHTS_FILE,
        coef=coef,
        intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        version=np.array(1),
        sample_count=np.array(len(df)),
    )
    print(f"[init] Saved weights  -> {WEIGHTS_FILE}")

    # Save feature metadata
    metadata = {
        "version": 1,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_maps": CATEGORICAL_MAPS,
        "binary_yes_no_cols": list(BINARY_YES_NO_COLS),
        "csv_to_feature_map": CSV_TO_FEATURE_MAP,
        "target": "next_semester_cgpa",
        "performance_tiers": {
            "Excellent": "cgpa >= 8.5",
            "Good":      "cgpa >= 7.0",
            "Average":   "cgpa >= 5.0",
            "At-Risk":   "cgpa < 5.0",
        },
        "training_samples": len(df),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[init] Saved metadata -> {METADATA_FILE}")
    print("[init] Initialisation complete.")


if __name__ == "__main__":
    main()
