from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

TABLES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Groups: {sorted(df['group'].unique())}")
print(f"Ages: {sorted(df['age_months'].unique())}")

print("\nFirst 10 rows")
print("-------------")
print(df.head(10))

print("\nRows per infant")
print("---------------")
rows_per_infant = df.groupby("infant_id").size()
print(rows_per_infant.describe())

print("\nAge-level summary")
print("-----------------")
age_summary = (
    df.groupby("age_months")
    .agg(
        n_rows=("infant_id", "size"),
        n_infants=("infant_id", "nunique"),
        resting_hr_mean=("resting_hr", "mean"),
        hrv_sdnn_mean=("hrv_sdnn", "mean"),
        mean_rr_mean=("mean_rr", "mean"),
        pupil_response_mean=("pupil_response", "mean"),
        ecg_hr_mean=("ecg_hr", "mean"),
        ppg_hr_mean=("ppg_hr", "mean"),
    )
    .reset_index()
)

age_summary["ppg_minus_ecg_mean"] = (
    df.assign(ppg_minus_ecg=df["ppg_hr"] - df["ecg_hr"])
    .groupby("age_months")["ppg_minus_ecg"]
    .mean()
    .values
)

age_summary = age_summary.round(2)

print(age_summary)

out_path = TABLES_DIR / "age_level_summary.csv"
age_summary.to_csv(out_path, index=False)

print(f"\nSaved summary table to: {out_path}")
