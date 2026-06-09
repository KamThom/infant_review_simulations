from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["group"] = df["group"].astype("category")
df["age_from_3mo"] = df["age_months"] - 3

# fixed effects- age, group, and age-by-group interaction
# random effect- each infant gets their own baseline HR
model = smf.mixedlm(
    "resting_hr ~ age_from_3mo * C(group)",
    data=df,
    groups=df["infant_id"],
)

result = model.fit(reml=True)

print("\nMixed-effects model: resting HR trajectory")
print("------------------------------------------")
print(result.summary())


coef_table = result.summary().tables[1]
coef_table.to_csv(TABLES_DIR / "mixed_effects_hr_model_coefficients.csv")

age_grid = [3, 6, 9, 12, 18, 24]
groups = sorted(df["group"].unique())

pred_rows = []
for group in groups:
    for age in age_grid:
        pred_rows.append(
            {
                "group": group,
                "age_months": age,
                "age_from_3mo": age - 3,
            }
        )

pred_df = pd.DataFrame(pred_rows)
pred_df["group"] = pd.Categorical(pred_df["group"], categories=df["group"].cat.categories)

pred_df["predicted_resting_hr"] = result.predict(pred_df)

group_means = (
    df.groupby(["group", "age_months"], observed=True)
    .agg(
        mean_resting_hr=("resting_hr", "mean"),
        sem_resting_hr=("resting_hr", "sem"),
        n=("resting_hr", "size"),
    )
    .reset_index()
)

group_means.to_csv(TABLES_DIR / "group_age_resting_hr_summary.csv", index=False)

plt.figure(figsize=(7, 5))

sample_infants = (
    df["infant_id"]
    .drop_duplicates()
    .sample(n=30, random_state=1)
)

for infant_id in sample_infants:
    infant_df = df[df["infant_id"] == infant_id]
    infant_age_mean = (
        infant_df.groupby("age_months")["resting_hr"]
        .mean()
        .reset_index()
    )
    plt.plot(
        infant_age_mean["age_months"],
        infant_age_mean["resting_hr"],
        linewidth=0.7,
        alpha=0.25,
    )

for group in groups:
    this_pred = pred_df[pred_df["group"] == group]
    plt.plot(
        this_pred["age_months"],
        this_pred["predicted_resting_hr"],
        marker="o",
        linewidth=2.5,
        label=f"{group} fitted trajectory",
    )

plt.xlabel("Age (months)")
plt.ylabel("Resting heart rate (bpm)")
plt.title("Mixed-effects model of resting HR across development")
plt.legend(frameon=False, fontsize=8)
plt.tight_layout()

png_path = FIGURES_DIR / "figure_mixed_effects_hr_trajectory.png"
pdf_path = FIGURES_DIR / "figure_mixed_effects_hr_trajectory.pdf"

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved model coefficients to: {TABLES_DIR / 'mixed_effects_hr_model_coefficients.csv'}")
print(f"Saved group-age summary to: {TABLES_DIR / 'group_age_resting_hr_summary.csv'}")
