from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from project_style import COLORS, apply_style


apply_style()

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)

if "age_from_3mo" not in df.columns:
    df["age_from_3mo"] = df["age_months"] - 3

df["age_years_after_3mo"] = df["age_from_3mo"] / 12.0

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Ages: {sorted(df['age_months'].unique())}")


model = smf.mixedlm(
    "resting_hr ~ age_years_after_3mo",
    data=df,
    groups=df["infant_id"],
    re_formula="~age_years_after_3mo",
)

result = model.fit(reml=True, method="lbfgs", maxiter=1000)

print("\nMixed-effects model: longitudinal resting HR")
print("--------------------------------------------")
print(result.summary())


coef_table = result.summary().tables[1]
coef_path = TABLES_DIR / "figure_03b_mixed_effects_hr_model_coefficients.csv"
coef_table.to_csv(coef_path)


age_grid = np.array([3, 6, 9, 12, 18, 24])
age_plot = np.linspace(3, 24, 100)

pred_df = pd.DataFrame(
    {
        "age_months": age_plot,
        "age_from_3mo": age_plot - 3,
        "age_years_after_3mo": (age_plot - 3) / 12.0,
    }
)
pred_df["predicted_resting_hr"] = result.predict(pred_df)

fixed_effect_names = ["Intercept", "age_years_after_3mo"]
cov_fe = result.cov_params().loc[fixed_effect_names, fixed_effect_names].to_numpy()
design = np.column_stack(
    [
        np.ones(len(pred_df)),
        pred_df["age_years_after_3mo"].to_numpy(),
    ]
)
pred_se = np.sqrt(np.sum((design @ cov_fe) * design, axis=1))
ci_multiplier = 2.58
pred_df["ci_lower"] = pred_df["predicted_resting_hr"] - ci_multiplier * pred_se
pred_df["ci_upper"] = pred_df["predicted_resting_hr"] + ci_multiplier * pred_se


infant_age_df = (
    df.groupby(["infant_id", "age_months"], observed=True, as_index=False)
    .agg(
        resting_hr=("resting_hr", "mean"),
    )
)

age_summary = (
    infant_age_df.groupby("age_months", observed=True)
    .agg(
        n_infant_age_rows=("resting_hr", "size"),
        n_infants=("infant_id", "nunique"),
        mean_resting_hr=("resting_hr", "mean"),
        sem_resting_hr=("resting_hr", "sem"),
    )
    .reset_index()
)
age_summary["ci95_resting_hr"] = 1.96 * age_summary["sem_resting_hr"]

age_summary_path = TABLES_DIR / "figure_03b_population_age_resting_hr_summary.csv"
age_summary.to_csv(age_summary_path, index=False)

pred_path = TABLES_DIR / "figure_03b_mixed_effects_hr_population_predictions.csv"
pred_df.to_csv(pred_path, index=False)


example_infants = (
    infant_age_df["infant_id"]
    .drop_duplicates()
    .sort_values()
    .head(30)
    .to_frame()
)

example_infants_path = TABLES_DIR / "figure_03b_example_infant_trajectories.csv"
example_infants.to_csv(example_infants_path, index=False)


fig, ax = plt.subplots(figsize=(7.1, 4.8))

example_ids = set(example_infants["infant_id"])
example_df = infant_age_df[infant_age_df["infant_id"].isin(example_ids)]

for infant_id, infant_df in example_df.groupby("infant_id"):
    ax.plot(
        infant_df["age_months"],
        infant_df["resting_hr"],
        color=COLORS["gray_mid"],
        linewidth=0.8,
        alpha=0.22,
        zorder=1,
    )

ax.fill_between(
    pred_df["age_months"],
    pred_df["ci_lower"],
    pred_df["ci_upper"],
    color=COLORS["REF"],
    alpha=0.18,
    linewidth=0,
    label="Population model CI",
    zorder=2,
)

ax.plot(
    pred_df["age_months"],
    pred_df["predicted_resting_hr"],
    color=COLORS["REF"],
    linewidth=3.0,
    label="Population trajectory",
    zorder=4,
)

ax.errorbar(
    age_summary["age_months"],
    age_summary["mean_resting_hr"],
    yerr=age_summary["ci95_resting_hr"],
    fmt="o",
    markersize=4,
    markerfacecolor="white",
    markeredgecolor=COLORS["black"],
    ecolor=COLORS["black"],
    elinewidth=1,
    capsize=2,
    alpha=0.95,
    label="Infant-age means, 95% CI",
    zorder=5,
)

ax.set_xlabel("Age (months)")
ax.set_ylabel("Resting heart rate (bpm)")
ax.set_title(
    "Mixed effects model captures longitudinal resting heart rate\n"
    "for the population and individual infants"
)

ax.set_xlim(2.5, 24.5)
ax.set_xticks(age_grid)

ax.set_ylim(95, 160)

ax.legend(frameon=False, loc="upper right")

fig.tight_layout()

png_path = FIGURES_DIR / "figure_03b_mixed_effects_hr_longitudinal_individual.png"
pdf_path = FIGURES_DIR / "figure_03b_mixed_effects_hr_longitudinal_individual.pdf"

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved model coefficients to: {coef_path}")
print(f"Saved age summary to: {age_summary_path}")
print(f"Saved population predictions to: {pred_path}")
print(f"Saved example infant IDs to: {example_infants_path}")
