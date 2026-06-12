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
df["group"] = df["group"].astype("category")

if "age_from_3mo" not in df.columns:
    df["age_from_3mo"] = df["age_months"] - 3

df["age_years_after_3mo"] = df["age_from_3mo"] / 12.0

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Groups: {sorted(df['group'].unique())}")
print(f"Ages: {sorted(df['age_months'].unique())}")


model = smf.mixedlm(
    "resting_hr ~ age_years_after_3mo * C(group, Treatment(reference='REF'))",
    data=df,
    groups=df["infant_id"],
    re_formula="~age_years_after_3mo",
)

result = model.fit(reml=True, method="lbfgs", maxiter=1000)

print("\nMixed-effects model: resting HR trajectory")
print("------------------------------------------")
print(result.summary())


coef_table = result.summary().tables[1]
coef_path = TABLES_DIR / "mixed_effects_hr_model_coefficients.csv"
coef_table.to_csv(coef_path)


age_grid = np.array([3, 6, 9, 12, 18, 24])
age_plot = np.linspace(3, 24, 100)
groups = ["REF", "COMPARISON"]

pred_rows = []
for group in groups:
    for age in age_plot:
        pred_rows.append(
            {
                "group": group,
                "age_months": age,
                "age_from_3mo": age - 3,
                "age_years_after_3mo": (age - 3) / 12.0,
            }
        )

pred_df = pd.DataFrame(pred_rows)
pred_df["group"] = pd.Categorical(
    pred_df["group"],
    categories=df["group"].cat.categories,
)
pred_df["predicted_resting_hr"] = result.predict(pred_df)

fixed_effect_names = list(result.fe_params.index)
cov_fe = result.cov_params().loc[fixed_effect_names, fixed_effect_names].to_numpy()
comparison_term = "C(group, Treatment(reference='REF'))[T.COMPARISON]"
interaction_term = (
    "age_years_after_3mo:"
    "C(group, Treatment(reference='REF'))[T.COMPARISON]"
)
group_indicator = (pred_df["group"].astype(str) == "COMPARISON").astype(float)
design_df = pd.DataFrame(
    {
        "Intercept": 1.0,
        comparison_term: group_indicator,
        "age_years_after_3mo": pred_df["age_years_after_3mo"],
        interaction_term: pred_df["age_years_after_3mo"] * group_indicator,
    }
)
design = design_df[fixed_effect_names].to_numpy()
pred_se = np.sqrt(np.sum((design @ cov_fe) * design, axis=1))
ci_multiplier = 2.58
pred_df["ci_lower"] = pred_df["predicted_resting_hr"] - ci_multiplier * pred_se
pred_df["ci_upper"] = pred_df["predicted_resting_hr"] + ci_multiplier * pred_se


infant_age_df = (
    df.groupby(["infant_id", "group", "age_months"], observed=True, as_index=False)
    .agg(
        resting_hr=("resting_hr", "mean"),
    )
)

group_age_summary = (
    infant_age_df.groupby(["group", "age_months"], observed=True)
    .agg(
        n_infant_age_rows=("resting_hr", "size"),
        n_infants=("infant_id", "nunique"),
        mean_resting_hr=("resting_hr", "mean"),
        sem_resting_hr=("resting_hr", "sem"),
    )
    .reset_index()
)
group_age_summary["ci95_resting_hr"] = 1.96 * group_age_summary["sem_resting_hr"]

group_age_summary_path = TABLES_DIR / "group_age_resting_hr_summary.csv"
group_age_summary.to_csv(group_age_summary_path, index=False)

pred_path = TABLES_DIR / "mixed_effects_hr_population_predictions.csv"
pred_df.to_csv(pred_path, index=False)


example_infants = (
    infant_age_df["infant_id"]
    .drop_duplicates()
    .sort_values()
    .head(30)
    .to_frame()
)

example_infants_path = TABLES_DIR / "figure_03_example_infant_trajectories.csv"
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

marker_by_group = {"REF": "o", "COMPARISON": "s"}
label_by_group = {"REF": "Reference population", "COMPARISON": "Comparison population"}

for group in groups:
    this_pred = pred_df[pred_df["group"] == group]
    this_summary = group_age_summary[group_age_summary["group"] == group]
    color = COLORS[group]

    ax.fill_between(
        this_pred["age_months"],
        this_pred["ci_lower"],
        this_pred["ci_upper"],
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        this_pred["age_months"],
        this_pred["predicted_resting_hr"],
        color=color,
        linewidth=3.0,
        label=label_by_group[group],
        zorder=4,
    )

    ax.errorbar(
        this_summary["age_months"],
        this_summary["mean_resting_hr"],
        yerr=this_summary["ci95_resting_hr"],
        fmt=marker_by_group[group],
        markersize=4,
        markerfacecolor="white",
        markeredgecolor=color,
        ecolor=color,
        elinewidth=1,
        capsize=2,
        alpha=0.95,
        zorder=5,
    )

ax.set_xlabel("Age (months)")
ax.set_ylabel("Resting heart rate (bpm)")
ax.set_title(
    "Mixed effects model tracks resting heart rate development\n"
    "for population groups and individual infants"
)

ax.set_xlim(2.5, 24.5)
ax.set_xticks(age_grid)

ax.set_ylim(95, 160)

ax.legend(frameon=False, loc="upper right")

fig.tight_layout()

png_path = FIGURES_DIR / "figure_03_mixed_effects_hr_trajectory.png"
pdf_path = FIGURES_DIR / "figure_03_mixed_effects_hr_trajectory.pdf"

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved model coefficients to: {coef_path}")
print(f"Saved group-age summary to: {group_age_summary_path}")
print(f"Saved population predictions to: {pred_path}")
print(f"Saved example infant IDs to: {example_infants_path}")
