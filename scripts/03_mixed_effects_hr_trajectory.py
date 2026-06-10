from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from project_style import COLORS, GROUP_LABELS, apply_style


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

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Groups: {sorted(df['group'].unique())}")
print(f"Ages: {sorted(df['age_months'].unique())}")


model = smf.mixedlm(
    "resting_hr ~ age_from_3mo * C(group, Treatment(reference='REF'))",
    data=df,
    groups=df["infant_id"],
)

result = model.fit(reml=True)

print("\nMixed-effects model: resting HR trajectory")
print("------------------------------------------")
print(result.summary())


coef_table = result.summary().tables[1]
coef_path = TABLES_DIR / "mixed_effects_hr_model_coefficients.csv"
coef_table.to_csv(coef_path)


age_grid = [3, 6, 9, 12, 18, 24]
groups = ["REF", "COMPARISON"]

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

pred_df["group"] = pd.Categorical(
    pred_df["group"],
    categories=df["group"].cat.categories,
)

pred_df["predicted_resting_hr"] = result.predict(pred_df)


group_age_summary = (
    df.groupby(["group", "age_months"], observed=True)
    .agg(
        n_rows=("resting_hr", "size"),
        n_infants=("infant_id", "nunique"),
        mean_resting_hr=("resting_hr", "mean"),
        sem_resting_hr=("resting_hr", "sem"),
    )
    .reset_index()
)

group_age_summary_path = TABLES_DIR / "group_age_resting_hr_summary.csv"
group_age_summary.to_csv(group_age_summary_path, index=False)


fig, ax = plt.subplots(figsize=(7.0, 4.8))

for group in groups:
    group_df = df[df["group"] == group]
    color = COLORS[group]

    for infant_id, infant_df in group_df.groupby("infant_id"):
        infant_age_mean = (
            infant_df.groupby("age_months", observed=True)["resting_hr"]
            .mean()
            .reset_index()
        )

        ax.plot(
            infant_age_mean["age_months"],
            infant_age_mean["resting_hr"],
            color=color,
            linewidth=0.7,
            alpha=0.18,
        )


for group in groups:
    this_pred = pred_df[pred_df["group"] == group]
    color = COLORS[group]
    label = GROUP_LABELS[group]

    ax.plot(
        this_pred["age_months"],
        this_pred["predicted_resting_hr"],
        color=color,
        marker="o",
        markersize=5,
        linewidth=2.8,
        label=label,
    )

for group in groups:
    summary_df = group_age_summary[group_age_summary["group"] == group]
    color = COLORS[group]

    ax.errorbar(
        summary_df["age_months"],
        summary_df["mean_resting_hr"],
        yerr=summary_df["sem_resting_hr"],
        fmt="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor=color,
        ecolor=color,
        elinewidth=1,
        capsize=2,
        alpha=0.9,
    )

ax.set_xlabel("Age (months)")
ax.set_ylabel("Resting heart rate (bpm)")
ax.set_title("Mixed-effects model separates developmental trajectories from infant-level variation")

ax.set_xlim(2.5, 24.5)

ax.set_ylim(100, 155)

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
