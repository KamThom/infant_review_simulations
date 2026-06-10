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

print("\nMixed-effects model for Fleming-style HR reference figure")
print("--------------------------------------------------------")
print(result.summary())


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

prediction_path = TABLES_DIR / "figure_08_fleming_style_model_predictions.csv"
pred_df.to_csv(prediction_path, index=False)


# Approximate visual anchors from Fleming et al. 2011 pediatric heart-rate
# centile charts. These contextualize the simulated trajectory; they are not
# used in the mixed-effects model.
reference_df = pd.DataFrame(
    {
        "age_months": [3, 6, 12, 24],
        "reference_median_hr": [138, 132, 122, 113],
        "reference_lower_hr": [105, 98, 90, 80],
        "reference_upper_hr": [175, 165, 150, 135],
        "source_note": [
            "Approximate visual anchor from Fleming et al. 2011, Lancet, PMID 21411136",
            "Approximate visual anchor from Fleming et al. 2011, Lancet, PMID 21411136",
            "Approximate visual anchor from Fleming et al. 2011, Lancet, PMID 21411136",
            "Approximate visual anchor from Fleming et al. 2011, Lancet, PMID 21411136",
        ],
    }
)

reference_path = TABLES_DIR / "figure_08_pediatric_hr_reference_points.csv"
reference_df.to_csv(reference_path, index=False)


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

summary_path = TABLES_DIR / "figure_08_group_age_resting_hr_summary.csv"
group_age_summary.to_csv(summary_path, index=False)


fig, ax = plt.subplots(figsize=(6.2, 5.2))

ax.fill_between(
    reference_df["age_months"],
    reference_df["reference_lower_hr"],
    reference_df["reference_upper_hr"],
    color="#BFE6F2",
    alpha=0.45,
    linewidth=0,
    label="Reference range",
)

ax.plot(
    reference_df["age_months"],
    reference_df["reference_lower_hr"],
    color="#5EC7DF",
    linewidth=1.0,
)

ax.plot(
    reference_df["age_months"],
    reference_df["reference_upper_hr"],
    color="#5EC7DF",
    linewidth=1.0,
)

ax.plot(
    reference_df["age_months"],
    reference_df["reference_median_hr"],
    color="#70AD47",
    linewidth=2.0,
    label="Reference median",
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


for group in groups:
    this_pred = pred_df[pred_df["group"] == group]
    color = COLORS[group]
    label = GROUP_LABELS[group]

    ax.step(
        this_pred["age_months"],
        this_pred["predicted_resting_hr"],
        where="post",
        color=color,
        linewidth=2.4,
        label=label,
    )

ax.set_xlabel("Age (months)")
ax.set_ylabel("Resting heart rate (bpm)")
ax.set_title("Fleming-style reference view of simulated HR trajectory")

ax.set_xlim(0, 25)
ax.set_ylim(70, 180)

ax.legend(frameon=False, loc="upper right")

fig.tight_layout()

png_path = FIGURES_DIR / "figure_08_fleming_style_hr_reference.png"
pdf_path = FIGURES_DIR / "figure_08_fleming_style_hr_reference.pdf"

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved model predictions to: {prediction_path}")
print(f"Saved group-age summary to: {summary_path}")
print(f"Saved pediatric HR reference points to: {reference_path}")
