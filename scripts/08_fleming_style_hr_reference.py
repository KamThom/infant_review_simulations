from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from patsy import build_design_matrices
from scipy.interpolate import PchipInterpolator

from project_style import COLORS, GROUP_LABELS, apply_style


apply_style()

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["group"] = df["group"].astype("category")
df["age_from_3mo"] = df["age_months"] - 3


# ---------------------------------------------------------------------
# Fit mixed-effects model
# ---------------------------------------------------------------------

model = smf.mixedlm(
    "resting_hr ~ age_from_3mo * C(group, Treatment(reference='REF'))",
    data=df,
    groups=df["infant_id"],
)

result = model.fit(reml=True)

print(result.summary())


# ---------------------------------------------------------------------
# Participant-level observed summaries
#
# Average repeated sessions within infant and age first.
# This prevents sessions from being treated as independent infants.
# ---------------------------------------------------------------------

infant_age_df = (
    df.groupby(
        ["infant_id", "group", "age_months"],
        observed=True,
        as_index=False,
    )
    .agg(
        resting_hr=("resting_hr", "mean"),
    )
)

group_age_summary = (
    infant_age_df.groupby(
        ["group", "age_months"],
        observed=True,
        as_index=False,
    )
    .agg(
        n_infants=("infant_id", "nunique"),
        mean_resting_hr=("resting_hr", "mean"),
        sem_resting_hr=("resting_hr", "sem"),
    )
)

group_age_summary_path = TABLES_DIR / "figure_08_group_age_resting_hr_summary.csv"
group_age_summary.to_csv(group_age_summary_path, index=False)


# ---------------------------------------------------------------------
# Dense population-level predictions
# ---------------------------------------------------------------------

groups = ["REF", "COMPARISON"]
age_dense = np.linspace(3, 24, 250)

pred_df = pd.DataFrame(
    [
        {
            "group": group,
            "age_months": age,
            "age_from_3mo": age - 3,
        }
        for group in groups
        for age in age_dense
    ]
)

pred_df["group"] = pd.Categorical(
    pred_df["group"],
    categories=df["group"].cat.categories,
)

# Reconstruct the fixed-effect design matrix used by the model.
design_matrix = build_design_matrices(
    [result.model.data.design_info],
    pred_df,
    return_type="dataframe",
)[0]

fixed_effects = result.fe_params

# cov_params() includes fixed- and random-effect parameters.
# Retain only the fixed-effect covariance block.
fixed_covariance = result.cov_params().loc[
    fixed_effects.index,
    fixed_effects.index,
]

X = design_matrix.loc[:, fixed_effects.index].to_numpy()
beta = fixed_effects.to_numpy()
cov_beta = fixed_covariance.to_numpy()

predicted = X @ beta

prediction_variance = np.einsum(
    "ij,jk,ik->i",
    X,
    cov_beta,
    X,
)

prediction_se = np.sqrt(
    np.clip(prediction_variance, a_min=0, a_max=None)
)

pred_df["predicted_resting_hr"] = predicted
pred_df["ci_lower"] = predicted - 1.96 * prediction_se
pred_df["ci_upper"] = predicted + 1.96 * prediction_se

prediction_path = TABLES_DIR / "figure_08_fleming_style_model_predictions.csv"
pred_df.to_csv(prediction_path, index=False)


# ---------------------------------------------------------------------
# Approximate external pediatric reference anchors
#
# These are illustrative anchors, not values used by the model.
# ---------------------------------------------------------------------

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

# Shape-preserving interpolation gives smoother curves without spline
# overshoot. It does not create new scientific information.
reference_age = np.linspace(3, 24, 250)

reference_lower = PchipInterpolator(
    reference_df["age_months"],
    reference_df["reference_lower_hr"],
)(reference_age)

reference_median = PchipInterpolator(
    reference_df["age_months"],
    reference_df["reference_median_hr"],
)(reference_age)

reference_upper = PchipInterpolator(
    reference_df["age_months"],
    reference_df["reference_upper_hr"],
)(reference_age)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.2, 4.5))

reference_color = "#56BFD7"
median_color = "#65A844"

# Very light envelope, subordinate to the cohort trajectories.
ax.fill_between(
    reference_age,
    reference_lower,
    reference_upper,
    color=reference_color,
    alpha=0.10,
    linewidth=0,
    label="Approximate pediatric envelope",
    zorder=1,
)

ax.plot(
    reference_age,
    reference_lower,
    color=reference_color,
    linewidth=1.0,
    zorder=2,
)

ax.plot(
    reference_age,
    reference_upper,
    color=reference_color,
    linewidth=1.0,
    zorder=2,
)

ax.plot(
    reference_age,
    reference_median,
    color=median_color,
    linewidth=2.0,
    label="Approximate pediatric median",
    zorder=3,
)


# Model trajectories and fixed-effect confidence intervals
for group in groups:
    this_pred = pred_df[pred_df["group"] == group]

    color = COLORS[group]

    # Avoid the external-reference / REF-group naming collision.
    if group == "REF":
        label = "Simulated REF cohort"
    else:
        label = GROUP_LABELS[group]

    ax.fill_between(
        this_pred["age_months"].to_numpy(),
        this_pred["ci_lower"].to_numpy(),
        this_pred["ci_upper"].to_numpy(),
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=4,
    )

    ax.plot(
        this_pred["age_months"],
        this_pred["predicted_resting_hr"],
        color=color,
        linewidth=2.4,
        label=label,
        zorder=5,
    )


# Observed participant-level means
for group in groups:
    this_summary = group_age_summary[
        group_age_summary["group"] == group
    ]

    color = COLORS[group]

    ax.scatter(
        this_summary["age_months"],
        this_summary["mean_resting_hr"],
        s=24,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2,
        zorder=6,
    )


# Fleming-like restraint: use a panel label rather than a large title.
ax.text(
    -0.11,
    1.02,
    "A",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    va="bottom",
)

ax.set_xlabel("Age (months)")
ax.set_ylabel("Resting heart rate (bpm)")

ax.set_xlim(2.5, 24.5)
ax.set_xticks([3, 6, 9, 12, 18, 24])

ax.set_ylim(75, 180)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=8,
)

fig.tight_layout()

png_path = FIGURES_DIR / "figure_08_hr_reference_trajectory.png"
pdf_path = FIGURES_DIR / "figure_08_hr_reference_trajectory.pdf"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved model predictions to: {prediction_path}")
print(f"Saved group-age summary to: {group_age_summary_path}")
print(f"Saved pediatric HR reference points to: {reference_path}")
