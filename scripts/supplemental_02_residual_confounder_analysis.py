from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

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

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Groups: {sorted(df['group'].unique())}")


model_omitted = smf.ols(
    "pupil_response ~ social_exposure + mean_luminance",
    data=df,
).fit()

model_adjusted = smf.ols(
    "pupil_response ~ social_exposure + mean_luminance + context_bump",
    data=df,
).fit()

df["resid_omitted_context"] = model_omitted.resid
df["resid_adjusted_context"] = model_adjusted.resid


omitted_coef = model_omitted.summary2().tables[1].reset_index()
omitted_coef = omitted_coef.rename(columns={"index": "term"})
omitted_coef["model"] = "context_omitted"

adjusted_coef = model_adjusted.summary2().tables[1].reset_index()
adjusted_coef = adjusted_coef.rename(columns={"index": "term"})
adjusted_coef["model"] = "context_included"

coef_table = pd.concat([omitted_coef, adjusted_coef], ignore_index=True)
coef_path = TABLES_DIR / "residual_confounder_model_coefficients.csv"
coef_table.to_csv(coef_path, index=False)


df["context_bin"] = pd.cut(
    df["context_activity_z"],
    bins=18,
)

resid_summary = (
    df.groupby("context_bin", observed=True)
    .agg(
        context_mid=("context_activity_z", "mean"),
        omitted_resid_mean=("resid_omitted_context", "mean"),
        adjusted_resid_mean=("resid_adjusted_context", "mean"),
        n=("infant_id", "size"),
    )
    .reset_index(drop=True)
)

summary_path = TABLES_DIR / "residual_confounder_binned_summary.csv"
resid_summary.to_csv(summary_path, index=False)


print("\nOmitted-context model")
print("---------------------")
print(model_omitted.summary())

print("\nAdjusted-context model")
print("----------------------")
print(model_adjusted.summary())

print(f"\nSaved coefficients to: {coef_path}")
print(f"Saved residual summary to: {summary_path}")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True)

plot_specs = [
    {
        "ax": axes[0],
        "resid_col": "resid_omitted_context",
        "title": "A. Context omitted",
        "smooth_color": COLORS["red"],
    },
    {
        "ax": axes[1],
        "resid_col": "resid_adjusted_context",
        "title": "B. Context included",
        "smooth_color": COLORS["black"],
    },
]

for spec in plot_specs:
    ax = spec["ax"]
    resid_col = spec["resid_col"]

    ax.scatter(
        df["context_activity_z"],
        df[resid_col],
        s=12,
        alpha=0.18,
        color=COLORS["gray_mid"],
        edgecolors="none",
    )

    smoothed = lowess(
        endog=df[resid_col],
        exog=df["context_activity_z"],
        frac=0.25,
        return_sorted=True,
    )

    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color=spec["smooth_color"],
        linewidth=2.6,
    )

    ax.axhline(
        0,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(spec["title"])
    ax.set_xlabel("Context activity, known in simulation (z)")

axes[0].set_ylabel("Regression residual")

fig.suptitle(
    "Residual structure reveals an omitted context effect",
    y=1.02,
)

fig.tight_layout()

png_path = FIGURES_DIR / "supplemental_02_residual_confounder_analysis.png"
pdf_path = FIGURES_DIR / "supplemental_02_residual_confounder_analysis.pdf"

fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
