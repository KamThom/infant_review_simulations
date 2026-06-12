from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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

infant_age_path = TABLES_DIR / "figure_08_infant_age_resting_hr.csv"
infant_age_df.to_csv(infant_age_path, index=False)


def quantile(probability):
    return lambda values: values.quantile(probability)


pooled_centiles = (
    infant_age_df.groupby("age_months", as_index=False)
    .agg(
        n_infant_age_rows=("resting_hr", "size"),
        p05=("resting_hr", quantile(0.05)),
        p25=("resting_hr", quantile(0.25)),
        p50=("resting_hr", quantile(0.50)),
        p75=("resting_hr", quantile(0.75)),
        p95=("resting_hr", quantile(0.95)),
    )
)

pooled_centile_path = TABLES_DIR / "figure_08_empirical_hr_centiles.csv"
pooled_centiles.to_csv(pooled_centile_path, index=False)

group_medians = (
    infant_age_df.groupby(["group", "age_months"], observed=True, as_index=False)
    .agg(
        n_infants=("infant_id", "nunique"),
        median_resting_hr=("resting_hr", "median"),
        mean_resting_hr=("resting_hr", "mean"),
    )
)

group_median_path = TABLES_DIR / "figure_08_group_median_resting_hr.csv"
group_medians.to_csv(group_median_path, index=False)


fig, ax = plt.subplots(figsize=(5.4, 4.5))

age = pooled_centiles["age_months"].to_numpy()

ax.fill_between(
    age,
    pooled_centiles["p25"].to_numpy(),
    pooled_centiles["p75"].to_numpy(),
    color="#DCEFF5",
    alpha=0.95,
    linewidth=0,
    label="All infants: 25th-75th centile",
    zorder=1,
)

ax.plot(
    age,
    pooled_centiles["p05"],
    color="#56BFD7",
    linewidth=1.0,
    alpha=0.9,
    label="All infants: 5th/95th centile",
    zorder=2,
)

ax.plot(
    age,
    pooled_centiles["p95"],
    color="#56BFD7",
    linewidth=1.0,
    alpha=0.9,
    zorder=2,
)

ax.plot(
    age,
    pooled_centiles["p50"],
    color="#65A844",
    linewidth=2.4,
    label="All infants: median",
    zorder=4,
)

for group in ["REF", "COMPARISON"]:
    this_group = group_medians[group_medians["group"] == group]
    label = "REF group median" if group == "REF" else "Comparison group median"

    ax.plot(
        this_group["age_months"],
        this_group["median_resting_hr"],
        color=COLORS[group],
        marker="o",
        markersize=4,
        linewidth=2.0,
        label=label,
        zorder=5,
    )

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
    fontsize=7.5,
)

fig.tight_layout()

png_path = FIGURES_DIR / "supplemental_08_empirical_hr_centiles.png"
pdf_path = FIGURES_DIR / "supplemental_08_empirical_hr_centiles.pdf"

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
print(f"Saved infant-age data to: {infant_age_path}")
print(f"Saved empirical centiles to: {pooled_centile_path}")
print(f"Saved group medians to: {group_median_path}")
