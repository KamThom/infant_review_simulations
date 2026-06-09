from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\nLoaded session_features_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Unique infants: {df['infant_id'].nunique()}")
print(f"Groups: {df['group'].value_counts().to_dict()}")


# ------------------------------------------------------------
# Define prediction task
# ------------------------------------------------------------
# Do NOT include infant_id. That would be direct cheating.
# Do NOT include support_active. That directly encodes SUPPORT group after 9 months.
# Do NOT include post_period for now. It is not wrong mathematically, but it muddies the story.
feature_cols = [
    "age_months",
    "resting_hr",
    "hrv_sdnn",
    "mean_rr",
    "pupil_social_response",
    "mean_luminance",
    "ecg_hr",
    "ppg_hr",
    "missing_fraction",
]

X = df[feature_cols]
y = df["group"]
groups = df["infant_id"]


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
# Random forest is intentionally flexible.
# That makes it able to exploit leakage if leakage is present.
model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=7,
                class_weight="balanced",
            ),
        ),
    ]
)

scorer = make_scorer(balanced_accuracy_score)


# ------------------------------------------------------------
# Bad validation: row-wise split
# ------------------------------------------------------------
# This lets rows from the same infant appear in both train and test.
row_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=7,
)

row_scores = cross_val_score(
    model,
    X,
    y,
    cv=row_cv,
    scoring=scorer,
)


# ------------------------------------------------------------
# Better validation: infant-blocked split
# ------------------------------------------------------------
# This keeps all rows from the same infant together.
group_cv = GroupKFold(n_splits=5)

group_scores = cross_val_score(
    model,
    X,
    y,
    cv=group_cv,
    groups=groups,
    scoring=scorer,
)


# ------------------------------------------------------------
# Summarize results
# ------------------------------------------------------------
summary = pd.DataFrame(
    {
        "validation_scheme": ["row_shuffled_cv"] * len(row_scores)
        + ["infant_blocked_cv"] * len(group_scores),
        "fold": list(range(1, len(row_scores) + 1))
        + list(range(1, len(group_scores) + 1)),
        "balanced_accuracy": list(row_scores) + list(group_scores),
    }
)

summary_path = TABLES_DIR / "prediction_leakage_cv_scores.csv"
summary.to_csv(summary_path, index=False)

mean_summary = (
    summary.groupby("validation_scheme")
    .agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        sd_balanced_accuracy=("balanced_accuracy", "std"),
    )
    .reset_index()
)

mean_summary_path = TABLES_DIR / "prediction_leakage_summary.csv"
mean_summary.to_csv(mean_summary_path, index=False)

print("\nPrediction leakage comparison")
print("-----------------------------")
print(mean_summary)

inflation = row_scores.mean() - group_scores.mean()
print(f"\nEstimated inflation from row-wise CV: {inflation:.3f}")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_data = [
    row_scores,
    group_scores,
]

labels = [
    "Row-shuffled CV\n(leaky)",
    "Infant-blocked CV\n(honest)",
]

plt.figure(figsize=(5.5, 4.5))

# Boxplot gives summary, scatter shows fold-to-fold spread.
plt.boxplot(
    plot_data,
    labels=labels,
    showmeans=True,
    widths=0.45,
)

for x_position, scores in enumerate(plot_data, start=1):
    jitter = np.random.default_rng(10).normal(0, 0.035, size=len(scores))
    plt.scatter(
        np.full(len(scores), x_position) + jitter,
        scores,
        s=35,
        alpha=0.8,
    )

plt.ylabel("Balanced accuracy")
plt.title("Prediction performance depends on validation design")
plt.axhline(1/3, linestyle="--", linewidth=1)
plt.text(
    2.15,
    1/3,
    "chance",
    va="center",
    fontsize=8,
)

plt.ylim(0.30, 0.82)

plt.tight_layout()

png_path = FIGURES_DIR / "figure_prediction_leakage.png"
pdf_path = FIGURES_DIR / "figure_prediction_leakage.pdf"

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved fold scores to: {summary_path}")
print(f"Saved summary to: {mean_summary_path}")