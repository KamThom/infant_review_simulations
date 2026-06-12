from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

y = (df["group"] == "COMPARISON").astype(int)
groups = df["infant_id"]


deployable_features = [
    "pupil_response",
    "mean_luminance",
    "social_exposure",
    "hrv_sdnn",
    "missing_fraction",
]

leaky_feature = "recording_context_code"

leaky_features = deployable_features + [leaky_feature]


def make_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=7,
                ),
            ),
        ]
    )


cv = GroupKFold(n_splits=5)

records = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, y, groups=groups), start=1):
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    X_train_deploy = df.iloc[train_idx][deployable_features]
    X_test_deploy = df.iloc[test_idx][deployable_features]

    deploy_model = make_model()
    deploy_model.fit(X_train_deploy, y_train)

    y_pred_deploy = deploy_model.predict(X_test_deploy)

    deploy_score = balanced_accuracy_score(y_test, y_pred_deploy)

    records.append(
        {
            "fold": fold_idx,
            "condition": "Deployable features only",
            "balanced_accuracy": deploy_score,
        }
    )

    X_train_leak = df.iloc[train_idx][leaky_features]
    X_test_leak = df.iloc[test_idx][leaky_features]

    leak_model = make_model()
    leak_model.fit(X_train_leak, y_train)

    y_pred_leak = leak_model.predict(X_test_leak)

    leak_score = balanced_accuracy_score(y_test, y_pred_leak)

    records.append(
        {
            "fold": fold_idx,
            "condition": "Leaked feature available",
            "balanced_accuracy": leak_score,
        }
    )


    X_test_leak_removed = X_test_leak.copy()
    X_test_leak_removed[leaky_feature] = X_train_leak[leaky_feature].mean()

    y_pred_removed = leak_model.predict(X_test_leak_removed)

    removed_score = balanced_accuracy_score(y_test, y_pred_removed)

    records.append(
        {
            "fold": fold_idx,
            "condition": "Leaky model at deployment",
            "balanced_accuracy": removed_score,
        }
    )


scores_df = pd.DataFrame(records)

scores_path = TABLES_DIR / "figure_07_leakage_failure_fold_scores.csv"
scores_df.to_csv(scores_path, index=False)

summary_df = (
    scores_df.groupby("condition")
    .agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        sd_balanced_accuracy=("balanced_accuracy", "std"),
    )
    .reset_index()
)

summary_path = TABLES_DIR / "figure_07_leakage_failure_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nLeakage failure summary")
print("-----------------------")
print(summary_df)


condition_order = [
    "Deployable features only",
    "Leaked feature available",
    "Leaky model at deployment",
]

plot_colors = {
    "Deployable features only": COLORS["gray_mid"],
    "Leaked feature available": COLORS["red"],
    "Leaky model at deployment": COLORS["black"],
}

x_positions = np.arange(len(condition_order))

fig, ax = plt.subplots(figsize=(7.0, 4.6))

means = []
stds = []

for condition in condition_order:
    vals = scores_df.loc[
        scores_df["condition"] == condition,
        "balanced_accuracy",
    ]
    means.append(vals.mean())
    stds.append(vals.std())

ax.bar(
    x_positions,
    means,
    yerr=stds,
    capsize=4,
    color=[plot_colors[c] for c in condition_order],
    alpha=0.85,
    width=0.62,
)


rng = np.random.default_rng(10)

for i, condition in enumerate(condition_order):
    vals = scores_df.loc[
        scores_df["condition"] == condition,
        "balanced_accuracy",
    ].to_numpy()

    jitter = rng.normal(0, 0.035, size=len(vals))

    ax.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        color="white",
        edgecolor=COLORS["black"],
        linewidth=0.7,
        s=35,
        zorder=3,
    )


ax.axhline(
    0.5,
    color=COLORS["black"],
    linestyle="--",
    linewidth=1,
    alpha=0.75,
)

ax.text(
    len(condition_order) - 0.35,
    0.515,
    "chance",
    ha="right",
    va="bottom",
    fontsize=8,
)

ax.set_xticks(x_positions)
ax.set_xticklabels(
    [
        "Deployable\nfeatures only",
        "Leaked feature\navailable",
        "Leaky model\nat deployment",
    ]
)

ax.set_ylabel("Balanced accuracy")
ax.set_title("Feature leakage can create apparent predictive performance")

ax.set_ylim(0.40, 1.04)

fig.tight_layout()

png_path = FIGURES_DIR / "supplemental_07_leakage_failure.png"
pdf_path = FIGURES_DIR / "supplemental_07_leakage_failure.pdf"

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved fold scores to: {scores_path}")
print(f"Saved summary to: {summary_path}")
