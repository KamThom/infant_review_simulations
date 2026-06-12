from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from project_style import COLORS, POSITION_COLORS, POSITION_LABELS, apply_style


apply_style()

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "position_windows_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\nLoaded position_windows_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Positions: {sorted(df['true_position'].unique())}")


feature_cols = [
    "movement_intensity",
    "posture_angle",
    "movement_variability",
    "vertical_accel_mean",
    "horizontal_accel_sd",
]

label_order = ["held", "supine", "prone", "sitting"]

X = df[feature_cols]
y = df["true_position"]

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.35,
    random_state=7,
    stratify=y,
)

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                random_state=7,
                class_weight="balanced",
            ),
        ),
    ]
)

model.fit(X.loc[train_idx], y.loc[train_idx])

df["predicted_position"] = model.predict(X)

test_pred = df.loc[test_idx, "predicted_position"]
test_true = df.loc[test_idx, "true_position"]

balanced_acc = balanced_accuracy_score(test_true, test_pred)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=label_order,
)

cm_norm = cm / cm.sum(axis=1, keepdims=True)

print("\nAutomated labeling performance")
print("------------------------------")
print(f"Balanced accuracy on held-out windows: {balanced_acc:.3f}")
print("\nConfusion matrix:")
print(pd.DataFrame(cm, index=label_order, columns=label_order))


output_df_path = TABLES_DIR / "position_windows_with_predictions.csv"
df.to_csv(output_df_path, index=False)

cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
cm_path = TABLES_DIR / "position_labeling_confusion_matrix.csv"
cm_df.to_csv(cm_path)

cm_norm_df = pd.DataFrame(cm_norm, index=label_order, columns=label_order)
cm_norm_path = TABLES_DIR / "position_labeling_confusion_matrix_normalized.csv"
cm_norm_df.to_csv(cm_norm_path)

summary_path = TABLES_DIR / "position_labeling_summary.csv"
pd.DataFrame(
    {
        "metric": [
            "model",
            "held_out_balanced_accuracy",
            "n_windows",
            "n_train",
            "n_test",
        ],
        "value": [
            "RandomForestClassifier",
            balanced_acc,
            len(df),
            len(train_idx),
            len(test_idx),
        ],
    }
).to_csv(summary_path, index=False)


fig = plt.figure(figsize=(10.5, 7.0))

grid = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.0, 0.75],
    width_ratios=[1.05, 0.95],
    hspace=0.42,
    wspace=0.32,
)

ax_scatter = fig.add_subplot(grid[0, 0])
ax_cm = fig.add_subplot(grid[0, 1])
ax_timeline = fig.add_subplot(grid[1, :])


for label in label_order:
    this_df = df[df["true_position"] == label]

    ax_scatter.scatter(
        this_df["posture_angle"],
        this_df["movement_intensity"],
        s=26,
        alpha=0.72,
        color=POSITION_COLORS[label],
        edgecolors="white",
        linewidth=0.4,
        label=POSITION_LABELS[label],
    )

ax_scatter.set_xlabel("Posture angle feature")
ax_scatter.set_ylabel("Movement intensity feature")
ax_scatter.legend(frameon=False, loc="upper left", ncol=2)


im = ax_cm.imshow(
    cm_norm,
    vmin=0,
    vmax=1,
    cmap="Greys",
)

ax_cm.set_xticks(np.arange(len(label_order)))
ax_cm.set_yticks(np.arange(len(label_order)))
ax_cm.set_xticklabels([POSITION_LABELS[x] for x in label_order], rotation=35, ha="right")
ax_cm.set_yticklabels([POSITION_LABELS[x] for x in label_order])

ax_cm.set_xlabel("Predicted label")
ax_cm.set_ylabel("True label")

for i in range(len(label_order)):
    for j in range(len(label_order)):
        value = cm_norm[i, j]
        text_color = "white" if value > 0.55 else COLORS["black"]

        ax_cm.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
        )

cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized proportion")


timeline_y = 0

for _, row in df.iterrows():
    label = row["predicted_position"]

    ax_timeline.broken_barh(
        [(row["time_start_s"] / 3600, (row["time_end_s"] - row["time_start_s"]) / 3600)],
        (timeline_y, 1),
        facecolors=POSITION_COLORS[label],
        edgecolors="none",
    )

ax_timeline.set_xlabel("Session time (hours)")
ax_timeline.set_yticks([])
ax_timeline.set_ylim(0, 1)
ax_timeline.set_xlim(df["time_start_s"].min() / 3600, df["time_end_s"].max() / 3600)

legend_handles = []
for label in label_order:
    handle = plt.Line2D(
        [0],
        [0],
        color=POSITION_COLORS[label],
        linewidth=8,
        label=POSITION_LABELS[label],
    )
    legend_handles.append(handle)

ax_timeline.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=4,
)


png_path = FIGURES_DIR / "figure_04_automated_position_labeling.png"
pdf_path = FIGURES_DIR / "figure_04_automated_position_labeling.pdf"

fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved predictions to: {output_df_path}")
print(f"Saved confusion matrix to: {cm_path}")
print(f"Saved normalized confusion matrix to: {cm_norm_path}")
print(f"Saved summary to: {summary_path}")
