from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_style import COLORS, apply_style


apply_style()

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "dense_timeseries_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)

print("\nLoaded dense_timeseries_df")
print("--------------------------")
print(f"Rows: {len(df)}")
print(f"Duration: {df['time_s'].min()} to {df['time_s'].max()} seconds")
print(f"RR missing samples: {df['rr_sensor'].isna().sum()} / {len(df)}")


observed_rr = df["rr_sensor"].to_numpy()
true_rr = df["latent_rr"].to_numpy()
time_s = df["time_s"].to_numpy()

n = len(df)

kalman_estimate = np.zeros(n)


valid_initial_values = observed_rr[~np.isnan(observed_rr)][:20]
x = np.mean(valid_initial_values)

P = 5.0
Q = 0.05
R = 5.00

for t in range(n):

    x_pred = x
    P_pred = P + Q
    z = observed_rr[t]

    if np.isnan(z):
        x = x_pred
        P = P_pred
    else:
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

    kalman_estimate[t] = x


df["kalman_rr_estimate"] = kalman_estimate

available = ~np.isnan(observed_rr)

sensor_rmse_available = np.sqrt(
    np.mean((observed_rr[available] - true_rr[available]) ** 2)
)

kalman_rmse_available = np.sqrt(
    np.mean((kalman_estimate[available] - true_rr[available]) ** 2)
)

kalman_rmse_all = np.sqrt(
    np.mean((kalman_estimate - true_rr) ** 2)
)

summary = pd.DataFrame(
    {
        "metric": [
            "sensor_rmse_available_samples",
            "kalman_rmse_available_samples",
            "kalman_rmse_all_samples",
            "missing_rr_samples",
            "total_samples",
            "Q_process_noise",
            "R_measurement_noise",
        ],
        "value": [
            sensor_rmse_available,
            kalman_rmse_available,
            kalman_rmse_all,
            int(df["rr_sensor"].isna().sum()),
            len(df),
            Q,
            R,
        ],
    }
)

summary_path = TABLES_DIR / "kalman_rr_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nKalman respiratory-rate recovery")
print("--------------------------------")
print(summary)


fig, ax = plt.subplots(figsize=(8.2, 4.4))

in_dropout = False
start = None

for i, is_missing in enumerate(df["rr_dropout"]):
    if is_missing and not in_dropout:
        start = df.loc[i, "time_s"]
        in_dropout = True

    if in_dropout and (not is_missing or i == len(df) - 1):
        end = df.loc[i, "time_s"]
        ax.axvspan(
            start,
            end,
            color=COLORS["gray_light"],
            alpha=0.6,
            linewidth=0,
        )
        in_dropout = False

ax.scatter(
    df["time_s"],
    df["rr_sensor"],
    s=13,
    alpha=0.35,
    color=COLORS["gray_mid"],
    edgecolors="none",
    label="Noisy sensor observations",
)


ax.plot(
    df["time_s"],
    df["latent_rr"],
    linestyle="--",
    linewidth=2.0,
    color=COLORS["black"],
    label="Known latent RR, simulated",
)

ax.plot(
    df["time_s"],
    df["kalman_rr_estimate"],
    linewidth=2.8,
    color=COLORS["REF"],
    label="State-space estimate",
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Respiratory rate (breaths/min)")
ax.set_title("State-space tracking through sensor noise and dropout")

ax.set_xlim(df["time_s"].min(), df["time_s"].max())

y_min = min(df["latent_rr"].min(), np.nanmin(df["rr_sensor"]), df["kalman_rr_estimate"].min()) - 2
y_max = max(df["latent_rr"].max(), np.nanmax(df["rr_sensor"]), df["kalman_rr_estimate"].max()) + 2
ax.set_ylim(y_min, y_max)

ax.legend(frameon=False, loc="upper right")

fig.tight_layout()

png_path = FIGURES_DIR / "figure_05_state_space_kalman_rr.png"
pdf_path = FIGURES_DIR / "figure_05_state_space_kalman_rr.pdf"

fig.savefig(png_path)
fig.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved summary to: {summary_path}")