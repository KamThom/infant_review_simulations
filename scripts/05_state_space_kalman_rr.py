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
kalman_uncertainty = np.zeros(n)


first_valid_idx = np.where(~np.isnan(observed_rr))[0][0]
x = observed_rr[first_valid_idx]

P = 10.0
Q = 0.20
R = 4.00

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
    kalman_uncertainty[t] = P


df["kalman_rr_estimate"] = kalman_estimate
df["kalman_uncertainty"] = kalman_uncertainty

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
        ],
        "value": [
            sensor_rmse_available,
            kalman_rmse_available,
            kalman_rmse_all,
            int(df["rr_sensor"].isna().sum()),
            len(df),
        ],
    }
)

summary_path = TABLES_DIR / "kalman_rr_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nKalman respiratory-rate recovery")
print("--------------------------------")
print(summary)


fig, ax = plt.subplots(figsize=(9.0, 4.8))

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
    df["kalman_rr_estimate"],
    linewidth=2.8,
    color=COLORS["REF"],
    label="State-space estimate",
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Respiratory rate (breaths/min)")
ax.set_title("State-space estimate tracks respiratory rate through dropout")

ax.set_xlim(df["time_s"].min(), df["time_s"].max())

y_min = min(np.nanmin(df["rr_sensor"]), df["kalman_rr_estimate"].min()) - 2
y_max = max(np.nanmax(df["rr_sensor"]), df["kalman_rr_estimate"].max()) + 2
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
