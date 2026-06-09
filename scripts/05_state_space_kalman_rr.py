from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "dense_timeseries_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Load dense time-series data
# ------------------------------------------------------------
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

# Storage for estimates
kalman_estimate = np.zeros(n)
kalman_uncertainty = np.zeros(n)

# Initial estimate:
# use the first available sensor value
first_valid_idx = np.where(~np.isnan(observed_rr))[0][0]
x = observed_rr[first_valid_idx]

# Initial uncertainty:
# larger number means "not very sure yet"
P = 10.0

# Process noise:
# how much we allow true RR to drift each second
Q = 0.20

# Measurement noise:
# how noisy we think the RR sensor is
R = 4.00

for t in range(n):
    # -------------------------
    # Predict step
    # -------------------------
    # We assume RR at this second is close to RR from previous second.
    x_pred = x
    P_pred = P + Q

    # -------------------------
    # Update step
    # -------------------------
    z = observed_rr[t]

    if np.isnan(z):
        # No sensor measurement.
        # Keep prediction and let uncertainty grow.
        x = x_pred
        P = P_pred
    else:
        # Sensor measurement exists.
        # Kalman gain decides how much to trust sensor vs prediction.
        K = P_pred / (P_pred + R)

        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

    kalman_estimate[t] = x
    kalman_uncertainty[t] = P


df["kalman_rr_estimate"] = kalman_estimate
df["kalman_uncertainty"] = kalman_uncertainty


# ------------------------------------------------------------
# Quantify recovery
# ------------------------------------------------------------
available = ~np.isnan(observed_rr)

sensor_rmse_available = np.sqrt(
    np.mean((observed_rr[available] - true_rr[available]) ** 2)
)

kalman_rmse_all = np.sqrt(
    np.mean((kalman_estimate - true_rr) ** 2)
)

kalman_rmse_available = np.sqrt(
    np.mean((kalman_estimate[available] - true_rr[available]) ** 2)
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


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 4.8))

# Shade dropout regions
in_dropout = False
start = None

for i, is_missing in enumerate(df["rr_dropout"]):
    if is_missing and not in_dropout:
        start = df.loc[i, "time_s"]
        in_dropout = True

    if in_dropout and (not is_missing or i == len(df) - 1):
        end = df.loc[i, "time_s"]
        plt.axvspan(start, end, alpha=0.15)
        in_dropout = False

# Noisy sensor observations
plt.scatter(
    df["time_s"],
    df["rr_sensor"],
    s=12,
    alpha=0.35,
    label="Noisy sensor observation",
)

# True latent state, known only because this is simulated
plt.plot(
    df["time_s"],
    df["latent_rr"],
    linestyle="--",
    linewidth=2,
    label="True latent RR (simulated)",
)

# Kalman estimate
plt.plot(
    df["time_s"],
    df["kalman_rr_estimate"],
    linewidth=2.5,
    label="State-space / Kalman estimate",
)

plt.xlabel("Time (s)")
plt.ylabel("Respiratory rate (breaths/min)")
plt.title("State-space tracking of latent respiratory rate")
plt.legend(frameon=False, fontsize=8)
plt.tight_layout()

png_path = FIGURES_DIR / "figure_state_space_kalman_rr.png"
pdf_path = FIGURES_DIR / "figure_state_space_kalman_rr.pdf"

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
print(f"Saved summary to: {summary_path}")