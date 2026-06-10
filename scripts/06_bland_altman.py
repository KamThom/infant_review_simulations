from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_PATH = DATA_DIR / "session_features_df.csv"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["sensor_mean_hr"] = (df["ecg_hr"] + df["ppg_hr"]) / 2
df["sensor_difference_hr"] = df["ppg_hr"] - df["ecg_hr"]

bias = df["sensor_difference_hr"].mean()
sd_diff = df["sensor_difference_hr"].std()

loa_upper = bias + 1.96 * sd_diff
loa_lower = bias - 1.96 * sd_diff

print("\nBland-Altman sensor agreement")
print("-----------------------------")
print(f"Mean bias, PPG - ECG: {bias:.2f} bpm")
print(f"Upper limit of agreement: {loa_upper:.2f} bpm")
print(f"Lower limit of agreement: {loa_lower:.2f} bpm")

summary = pd.DataFrame(
    {
        "metric": [
            "mean_bias_ppg_minus_ecg_bpm",
            "sd_difference_bpm",
            "upper_limit_of_agreement_bpm",
            "lower_limit_of_agreement_bpm",
        ],
        "value": [
            bias,
            sd_diff,
            loa_upper,
            loa_lower,
        ],
    }
)

summary.to_csv(TABLES_DIR / "bland_altman_summary.csv", index=False)

plt.figure(figsize=(6, 4.5))

plt.scatter(
    df["sensor_mean_hr"],
    df["sensor_difference_hr"],
    alpha=0.35,
    s=18,
)

plt.axhline(bias, linestyle="-", linewidth=1.5, label=f"Mean bias = {bias:.2f} bpm")
plt.axhline(loa_upper, linestyle="--", linewidth=1, label=f"+1.96 SD = {loa_upper:.2f}")
plt.axhline(loa_lower, linestyle="--", linewidth=1, label=f"-1.96 SD = {loa_lower:.2f}")
plt.axhline(0, linestyle=":", linewidth=1)

plt.xlabel("Mean heart rate from ECG and PPG sensors (bpm)")
plt.ylabel("PPG - ECG heart rate (bpm)")
plt.title("Bland-Altman agreement between simulated HR sensors")
plt.legend(frameon=False, fontsize=8)

plt.tight_layout()

png_path = FIGURES_DIR / "figure_06_bland_altman_sensor_agreement.png"
pdf_path = FIGURES_DIR / "figure_06_bland_altman_sensor_agreement.pdf"

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)

print(f"\nSaved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")
