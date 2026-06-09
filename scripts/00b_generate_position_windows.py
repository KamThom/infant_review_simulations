from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


PARAMS = {
    "seed": 104,
    "infant_id": "I001",
    "session_id": "position_session_001",
    "duration_min": 30,
    "window_size_s": 10,
}


STATE_PARAMS = {
    "held": {
        "movement_intensity_mean": 0.72,
        "movement_intensity_sd": 0.12,
        "posture_angle_mean": 65.0,
        "posture_angle_sd": 14.0,
        "movement_variability_mean": 0.34,
        "movement_variability_sd": 0.08,
        "vertical_accel_mean": 0.45,
        "vertical_accel_sd": 0.12,
        "horizontal_accel_sd_mean": 0.28,
        "horizontal_accel_sd_sd": 0.07,
    },
    "supine": {
        "movement_intensity_mean": 0.18,
        "movement_intensity_sd": 0.08,
        "posture_angle_mean": 5.0,
        "posture_angle_sd": 10.0,
        "movement_variability_mean": 0.09,
        "movement_variability_sd": 0.04,
        "vertical_accel_mean": 0.05,
        "vertical_accel_sd": 0.08,
        "horizontal_accel_sd_mean": 0.10,
        "horizontal_accel_sd_sd": 0.04,
    },
    "prone": {
        "movement_intensity_mean": 0.28,
        "movement_intensity_sd": 0.09,
        "posture_angle_mean": -8.0,
        "posture_angle_sd": 11.0,
        "movement_variability_mean": 0.14,
        "movement_variability_sd": 0.05,
        "vertical_accel_mean": -0.08,
        "vertical_accel_sd": 0.08,
        "horizontal_accel_sd_mean": 0.16,
        "horizontal_accel_sd_sd": 0.05,
    },
    "sitting": {
        "movement_intensity_mean": 0.42,
        "movement_intensity_sd": 0.10,
        "posture_angle_mean": 85.0,
        "posture_angle_sd": 12.0,
        "movement_variability_mean": 0.20,
        "movement_variability_sd": 0.06,
        "vertical_accel_mean": 0.82,
        "vertical_accel_sd": 0.10,
        "horizontal_accel_sd_mean": 0.18,
        "horizontal_accel_sd_sd": 0.05,
    },
}


def sample_next_state(rng, current_state):
    states = ["held", "supine", "prone", "sitting"]

    transition_probs = {
        "held":    [0.78, 0.10, 0.04, 0.08],
        "supine":  [0.10, 0.72, 0.12, 0.06],
        "prone":   [0.08, 0.16, 0.68, 0.08],
        "sitting": [0.14, 0.06, 0.06, 0.74],
    }

    return rng.choice(states, p=transition_probs[current_state])


def generate_position_windows(params):
    rng = np.random.default_rng(params["seed"])

    duration_s = params["duration_min"] * 60
    window_size_s = params["window_size_s"]
    n_windows = duration_s // window_size_s

    rows = []

    state = rng.choice(["held", "supine", "prone", "sitting"], p=[0.30, 0.35, 0.20, 0.15])

    for window_idx in range(n_windows):
        if window_idx > 0:
            state = sample_next_state(rng, state)

        p = STATE_PARAMS[state]

        movement_intensity = rng.normal(
            p["movement_intensity_mean"],
            p["movement_intensity_sd"],
        )

        posture_angle = rng.normal(
            p["posture_angle_mean"],
            p["posture_angle_sd"],
        )

        movement_variability = rng.normal(
            p["movement_variability_mean"],
            p["movement_variability_sd"],
        )

        vertical_accel_mean = rng.normal(
            p["vertical_accel_mean"],
            p["vertical_accel_sd"],
        )

        horizontal_accel_sd = rng.normal(
            p["horizontal_accel_sd_mean"],
            p["horizontal_accel_sd_sd"],
        )


        movement_intensity = np.clip(movement_intensity, 0.0, 1.3)
        movement_variability = np.clip(movement_variability, 0.0, 0.8)
        horizontal_accel_sd = np.clip(horizontal_accel_sd, 0.0, 0.8)
        posture_angle = np.clip(posture_angle, -40.0, 115.0)
        vertical_accel_mean = np.clip(vertical_accel_mean, -0.35, 1.15)

        start_s = window_idx * window_size_s
        end_s = start_s + window_size_s

        rows.append(
            {
                "window_id": f"W{window_idx + 1:04d}",
                "infant_id": params["infant_id"],
                "session_id": params["session_id"],
                "time_start_s": start_s,
                "time_end_s": end_s,
                "true_position": state,
                "movement_intensity": round(movement_intensity, 3),
                "posture_angle": round(posture_angle, 2),
                "movement_variability": round(movement_variability, 3),
                "vertical_accel_mean": round(vertical_accel_mean, 3),
                "horizontal_accel_sd": round(horizontal_accel_sd, 3),
            }
        )

    return pd.DataFrame(rows)


def main():
    df = generate_position_windows(PARAMS)

    data_path = DATA_DIR / "position_windows_df.csv"
    df.to_csv(data_path, index=False)

    summary = (
        df.groupby("true_position")
        .agg(
            n_windows=("window_id", "size"),
            mean_movement_intensity=("movement_intensity", "mean"),
            mean_posture_angle=("posture_angle", "mean"),
            mean_movement_variability=("movement_variability", "mean"),
        )
        .round(3)
        .reset_index()
    )

    summary_path = TABLES_DIR / "position_window_generation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nGenerated position-window dataset")
    print("---------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {data_path}")
    print("\nPosition summary:")
    print(summary)


if __name__ == "__main__":
    main()