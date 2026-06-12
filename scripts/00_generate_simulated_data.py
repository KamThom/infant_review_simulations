from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "base_data"
TABLES_DIR = PROJECT_DIR / "tables"

DATA_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


PARAMS = {
    "seed": 42,
    "n_infants": 120,
    "ages_months": [3, 6, 9, 12, 18, 24],
    "sessions_per_age": 3,
    "session_keep_prob": 0.90,

    "ref_hr_at_3mo": 137.0,
    "comparison_hr_at_3mo": 143.0,
    "ref_hr_slope_per_month_after_3mo": -1.10,
    "comparison_hr_slope_per_month_after_3mo": -0.82,
    "hr_between_infant_sd": 5.5,
    "ref_hr_infant_slope_sd_per_month_after_3mo": 0.08,
    "comparison_hr_infant_slope_sd_per_month_after_3mo": 0.20,
    "hr_session_noise_sd": 3.0,

    "hrv_mean_sdnn_ms": 24.0,
    "hrv_age_slope_per_month_after_3mo": 0.01,
    "hrv_between_infant_sd": 2.5,
    "hrv_session_noise_sd": 1.6,

    "rr_at_3mo": 43.0,
    "rr_slope_per_month_after_3mo": -0.78,
    "rr_between_infant_sd": 2.2,
    "rr_session_noise_sd": 1.4,

    "social_exposure_beta": 0.22,
    "luminance_beta": -0.10,
    "context_bump_beta": 0.18,
    "pupil_between_infant_sd": 0.04,
    "pupil_session_noise_sd": 0.04,

    "ppg_bias_bpm": -1.6,
    "ecg_noise_sd": 1.0,
    "ppg_noise_sd": 2.4,

    "dense_duration_s": 300,
    "dense_rr_baseline": 38.0,
    "dense_arousal_to_rr": 2.5,
    "dense_rr_sensor_noise_sd": 2.0,
}


def generate_session_features_df(params: dict) -> pd.DataFrame:
    rng = np.random.default_rng(params["seed"])

    n_infants = params["n_infants"]
    ages = params["ages_months"]
    sessions_per_age = params["sessions_per_age"]

    groups = rng.choice(
        ["REF", "COMPARISON"],
        size=n_infants,
        p=[0.50, 0.50],
    )

    rows = []

    for infant_index in range(n_infants):
        infant_id = f"I{infant_index + 1:03d}"
        group = groups[infant_index]

        infant_hr_offset = rng.normal(0, params["hr_between_infant_sd"])
        hr_slope_sd_key = (
            "comparison_hr_infant_slope_sd_per_month_after_3mo"
            if group == "COMPARISON"
            else "ref_hr_infant_slope_sd_per_month_after_3mo"
        )
        infant_hr_slope_offset = rng.normal(
            0,
            params[hr_slope_sd_key],
        )
        infant_rr_offset = rng.normal(0, params["rr_between_infant_sd"])
        infant_hrv_offset = rng.normal(0, params["hrv_between_infant_sd"])
        infant_pupil_offset = rng.normal(0, params["pupil_between_infant_sd"])

        for age in ages:
            age_from_3mo = age - 3

            for session in range(1, sessions_per_age + 1):
            
                if rng.random() > params["session_keep_prob"]:
                    continue

                if group == "REF":
                    hr_mean = (
                        params["ref_hr_at_3mo"]
                        + params["ref_hr_slope_per_month_after_3mo"] * age_from_3mo
                    )
                else:
                    hr_mean = (
                        params["comparison_hr_at_3mo"]
                        + params["comparison_hr_slope_per_month_after_3mo"] * age_from_3mo
                    )

                resting_hr = (
                    hr_mean
                    + infant_hr_offset
                    + infant_hr_slope_offset * age_from_3mo
                    + rng.normal(0, params["hr_session_noise_sd"])
                )

                hrv_sdnn = (
                    params["hrv_mean_sdnn_ms"]
                    + params["hrv_age_slope_per_month_after_3mo"] * age_from_3mo
                    + infant_hrv_offset
                    + rng.normal(0, params["hrv_session_noise_sd"])
                )
                hrv_sdnn = max(hrv_sdnn, 8.0)

                mean_rr = (
                    params["rr_at_3mo"]
                    + params["rr_slope_per_month_after_3mo"] * age_from_3mo
                    + infant_rr_offset
                    + rng.normal(0, params["rr_session_noise_sd"])
                )

                
                mean_luminance = np.clip(rng.normal(0.51, 0.04), 0.35, 0.70)

            
                context_activity_z = rng.normal(0, 1)
                context_bump = np.exp(
                    -0.5 * ((context_activity_z - 1.1) / 0.35) ** 2
                )

                social_exposure = np.clip(
                    0.45
                    + 0.20 * context_bump
                    + (0.04 if group == "COMPARISON" else 0.0)
                    + rng.normal(0, 0.10),
                    0,
                    1,
                )

                pupil_response = (
                    0.20
                    + params["social_exposure_beta"] * social_exposure
                    + params["luminance_beta"] * mean_luminance
                    + params["context_bump_beta"] * context_bump
                    + infant_pupil_offset
                    + rng.normal(0, params["pupil_session_noise_sd"])
                )

                ecg_hr = resting_hr + rng.normal(0, params["ecg_noise_sd"])
                ppg_hr = (
                    resting_hr
                    + params["ppg_bias_bpm"]
                    + rng.normal(0, params["ppg_noise_sd"])
                )

                missing_fraction = np.clip(rng.beta(1.5, 18.0), 0, 0.40)

                group_code = 1 if group == "COMPARISON" else 0
                recording_context_code = group_code + rng.normal(0, 0.12)

                rows.append(
                    {
                        "infant_id": infant_id,
                        "group": group,
                        "age_months": age,
                        "age_from_3mo": age_from_3mo,
                        "session": session,

                        "resting_hr": round(resting_hr, 1),
                        "hrv_sdnn": round(hrv_sdnn, 1),
                        "mean_rr": round(mean_rr, 1),

                        "mean_luminance": round(mean_luminance, 3),
                        "context_activity_z": round(context_activity_z, 3),
                        "context_bump": round(context_bump, 3),
                        "social_exposure": round(social_exposure, 3),
                        "pupil_response": round(pupil_response, 3),

                        "ecg_hr": round(ecg_hr, 1),
                        "ppg_hr": round(ppg_hr, 1),

                        "missing_fraction": round(missing_fraction, 3),

                        "recording_context_code": round(recording_context_code, 3),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["infant_id", "age_months", "session"]).reset_index(drop=True)
    return df


def generate_dense_timeseries_df(params: dict) -> pd.DataFrame:
    rng = np.random.default_rng(params["seed"] + 1000)

    duration_s = params["dense_duration_s"]
    time_s = np.arange(duration_s)

    latent_arousal = np.zeros(duration_s)
    latent_arousal[0] = rng.normal(0, 0.15)

    for t in range(1, duration_s):
        latent_arousal[t] = 0.975 * latent_arousal[t - 1] + rng.normal(0, 0.06)

    latent_rr = (
        params["dense_rr_baseline"]
        + params["dense_arousal_to_rr"] * latent_arousal
        + rng.normal(0, 0.18, duration_s)
    )

    rr_sensor = (
        latent_rr
        + rng.normal(0, params["dense_rr_sensor_noise_sd"], duration_s)
    )

    dropout = np.zeros(duration_s, dtype=bool)

    for start in [85, 160, 235]:
        dropout[start:start + 10] = True

    rr_sensor[dropout] = np.nan

    hr_sensor = 125.0 + 9.0 * latent_arousal + rng.normal(0, 2.3, duration_s)
    pupil_sensor = 4.1 + 0.22 * latent_arousal + rng.normal(0, 0.04, duration_s)
    luminance = np.clip(0.52 + rng.normal(0, 0.03, duration_s), 0.35, 0.70)

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "infant_id": "I001",
            "session_id": "dense_001",

            "latent_arousal": np.round(latent_arousal, 3),
            "latent_rr": np.round(latent_rr, 2),
            "rr_sensor": np.round(rr_sensor, 2),
            "rr_dropout": dropout,

            "hr_sensor": np.round(hr_sensor, 2),
            "pupil_sensor": np.round(pupil_sensor, 3),
            "luminance": np.round(luminance, 3),
        }
    )

    return df


def save_parameters(params: dict) -> None:
    json_path = DATA_DIR / "simulation_parameters.json"
    csv_path = DATA_DIR / "simulation_parameters.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    param_rows = [{"parameter": key, "value": value} for key, value in params.items()]
    pd.DataFrame(param_rows).to_csv(csv_path, index=False)


def main() -> None:
    session_df = generate_session_features_df(PARAMS)
    dense_df = generate_dense_timeseries_df(PARAMS)

    session_path = DATA_DIR / "session_features_df.csv"
    dense_path = DATA_DIR / "dense_timeseries_df.csv"

    session_df.to_csv(session_path, index=False)
    dense_df.to_csv(dense_path, index=False)

    save_parameters(PARAMS)

    print("\nGenerated simulated datasets")
    print("----------------------------")
    print(f"session_features_df: {session_df.shape} -> {session_path}")
    print(f"dense_timeseries_df: {dense_df.shape} -> {dense_path}")
    print(f"parameters saved to: {DATA_DIR / 'simulation_parameters.json'}")
    print(f"parameters saved to: {DATA_DIR / 'simulation_parameters.csv'}")

    print("\nAge-level means")
    print("---------------")
    age_summary = (
        session_df.groupby("age_months")
        .agg(
            n_rows=("infant_id", "size"),
            n_infants=("infant_id", "nunique"),
            resting_hr_mean=("resting_hr", "mean"),
            hrv_sdnn_mean=("hrv_sdnn", "mean"),
            mean_rr_mean=("mean_rr", "mean"),
        )
        .round(2)
    )
    print(age_summary)


if __name__ == "__main__":
    main()
