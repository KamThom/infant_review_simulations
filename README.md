# Infant Review Simulations

This repository contains synthetic data and analysis scripts used to generate illustrative figures and tables for a scientific review paper about infant sensing, modeling, and interpretation.

The project is designed to demonstrate analysis workflows, not to provide evidence about real infant biology, physiology, behavior, or development. All datasets and model outputs in this repository are generated or derived from simulation code.

## Repository Structure

- `scripts/`: numbered scripts for data generation, data checks, statistical examples, machine-learning examples, and figure generation.
- `base_data/`: generated synthetic source datasets used by the analysis scripts.
- `tables/`: ignored generated CSV summaries, model outputs, and figure-support tables.
- `figures/`: generated PNG and PDF figures.
- `requirements.txt`: minimal Python dependency list.

## Setup

Create and activate a local virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generated Data

The main synthetic datasets are:

- `base_data/session_features_df.csv`: session-level infant summaries across simulated infants, ages, and repeated sessions.
- `base_data/dense_timeseries_df.csv`: second-level synthetic time series for a single dense recording.
- `base_data/position_windows_df.csv`: 10-second synthetic wearable-sensor windows for an infant-position labeling example.

The simulation parameters are saved in:

- `base_data/simulation_parameters.json`
- `base_data/simulation_parameters.csv`

## Running Individual Analyses

This project is intentionally organized so figures can be regenerated one by one. There is no run-all script.

Generate the main synthetic session and dense time-series data:

```bash
python scripts/00_generate_simulated_data.py
```

Generate the synthetic position-window dataset:

```bash
python scripts/00b_generate_position_windows.py
```

Check the generated session data and create the age-level summary table:

```bash
python scripts/01_check_session_data.py
```

Then run whichever figure script you need:

```bash
python scripts/supplemental_02_residual_confounder_analysis.py
python scripts/03_mixed_effects_hr_trajectory.py
python scripts/03b_mixed_effects_hr_longitudinal_individual.py
python scripts/04_automated_position_labeling.py
python scripts/05_state_space_kalman_rr.py
python scripts/supplemental_06_bland_altman.py
python scripts/supplemental_07_leakage_failure.py
python scripts/supplemental_08_empirical_hr_centiles.py
```

Most analysis scripts overwrite their corresponding files in `tables/` and `figures/`. The contents of `tables/` are generated artifacts and are ignored by git.

## Model Details

- `supplemental_02_residual_confounder_analysis.py` uses ordinary least squares regression to compare an omitted-context model against a context-adjusted model. The purpose is to show how residual patterns can reveal missing contextual structure.
- `03_mixed_effects_hr_trajectory.py` uses a linear mixed-effects model with infant-level random intercepts and random age slopes. Fixed effects include age, group, and the age-by-group interaction, so the figure can show both individual repeated-measure trajectories and population-level reference versus comparison trends.
- `03b_mixed_effects_hr_longitudinal_individual.py` uses a simpler linear mixed-effects model with age as the fixed effect and infant-level random intercepts and random age slopes. This companion removes the group comparison layer to focus on longitudinal change within infants.
- `04_automated_position_labeling.py` uses a random forest classifier with median imputation. It predicts simulated infant position from wearable-window features using a held-out window split from one synthetic 3-hour session, so it is an example of an automated-labeling workflow rather than a validated deployment model.
- `05_state_space_kalman_rr.py` uses a one-dimensional Kalman-style state-space filter. It smooths noisy respiratory-rate observations, carries estimates through missing sensor samples, and compares the estimate with the simulated latent signal for diagnostic purposes.
- `supplemental_06_bland_altman.py` is not a predictive model. It uses a Bland-Altman agreement analysis to compare two simulated heart-rate sensor measurements by plotting their mean against their difference.
- `supplemental_07_leakage_failure.py` uses logistic regression pipelines evaluated with grouped cross-validation by infant. It contrasts a deployable feature set with a leaky feature set to show how recording-context leakage can inflate performance.
- `supplemental_08_empirical_hr_centiles.py` is not model-based. It averages repeated sessions within infant and age, then computes empirical heart-rate centiles from the simulated cohort.

## Figure Guide

### Figure 02: Residual Confounder Analysis

`scripts/supplemental_02_residual_confounder_analysis.py` demonstrates how residual diagnostics can reveal structure left behind when a simulated contextual variable is omitted from a regression model. The example compares an omitted-context model with a context-adjusted model.

Outputs:

- `figures/supplemental_02_residual_confounder_analysis.png`
- `figures/supplemental_02_residual_confounder_analysis.pdf`
- `tables/residual_confounder_model_coefficients.csv`
- `tables/residual_confounder_binned_summary.csv`

### Figure 03: Mixed-Effects Heart-Rate Trajectory

`scripts/03_mixed_effects_hr_trajectory.py` illustrates a mixed-effects model for repeated infant session summaries across age. The figure shows selected infant-level trajectories in gray, group-age means, and model-estimated population trajectories for the simulated reference and comparison groups.

Outputs:

- `figures/figure_03_mixed_effects_hr_trajectory.png`
- `figures/figure_03_mixed_effects_hr_trajectory.pdf`
- `tables/mixed_effects_hr_model_coefficients.csv`
- `tables/group_age_resting_hr_summary.csv`
- `tables/mixed_effects_hr_population_predictions.csv`
- `tables/figure_03_example_infant_trajectories.csv`

### Figure 03b: Longitudinal Individual Trajectory Companion

`scripts/03b_mixed_effects_hr_longitudinal_individual.py` is a companion version of Figure 03 that removes the group comparison layer. It emphasizes longitudinal change by showing individual infant trajectories, infant-age means, and one population trajectory.

Outputs:

- `figures/figure_03b_mixed_effects_hr_longitudinal_individual.png`
- `figures/figure_03b_mixed_effects_hr_longitudinal_individual.pdf`
- `tables/figure_03b_mixed_effects_hr_model_coefficients.csv`
- `tables/figure_03b_population_age_resting_hr_summary.csv`
- `tables/figure_03b_mixed_effects_hr_population_predictions.csv`
- `tables/figure_03b_example_infant_trajectories.csv`

### Figure 04: Automated Position Labeling

`scripts/04_automated_position_labeling.py` demonstrates how simulated wearable-window features from a 3-hour session can be mapped to infant-position labels. The current example uses a random held-out window split within one synthetic infant/session, so the performance estimate should be interpreted as an illustration of the workflow rather than deployable generalization.

Outputs:

- `figures/figure_04_automated_position_labeling.png`
- `figures/figure_04_automated_position_labeling.pdf`
- `tables/position_windows_with_predictions.csv`
- `tables/position_labeling_summary.csv`
- `tables/position_labeling_confusion_matrix.csv`
- `tables/position_labeling_confusion_matrix_normalized.csv`

### Figure 05: State-Space Respiratory-Rate Estimate

`scripts/05_state_space_kalman_rr.py` demonstrates a simple state-space/Kalman-style estimate for respiratory rate from noisy and intermittently missing synthetic observations. The simulated latent respiratory-rate signal is used for diagnostic evaluation, not as a real-world target.

Outputs:

- `figures/figure_05_state_space_kalman_rr.png`
- `figures/figure_05_state_space_kalman_rr.pdf`
- `figures/figure_05_state_space_kalman_rr_with_ground_truth.png`
- `figures/figure_05_state_space_kalman_rr_with_ground_truth.pdf`
- `tables/kalman_rr_summary.csv`

### Figure 06: Bland-Altman Sensor Agreement

`scripts/supplemental_06_bland_altman.py` illustrates a Bland-Altman comparison between two simulated heart-rate sensors. This example is about sensor agreement, not predictor-outcome association.

Outputs:

- `figures/supplemental_06_bland_altman_sensor_agreement.png`
- `figures/supplemental_06_bland_altman_sensor_agreement.pdf`
- `tables/bland_altman_summary.csv`

### Figure 07: Leakage Failure

`scripts/supplemental_07_leakage_failure.py` demonstrates how a leaked recording-context feature can inflate apparent predictive performance. The grouped cross-validation split keeps rows from the same simulated infant together.

Outputs:

- `figures/supplemental_07_leakage_failure.png`
- `figures/supplemental_07_leakage_failure.pdf`
- `tables/figure_07_leakage_failure_fold_scores.csv`
- `tables/figure_07_leakage_failure_summary.csv`

### Figure 08: Empirical Heart-Rate Centiles

`scripts/supplemental_08_empirical_hr_centiles.py` creates an empirical centile view from the simulated cohort. Repeated sessions are first averaged within infant and age before centiles are computed. This figure is not an external pediatric reference curve.

Outputs:

- `figures/supplemental_08_empirical_hr_centiles.png`
- `figures/supplemental_08_empirical_hr_centiles.pdf`
- `tables/figure_08_infant_age_resting_hr.csv`
- `tables/figure_08_empirical_hr_centiles.csv`
- `tables/figure_08_group_median_resting_hr.csv`

## Interpretation Rules

- Treat every result as a synthetic demonstration.
- Do not cite generated estimates as empirical infant findings.
- Preserve participant-level structure when evaluating models.
- Prefer infant- or group-aware splits when evaluating predictive models.
- Be explicit when an analysis is illustrative rather than a valid generalization test.

## Reproducibility Notes

The simulation scripts use fixed random seeds. The generated outputs are therefore intended to be reproducible when run with compatible package versions.
