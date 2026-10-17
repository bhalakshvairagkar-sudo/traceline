# Traceline Workstream C — VALIDATION

## Purpose and evidence status

This report defines the evaluation method. It does not assert positioning-model performance. The included fixture is synthetic smoke-test data with controlled prediction biases, not an official development set and not hidden-set evidence.

## Data inputs

Use the shared CSV schemas defined in the Traceline Prototype Execution Guide. Record the dataset manifest, source commit, generator version, base/run seeds, signal-model configuration and split used for every run. Report whether every result is based on synthetic simulation or real hardware; this prototype uses simulated measurements unless evidence explicitly says otherwise.

## Splits and leakage prevention

- Training residuals may be used to fit error-map corrections only.
- Validate the error map on distinct development run IDs.
- Split whole trajectories, never adjacent rows/time steps from a trajectory.
- Hold the clinic venue out of training and development as specified in the guide.
- Select NLOS and emergency uncertainty thresholds on development data; freeze them before the final hidden evaluation.
- Coordinate the final secret hidden evaluation with Part A only after the model/settings freeze. Run it once; do not tune afterward and do not commit hidden truth/labels.

## Required metrics

For each method, report step-level Euclidean error in metres (mean, median, RMSE, P90, P95), track-level metrics, prediction coverage and missing-prediction count. Where data is available, report NLOS precision/recall/F1/confusion matrix, uncertainty calibration diagnostics, forecast error by horizon, emergency-flag precision/recall/F1, and robustness slices by venue, anchor count, noise, NLOS, packet loss, crowd, drift, outliers and dropout duration.

## Baseline comparison

B1 weighted least-squares trilateration, B2 residual-based rejection/down-weighting, B3 extended Kalman filter and Traceline must receive identical readings and active-anchor configurations. Baseline settings must be tuned only on train/dev.

## Error-map validation

The error map uses predicted-coordinate cells because truth is unavailable at inference. Report support counts as the number of independent runs per cell. Corrections below the support threshold are disabled. Report before/after results on held-out dev trajectories, including unsupported cells. Only claim improvement when the measured development results support it. Out-of-fold training predictions are preferred; disclose if model predictions were in-sample.

## Stress-test validation

Generate at least 5,000 unique seeded condition configurations covering anchor geometry/count, noise, NLOS rate/bias, crowd, packet loss/outages, drift and outliers. Predeclare the P90 failure criterion before scoring. The generated matrix alone is not a stress-test result: connect it to Part A's simulator and frozen Part B predictors, record all successful/failed condition runs, and produce failure-boundary summaries/heatmaps from measured outputs.

## Uncertainty and emergency reporting

Do not claim probability calibration from `sigma_m` by name alone. Empirically compare error versus sigma and nominal region coverage only after defining and testing the uncertainty semantics. Emergency flags are a prioritization visualization, not a certified life-safety control.

## Results table to complete from actual run

| Split / seed | Method | Median error (m) | RMSE (m) | P90 (m) | P95 (m) | Prediction coverage | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Pending actual run | B1 | — | — | — | — | — | — |
| Pending actual run | B2 | — | — | — | — | — | — |
| Pending actual run | B3 | — | — | — | — | — | — |
| Pending actual run | Traceline | — | — | — | — | — | — |

Do not fill this table from the included smoke-test fixture.
