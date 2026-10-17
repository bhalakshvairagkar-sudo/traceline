# Traceline Workstream C — Evaluation, Error Learning and Stress Testing

This implementation owns **`eval/` only** and follows the Workstream C section of the Traceline Prototype Execution Guide. It implements position/error metrics, fair baseline comparison, uncertainty diagnostics, optional NLOS and forecast sidecars, a trajectory-separated self-learning error map, crowd-zone analysis, emergency-uncertainty flags, and a reproducible stress-condition generator.

> **Important:** `fixtures/` is a tiny deterministic synthetic smoke-test dataset, not benchmark data. The included fixture intentionally has controlled method biases so that the code paths can be tested. Do not present its metrics as real positioning performance or as evidence of a real Traceline model.

## 1. Environment

The shared repository's frozen `shared/versions.json` is authoritative. Before merging this standalone folder, align the versions below with that file and the repository's existing dependency lock. Do not silently change shared versions.

```bash
cd eval
python -m pip install -r requirements.txt
make test
```

The evaluator uses Python, NumPy, pandas and Matplotlib. No cloud service, API key, server, or proprietary hardware is required for this evaluation code itself. Run it against the team's local generated CSV files.

## 2. Inputs and data contracts

The main data directory must contain the unchanged shared-contract files:

- `truth.csv`: `run_id,t,x_m,y_m`.
- `measurements.csv`: `run_id,t,anchor_id,range_m,rssi_dbm`.
- Optional `labels_nlos.csv`: `run_id,t,anchor_id,is_nlos,bias_m`.
- Optional `crowd.csv`: `run_id,t,zone_id,density_people_m2`.
- Optional evaluator-only `run_metadata.csv`, one row per `run_id`, for condition tags not represented in the shared CSV schema (for example `noise_level`, `noise_multiplier`, `drift_condition`, `outlier_condition`, `dropout_duration_steps`, `scenario_id`). It is a sidecar; it does not change a shared contract.

The predictions directory must contain one canonical prediction file per method:

```text
predictions/
  B1.csv
  B2.csv
  B3.csv
  Traceline.csv
  nlos_probabilities/       # optional, if per-anchor probabilities are available
    B1.csv
    B2.csv
    B3.csv
    Traceline.csv
  predicted_positions/      # optional, if forecast output is available
    Traceline.csv
```

Each method prediction file uses the canonical columns `run_id,t,x_hat_m,y_hat_m,sigma_m,flags`. `flags` is a JSON array encoded as a CSV string. Position scoring left-joins predictions against every truth step, so missing predictions decrease coverage and are not silently discarded. Error metrics over available predictions are always reported alongside coverage.

The optional NLOS-probability sidecar is evaluator-specific and must contain `run_id,t,anchor_id,nlos_probability`. The optional forecast sidecar is evaluator-specific and must contain `run_id,t,horizon_steps,x_hat_m,y_hat_m`. These sidecars avoid changing the shared `predictions.csv` contract. `horizon_steps` is a positive integer and the frozen time step is 0.5 seconds.

Run IDs are parsed according to the contract form, for example `hospital-dev-0012`. When using non-contract IDs, add explicit handling rather than guessing the split.

## 3. Main evaluation

```bash
python run_eval.py \
  --data-dir ../datasets/public \
  --predictions-dir ../model_outputs \
  --output-dir outputs/evaluation \
  --split dev
```

The runner scores B1 weighted least-squares trilateration, B2 residual-based rejection/down-weighting, B3 the extended Kalman filter and Traceline on the same truth steps. Pass `--methods B1 B2 B3 Traceline` to select files. Every method is left-joined to the same ground-truth rows; coverage and missing prediction counts expose incomplete outputs.

Created reports include:

- `reports/metrics_summary.csv` and `.json`: mean, median, RMSE, P90/P95, track summaries and coverage.
- `reports/track_metrics.csv`: per-run/trajectory metrics so long tracks do not dominate silently.
- `reports/step_errors.csv`: scored steps, estimates, errors and evaluation strata.
- `reports/robustness_slices.csv`: available breakdowns by venue, active-anchor count, NLOS, observed missingness, crowd, and metadata tags.
- `reports/calibration_bins_<method>.csv`: error-versus-sigma-bin diagnostics.
- `reports/forecast_errors_<method>.csv`: horizon-specific forecast errors when a forecast sidecar exists.
- `plots/`: median error, CDF and P90-by-anchor-count plots where data permits.

If an NLOS probability sidecar and NLOS labels exist, the runner selects a threshold on development data by maximum F1 unless `--nlos-threshold` is supplied; the selected threshold is recorded. For a final hidden evaluation, pass the already-frozen threshold and explicitly acknowledge the one-time hidden-set run:

```bash
python run_eval.py --data-dir ../datasets/final_hidden \
  --predictions-dir ../model_outputs/frozen \
  --output-dir outputs/hidden_eval --split hidden \
  --nlos-threshold 0.50 --confirm-hidden
```

Replace `0.50` with the threshold actually selected on development data; do not copy the example value blindly. Hidden truth/labels must not be exposed to Part B or used to tune anything. Run hidden evaluation only once after the model/settings freeze and coordination with Part A. Avoid `--split all` when hidden data is present unless this same restriction is intentionally satisfied.

### Uncertainty semantics

The code reports error coverage at several multiples of `sigma_m`, error-versus-sigma bins, and rank correlation. It does **not** claim a nominal probability coverage interpretation merely because a column is named `sigma_m`. Confirm and document the semantics using calibration experiments first, as required by the shared contract.

## 4. Self-learning error map

```bash
python error_map.py \
  --truth ../datasets/public/truth.csv \
  --predictions ../model_outputs/Traceline.csv \
  --output-dir outputs/error_map \
  --cell-size-m 1.0 --min-support-runs 3 \
  --train-prediction-provenance out_of_fold
```

The command fits residual corrections on `train` runs only and evaluates the corrected positions on separate `dev` runs. Cells are assigned from the **estimated position**, since the true location is unknown at inference time. Corrections are computed from per-run cell averages; `support_count` is the number of distinct training trajectories, not raw time samples. Cells below `min-support-runs` are not corrected.

Outputs: `error_map_cells.csv`, `error_map_dev_validation.csv`, `error_map_validation.json`, `error_map_vectors.png`, and `error_map_before_after.png`.

**Prefer out-of-fold/held-out training predictions** when fitting residual corrections. In-sample model residuals can be optimistic if Part B overfits. `--train-prediction-provenance` records the source of training predictions and warns if they are not identified as out-of-fold. The code rejects overlapping train/dev run IDs; never randomly split adjacent time steps from one trajectory across train and dev.

## 5. Crowd degradation and crowd-input ablation

```bash
python crowd_analysis.py \
  --data-dir ../datasets/public \
  --predictions-dir ../model_outputs \
  --venues-dir ../shared/venues \
  --output-dir outputs/crowd \
  --methods Traceline --split dev
```

The script tests ground-truth positions against the venue JSON's `zone_type: crowd` polygons, rather than assuming that a crowd-density row means the target itself is inside a crowd zone.

For a direct model ablation, Part B should provide two prediction files (one with crowd input and one without). Example:

```bash
python crowd_analysis.py ... \
  --methods Traceline Traceline_no_crowd \
  --with-crowd-method Traceline \
  --without-crowd-method Traceline_no_crowd
```

Outputs include `crowd_zone_metrics.csv`, `crowd_analysis.json`, a P90 chart and, when two ablation methods are provided, paired step differences. Positive `crowd_input_benefit_m` means the with-crowd model had lower error on that step. Do not claim the crowd input helps unless this measured comparison supports it.

## 6. Emergency uncertainty flags

```bash
python emergency_flags.py \
  --truth ../datasets/public/truth.csv \
  --predictions ../model_outputs/Traceline.csv \
  --output-dir outputs/emergency \
  --high-error-threshold-m <predeclared-value-in-metres> \
  --split dev --seed 2026
```

`high-error-threshold-m` must be chosen and recorded before evaluating the flags; it is not a universal safety threshold. Development run IDs are divided into disjoint calibration and evaluation groups. The uncertainty threshold is selected by maximum F1 on calibration runs and evaluated on the held-out development runs. The script saves `frozen_emergency_threshold.json` for reuse after freeze. For a post-freeze hidden evaluation, reuse this threshold unchanged with `--fixed-uncertainty-threshold-m <frozen-value> --split hidden --confirm-hidden`.

Outputs include precision, recall, F1, confusion counts, coverage, flagged rows and threshold-sweep plots. This is a simulated prioritization diagnostic, **not** life-safety certification or a replacement for clinical/industrial procedures.

## 7. 5,000+ stress conditions

Generate the reproducible condition matrix:

```bash
python stress_test.py --output-dir outputs/stress --n-conditions 5000 --seed 4200
```

This produces `stress_conditions.csv` and `stress_manifest.json`. The conditions cover active-anchor count, range/RSSI noise multiplier, NLOS rate and bias, crowd density, packet loss, outage duration, drift rate, and outlier rate/scale. Same seed + same version yields the same unique matrix.

**Condition generation is not the same as model stress evaluation.** A real score requires an adapter that uses the simulator and predictors. Supply it as `module:function` or use the included reference benchmark adapter `--adapter default` (which implements `stress_adapter:evaluate_conditions`):

```bash
# Run 5,000 stress conditions with reference benchmark simulation and heatmap generation:
python stress_test.py --output-dir outputs/stress \
  --n-conditions 5000 --seed 4200 \
  --adapter default \
  --failure-p90-threshold-m 3.0
```

The adapter contract is:

```python
def evaluate_conditions(conditions, *, seed: int, output_dir: Path) -> pandas.DataFrame:
    """Run simulator and predictors, return one row per condition and method."""
    # Required columns: condition_id, method, p90_error_m
    # Optional columns: rmse_m, median_error_m, prediction_coverage, failure_reason, status
    ...
```

The adapter must apply each condition to genuine simulated measurements, run the actual predictors, compare to generated ground truth, and return measured statistics. `stress_test.py` generates:
- `stress_conditions.csv`: The complete reproducible matrix of 5,000+ conditions.
- `stress_scores.csv`: Measured P90, RMSE, and median errors across B1, B2, B3, and Traceline.
- `stress_failure_summary.csv`: Per-method failure rates against the predeclared threshold.
- `stress_failure_boundaries.csv`: Failure rate and median P90 error across stress factor strata.
- `stress_noise_failure_plot.png`: Scatter plot of noise response and failure boundary.
- `stress_failure_heatmap.png`: 2D failure rate heatmap (anchor count vs noise multiplier) for all methods.

## 8. Reproducible commands

### Via Make (Linux / macOS / Git Bash):

```bash
make test        # Run all 26 pytest test cases across 5 test suites
make eval        # Fair evaluation across B1, B2, B3, and Traceline
make errormap    # Train error map on train runs, validate on held-out dev runs
make crowd       # Measure performance inside vs outside crowd zones
make emergency HIGH_ERROR_THRESHOLD_M=0.2  # Select threshold on dev calibration, score on held-out dev
make stress      # Run 5,000-condition stress test with default adapter & heatmap
make report      # Run complete end-to-end evaluation pipeline
```

### Via PowerShell (Windows):

```powershell
.\run_all.ps1 -Target test
.\run_all.ps1 -Target eval
.\run_all.ps1 -Target errormap
.\run_all.ps1 -Target crowd
.\run_all.ps1 -Target emergency -HighErrorThresholdM 0.2
.\run_all.ps1 -Target stress -FailureP90M 3.0
.\run_all.ps1 -Target report
```

### Override paths for team datasets:

```bash
make eval DATA_DIR=../datasets/public PREDICTIONS_DIR=../model_outputs OUTPUT_DIR=outputs
make crowd DATA_DIR=../datasets/public PREDICTIONS_DIR=../model_outputs VENUES_DIR=../shared/venues OUTPUT_DIR=outputs
```
