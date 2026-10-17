# Traceline Workstream C — Drop-in Implementation

This package implements **Track 4 · Problem Statement 2 · Part C** from the supplied *Traceline Prototype Execution Guide*: evaluation, error learning, stress testing, crowd degradation analysis, and emergency uncertainty flags.

## Quick start

1. Extract the archive into the root of your Traceline repository. It provides an `eval/` directory.
2. Before merging, align the standalone `eval/requirements.txt` pins with the frozen `shared/versions.json` in your repository.
3. Run the smoke tests:

   ```bash
   cd eval
   python -m pip install -r requirements.txt
   make test
   make report
   ```

`make report` defaults to an included, deterministic synthetic smoke-test fixture. Those generated numbers verify that code paths work; they are **not** official benchmark results, real-hardware measurements, or proof that Traceline beats the baselines.

## Implemented

- Position error, mean/median/RMSE/P90/P95, per-track aggregation and prediction coverage.
- Fair B1/B2/B3/Traceline report generation, condition slices, plots, uncertainty diagnostics, optional NLOS and future-position sidecars.
- Self-learning error map using train-only residuals and held-out development validation; rejects train/dev run overlap and reports support by distinct trajectories.
- Crowd-zone membership from venue polygon geometry, plus paired with-crowd/no-crowd model comparisons when Part B supplies both outputs.
- Emergency uncertainty flag threshold selection on a calibration subset of development runs and reporting on different development runs.
- Deterministic generation of 5,000+ unique stress-condition configurations and a strict adapter contract for measured scores from Part A's simulator and Part B's frozen models.
- Tests, reproducibility manifests, `README.md`, `DECISIONS.md`, and `VALIDATION.md`.

## Integration boundary

The archive does not include the team's actual Part A simulator, Part B trained model, or the full project repository. Accordingly, the stress matrix generator does **not** invent positioning scores. To complete real 5,000-condition stress scoring, implement the documented adapter in `eval/README.md` against the real simulator and frozen B1/B2/B3/Traceline predictors. Then run it with a failure threshold declared before scoring.

The implementation keeps the guide's shared CSV schemas unchanged. The `run_metadata.csv`, `nlos_probabilities/`, and `predicted_positions/` inputs are documented as evaluation sidecars only.

## Important files

See [`eval/README.md`](eval/README.md) for full input contracts, command usage, output schemas, hidden-set safeguards, and integration instructions. `eval/VALIDATION.md` is the evidence/reporting protocol and intentionally leaves real performance results blank until you run against the team's actual datasets.
