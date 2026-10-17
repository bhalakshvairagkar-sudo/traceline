# Workstream C — DECISIONS

1. **Ownership boundary:** all code and outputs from this workstream are under `eval/`. Shared files, Part A/B/D folders and the finished marketing site are not modified.
2. **Contracts:** `measurements.csv`, `truth.csv`, `labels_nlos.csv`, `crowd.csv` and canonical `predictions.csv` schemas remain unchanged. Evaluation-only sidecars (`run_metadata.csv`, `nlos_probabilities/<method>.csv`, `predicted_positions/<method>.csv`) are documented separately and must not be confused with shared contract files.
3. **Missing predictions:** scoring left-joins predictions to truth and reports prediction coverage and missing counts. Error metrics only summarize finite predictions; never interpret these without coverage.
4. **Track weighting:** point metrics are reported along with per-track metrics, so very long trajectories do not dominate silently.
5. **Error map:** cells use predicted position at inference; support count is unique training-run count. Corrections are learned on train and validated on disjoint dev run IDs. Out-of-fold train predictions are preferred to reduce residual overfit bias.
6. **Uncertainty semantics:** empirical error-versus-sigma diagnostics are reported without assuming nominal confidence coverage until sigma semantics are independently verified.
7. **Emergency flags:** the high-error definition is a caller-supplied predeclared threshold. An uncertainty threshold is selected on a calibration subset of dev runs and evaluated on separate dev runs. Final hidden evaluation requires the frozen threshold.
8. **Stress testing:** generated condition grids are not labelled as model scores. Scores require an adapter to the actual simulator and frozen model. The P90 failure criterion is caller-supplied before evaluation.
9. **Synthetic fixture:** included only for smoke tests; it intentionally contains controlled biases and must never be cited as competition performance.
10. **Dependency alignment:** `requirements.txt` makes this folder runnable standalone, but exact package versions must be reconciled with the repository's frozen `shared/versions.json` before merge.
11. **Reference Stress Adapter & Heatmap:** `stress_adapter.py` provides a self-contained, reproducible signal-model simulator and baseline solver (B1, B2, B3, Traceline) to verify the 5,000-condition stress runner and generate 2D failure-boundary heatmaps end-to-end without external dependencies. Official competition scoring substitutes this with the team's official simulator and frozen model via `--adapter module:function`.

All choices not specified by the guide are recorded here for team review. Any shared-contract change must follow the guide's four-member approval process.
