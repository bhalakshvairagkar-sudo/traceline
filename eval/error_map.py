"""Learn and validate location-specific residual corrections without trajectory leakage."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both direct script execution and package imports.
    from .common import PREDICTION_COLUMNS, TRUTH_COLUMNS, add_run_identity, ensure_unique, read_contract_csv, write_json, write_manifest, json_safe
    from .metrics import position_errors, summarize_errors
except ImportError:  # pragma: no cover - exercised by standalone CLI
    from common import PREDICTION_COLUMNS, TRUTH_COLUMNS, add_run_identity, ensure_unique, read_contract_csv, write_json, write_manifest, json_safe
    from metrics import position_errors, summarize_errors


def _scored_positions(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join predictions and truth and retain residual components for error-map fitting."""
    joined = position_errors(predictions, truth)
    joined = add_run_identity(joined)
    joined["dx_m"] = joined["x_m"] - joined["x_hat_m"]
    joined["dy_m"] = joined["y_m"] - joined["y_hat_m"]
    return joined


def fit_error_map(train_scored: pd.DataFrame, cell_size_m: float = 1.0,
                  min_support_runs: int = 3) -> pd.DataFrame:
    """Fit a per-venue cell correction using per-run mean residuals.

    Cells are assigned using the *predicted* coordinate, since true coordinates are unavailable
    at inference. Each run contributes at most one residual vector per cell to avoid allowing a
    long trajectory to dominate the learned correction. support_count therefore counts distinct
    trajectories, not time steps.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if min_support_runs < 1:
        raise ValueError("min_support_runs must be >= 1")
    required = {"run_id", "venue_id", "x_hat_m", "y_hat_m", "dx_m", "dy_m", "has_prediction"}
    if missing := required - set(train_scored.columns):
        raise ValueError(f"Training scored data missing columns: {sorted(missing)}")
    data = train_scored.loc[
        train_scored["has_prediction"] & train_scored["dx_m"].notna() & train_scored["dy_m"].notna()
    ].copy()
    if data.empty:
        raise ValueError("No valid training predictions available to fit an error map")
    data["cell_x"] = np.floor(data["x_hat_m"] / cell_size_m).astype(int)
    data["cell_y"] = np.floor(data["y_hat_m"] / cell_size_m).astype(int)
    # Equalize track contribution: one vector per run, venue and cell.
    run_cell = data.groupby(["venue_id", "run_id", "cell_x", "cell_y"], as_index=False).agg(
        run_dx_m=("dx_m", "mean"), run_dy_m=("dy_m", "mean"), observations_in_run_cell=("dx_m", "size"),
    )
    cells = run_cell.groupby(["venue_id", "cell_x", "cell_y"], as_index=False).agg(
        support_count=("run_id", "nunique"),
        observation_count=("observations_in_run_cell", "sum"),
        mean_dx_m=("run_dx_m", "mean"),
        mean_dy_m=("run_dy_m", "mean"),
        residual_std_x_m=("run_dx_m", "std"),
        residual_std_y_m=("run_dy_m", "std"),
    )
    cells[["residual_std_x_m", "residual_std_y_m"]] = cells[["residual_std_x_m", "residual_std_y_m"]].fillna(0.0)
    cells["proposed_dx_m"] = cells["mean_dx_m"]
    cells["proposed_dy_m"] = cells["mean_dy_m"]
    cells["proposed_correction_m"] = np.sqrt(cells["proposed_dx_m"] ** 2 + cells["proposed_dy_m"] ** 2)
    cells["correction_uncertainty_m"] = np.sqrt(cells["residual_std_x_m"] ** 2 + cells["residual_std_y_m"] ** 2)
    cells["cell_center_x_m"] = (cells["cell_x"] + 0.5) * cell_size_m
    cells["cell_center_y_m"] = (cells["cell_y"] + 0.5) * cell_size_m
    cells["cell_size_m"] = float(cell_size_m)
    cells["min_support_runs"] = int(min_support_runs)
    cells["correction_applied_by_default"] = cells["support_count"] >= min_support_runs
    cells.loc[~cells["correction_applied_by_default"], ["mean_dx_m", "mean_dy_m"]] = 0.0
    cells["correction_m"] = np.sqrt(cells["mean_dx_m"] ** 2 + cells["mean_dy_m"] ** 2)
    cells["uncertainty_semantics"] = "between-run residual-vector dispersion; not calibrated probabilistic uncertainty"
    return cells.sort_values(["venue_id", "cell_x", "cell_y"]).reset_index(drop=True)


def apply_error_map(predictions: pd.DataFrame, error_map: pd.DataFrame,
                    cell_size_m: float = 1.0, min_support_runs: int = 3) -> pd.DataFrame:
    """Apply learned corrections based only on predicted location; unsupported cells stay unchanged."""
    result = add_run_identity(predictions)
    result["x_hat_corrected_m"] = pd.to_numeric(result["x_hat_m"], errors="coerce")
    result["y_hat_corrected_m"] = pd.to_numeric(result["y_hat_m"], errors="coerce")
    result["error_map_support_runs"] = 0
    result["error_map_correction_applied"] = False
    if error_map.empty:
        return result
    mapping = error_map.set_index(["venue_id", "cell_x", "cell_y"])
    valid = result["x_hat_m"].notna() & result["y_hat_m"].notna()
    result.loc[valid, "cell_x"] = np.floor(pd.to_numeric(result.loc[valid, "x_hat_m"]) / cell_size_m).astype(int)
    result.loc[valid, "cell_y"] = np.floor(pd.to_numeric(result.loc[valid, "y_hat_m"]) / cell_size_m).astype(int)
    for idx, row in result.loc[valid].iterrows():
        key = (row["venue_id"], int(row["cell_x"]), int(row["cell_y"]))
        if key not in mapping.index:
            continue
        cell = mapping.loc[key]
        support = int(cell["support_count"])
        result.at[idx, "error_map_support_runs"] = support
        if support >= min_support_runs:
            result.at[idx, "x_hat_corrected_m"] = float(row["x_hat_m"]) + float(cell["mean_dx_m"])
            result.at[idx, "y_hat_corrected_m"] = float(row["y_hat_m"]) + float(cell["mean_dy_m"])
            result.at[idx, "error_map_correction_applied"] = True
    return result


def validate_error_map(train_predictions: pd.DataFrame, dev_predictions: pd.DataFrame,
                       truth: pd.DataFrame, cell_size_m: float = 1.0,
                       min_support_runs: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit on training run IDs and evaluate before/after on disjoint development run IDs."""
    train_identified = add_run_identity(train_predictions)
    dev_identified = add_run_identity(dev_predictions)
    train_ids = set(train_predictions["run_id"].astype(str))
    dev_ids = set(dev_predictions["run_id"].astype(str))
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(f"Trajectory leakage: train and dev predictions share run IDs: {sorted(overlap)[:5]}")
    invalid_train = sorted(train_identified.loc[train_identified["split"] != "train", "run_id"].astype(str).unique().tolist())
    invalid_dev = sorted(dev_identified.loc[dev_identified["split"] != "dev", "run_id"].astype(str).unique().tolist())
    if invalid_train:
        raise ValueError(f"Training error-map inputs must contain train run IDs only: {invalid_train[:5]}")
    if invalid_dev:
        raise ValueError(f"Error-map validation inputs must contain dev run IDs only: {invalid_dev[:5]}")
    train_truth = truth.loc[truth["run_id"].astype(str).isin(train_ids)].copy()
    dev_truth = truth.loc[truth["run_id"].astype(str).isin(dev_ids)].copy()
    if train_truth.empty or dev_truth.empty:
        raise ValueError("Both train and dev truth rows are required")
    train_scored = _scored_positions(train_predictions, train_truth)
    dev_scored = _scored_positions(dev_predictions, dev_truth)
    error_map = fit_error_map(train_scored, cell_size_m, min_support_runs)
    corrected = apply_error_map(dev_predictions, error_map, cell_size_m, min_support_runs)
    dev_join = dev_truth[["run_id", "t", "x_m", "y_m"]].merge(
        corrected[["run_id", "t", "x_hat_m", "y_hat_m", "x_hat_corrected_m", "y_hat_corrected_m",
                  "error_map_support_runs", "error_map_correction_applied"]],
        on=["run_id", "t"], how="left", validate="one_to_one")
    dev_join["before_error_m"] = np.sqrt((dev_join["x_hat_m"] - dev_join["x_m"]) ** 2 + (dev_join["y_hat_m"] - dev_join["y_m"]) ** 2)
    dev_join["after_error_m"] = np.sqrt((dev_join["x_hat_corrected_m"] - dev_join["x_m"]) ** 2 + (dev_join["y_hat_corrected_m"] - dev_join["y_m"]) ** 2)
    before = summarize_errors(dev_join["before_error_m"])
    after = summarize_errors(dev_join["after_error_m"])
    delta = {}
    for metric in ("mean_m", "median_m", "rmse_m", "p90_m", "p95_m"):
        b, a = before[metric], after[metric]
        delta[f"{metric}_relative_change"] = (float((a - b) / b) if b not in (None, 0) and a is not None else None)
    summary = {
        "train_runs": int(len(train_ids)),
        "dev_runs": int(len(dev_ids)),
        "cell_size_m": float(cell_size_m),
        "min_support_runs": int(min_support_runs),
        "supported_cells": int((error_map["support_count"] >= min_support_runs).sum()),
        "total_cells": int(len(error_map)),
        "dev_truth_steps": int(len(dev_join)),
        "dev_prediction_coverage": float((dev_join["x_hat_m"].notna() & dev_join["y_hat_m"].notna()).mean()),
        "correction_applied_fraction_of_predicted_steps": float(
            dev_join.loc[dev_join["x_hat_m"].notna() & dev_join["y_hat_m"].notna(), "error_map_correction_applied"].fillna(False).mean()
        ),
        "before": before,
        "after": after,
        "change": delta,
        "claim_rule": "Report improvement only if measured on the held-out development trajectories; no improvement is assumed.",
    }
    return error_map, dev_join, json_safe(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Path to truth.csv")
    parser.add_argument("--predictions", required=True, help="Single method predictions CSV with both train and dev run IDs")
    parser.add_argument("--output-dir", default="eval_output/error_map")
    parser.add_argument("--cell-size-m", type=float, default=1.0)
    parser.add_argument("--min-support-runs", type=int, default=3)
    parser.add_argument("--train-prediction-provenance", choices=["out_of_fold", "in_sample", "unknown"], default="unknown",
                        help="Document how train predictions were produced; out_of_fold is preferred")
    args = parser.parse_args()
    truth = read_contract_csv(args.truth, TRUTH_COLUMNS, "truth.csv")
    predictions = read_contract_csv(args.predictions, PREDICTION_COLUMNS, "predictions.csv")
    ensure_unique(predictions, ["run_id", "t"], "predictions")
    identity = add_run_identity(predictions)
    train_predictions = identity.loc[identity["split"] == "train", list(PREDICTION_COLUMNS)].copy()
    dev_predictions = identity.loc[identity["split"] == "dev", list(PREDICTION_COLUMNS)].copy()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    error_map, validation, summary = validate_error_map(train_predictions, dev_predictions, truth,
                                                        args.cell_size_m, args.min_support_runs)
    error_map.to_csv(out / "error_map_cells.csv", index=False)
    validation.to_csv(out / "error_map_dev_validation.csv", index=False)
    summary["train_prediction_provenance"] = args.train_prediction_provenance
    if args.train_prediction_provenance != "out_of_fold":
        summary["provenance_warning"] = (
            "Training predictions may be in-sample. Treat the map as exploratory unless train residuals came from out-of-fold "
            "or otherwise held-out predictions. Development validation remains trajectory-separated."
        )
    write_json(out / "error_map_validation.json", summary)
    write_manifest(out / "manifest.json", {
        "script": "error_map.py", "truth_path": args.truth, "predictions_path": args.predictions,
        "cell_size_m": args.cell_size_m, "min_support_runs": args.min_support_runs,
        "train_prediction_provenance": args.train_prediction_provenance,
    }, extra={"train_runs": summary["train_runs"], "dev_runs": summary["dev_runs"], "validation_summary": summary})

    supported = error_map.loc[error_map["correction_applied_by_default"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    if not supported.empty:
        q = ax.quiver(supported["cell_center_x_m"], supported["cell_center_y_m"],
                      supported["mean_dx_m"], supported["mean_dy_m"], supported["correction_m"], angles="xy", scale_units="xy", scale=1)
        fig.colorbar(q, ax=ax, label="Correction magnitude (m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Learned location-specific corrections (supported cells only)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "error_map_vectors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Before", "After"], [summary["before"]["rmse_m"] or 0, summary["after"]["rmse_m"] or 0])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Error-map validation on held-out development trajectories")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "error_map_before_after.png", dpi=160)
    plt.close(fig)
    print(f"Wrote error-map outputs to {out}")
    print(pd.Series({"before_rmse_m": summary["before"]["rmse_m"], "after_rmse_m": summary["after"]["rmse_m"],
                     "supported_cells": summary["supported_cells"]}).to_string())


if __name__ == "__main__":
    main()
