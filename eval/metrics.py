"""Positioning, NLOS, uncertainty and forecasting metrics for Traceline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both direct script execution and package imports.
    from .common import PREDICTION_COLUMNS, TRUTH_COLUMNS, ensure_unique, finite_or_nan, read_contract_csv, write_manifest, write_json
except ImportError:  # pragma: no cover - exercised by the standalone CLI
    from common import PREDICTION_COLUMNS, TRUTH_COLUMNS, ensure_unique, finite_or_nan, read_contract_csv, write_manifest, write_json


def position_errors(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Left-join predictions to truth and compute Euclidean error and coverage."""
    ensure_unique(truth, ["run_id", "t"], "truth")
    missing = TRUTH_COLUMNS - set(truth.columns)
    if missing:
        raise ValueError(f"truth missing columns: {sorted(missing)}")
    # Ground truth and join keys are never silently discarded: invalid truth means the sample
    # cannot be scored reliably and should be fixed at data preparation time.
    truth_numeric = truth[["t", "x_m", "y_m"]].apply(pd.to_numeric, errors="coerce")
    if truth_numeric.isna().any().any() or not np.isfinite(truth_numeric.to_numpy(dtype=float)).all():
        raise ValueError("truth contains non-numeric/non-finite t or position values")
    if truth["run_id"].isna().any() or truth["run_id"].astype(str).str.strip().eq("").any():
        raise ValueError("truth contains empty run_id values")
    missing_pred = PREDICTION_COLUMNS - set(predictions.columns)
    if missing_pred:
        raise ValueError(f"predictions missing columns: {sorted(missing_pred)}")
    ensure_unique(predictions, ["run_id", "t"], "predictions")
    pred_time = pd.to_numeric(predictions["t"], errors="coerce")
    if pred_time.isna().any() or not np.isfinite(pred_time.to_numpy(dtype=float)).all():
        raise ValueError("predictions contains non-numeric/non-finite t values")
    if predictions["run_id"].isna().any() or predictions["run_id"].astype(str).str.strip().eq("").any():
        raise ValueError("predictions contains empty run_id values")

    base_truth = truth[["run_id", "t", "x_m", "y_m"]].copy()
    pred = predictions[["run_id", "t", "x_hat_m", "y_hat_m", "sigma_m", "flags"]].copy()
    for col in ("t", "x_m", "y_m"):
        base_truth[col] = finite_or_nan(base_truth[col])
    for col in ("t", "x_hat_m", "y_hat_m", "sigma_m"):
        pred[col] = finite_or_nan(pred[col])
    joined = base_truth.merge(pred, on=["run_id", "t"], how="left", validate="one_to_one", indicator=True)
    joined["has_prediction"] = joined["_merge"].eq("both") & joined["x_hat_m"].notna() & joined["y_hat_m"].notna()
    dx = joined["x_hat_m"] - joined["x_m"]
    dy = joined["y_hat_m"] - joined["y_m"]
    joined["error_m"] = np.sqrt(dx * dx + dy * dy)
    joined.loc[~joined["has_prediction"], "error_m"] = np.nan
    return joined.drop(columns=["_merge"])


def summarize_errors(errors: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    """Calculate requested point-error statistics on finite predictions only."""
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean_m": None, "median_m": None, "rmse_m": None,
                "p90_m": None, "p95_m": None, "max_m": None}
    return {
        "n": int(arr.size),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr ** 2))),
        "p90_m": float(np.quantile(arr, 0.90)),
        "p95_m": float(np.quantile(arr, 0.95)),
        "max_m": float(np.max(arr)),
    }


def track_level_metrics(step_errors: pd.DataFrame) -> pd.DataFrame:
    """Return one record per run; tracks receive equal weight in downstream summaries."""
    records: list[dict[str, Any]] = []
    for run_id, group in step_errors.groupby("run_id", sort=True):
        truth_count = len(group)
        valid = group["error_m"].dropna().to_numpy(dtype=float)
        stats = summarize_errors(valid)
        records.append({
            "run_id": str(run_id),
            "n_truth_steps": int(truth_count),
            "n_predicted_steps": int(group["has_prediction"].sum()),
            "prediction_coverage": float(group["has_prediction"].mean()) if truth_count else None,
            "mean_error_m": stats["mean_m"],
            "median_error_m": stats["median_m"],
            "rmse_error_m": stats["rmse_m"],
            "p90_error_m": stats["p90_m"],
            "p95_error_m": stats["p95_m"],
            "max_error_m": stats["max_m"],
        })
    return pd.DataFrame.from_records(records)


def summarize_tracks(track_metrics: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize track-level distributions, avoiding long tracks dominating the result."""
    if track_metrics.empty:
        return {"n_tracks": 0, "track_mean_error_m": None, "track_median_error_m": None,
                "track_rmse_error_m": None, "track_p90_error_m": None, "track_p95_error_m": None,
                "mean_track_coverage": None}
    values = track_metrics["mean_error_m"].dropna().to_numpy(dtype=float)
    rmse = track_metrics["rmse_error_m"].dropna().to_numpy(dtype=float)
    coverage = track_metrics["prediction_coverage"].dropna().to_numpy(dtype=float)
    return {
        "n_tracks": int(len(track_metrics)),
        "track_mean_error_m": float(values.mean()) if values.size else None,
        "track_median_error_m": float(np.median(values)) if values.size else None,
        "track_rmse_error_m": float(np.sqrt(np.mean(rmse ** 2))) if rmse.size else None,
        "track_p90_error_m": float(np.quantile(values, 0.90)) if values.size else None,
        "track_p95_error_m": float(np.quantile(values, 0.95)) if values.size else None,
        "mean_track_coverage": float(coverage.mean()) if coverage.size else None,
    }


def binary_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, Any]:
    """Calculate binary precision, recall, F1 and confusion counts with safe zero handling."""
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have equal shapes")
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(truth.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def emergency_flag_precision(actual_high_error: pd.Series | np.ndarray, uncertainty_flagged: pd.Series | np.ndarray) -> dict[str, Any]:
    """Calculate emergency-flag precision, recall, F1 and confusion counts against actual high-error steps."""
    metrics = binary_classification_metrics(actual_high_error, uncertainty_flagged)
    metrics["flag_precision"] = metrics["precision"]
    metrics["flag_recall"] = metrics["recall"]
    metrics["flag_f1"] = metrics["f1"]
    return metrics


def nlos_metrics(labels: pd.DataFrame, probabilities: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    """Score per-anchor NLOS probabilities against labels_nlos.csv labels."""
    required_labels = {"run_id", "t", "anchor_id", "is_nlos"}
    required_probs = {"run_id", "t", "anchor_id", "nlos_probability"}
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"NLOS labels missing columns: {sorted(missing)}")
    if missing := required_probs - set(probabilities.columns):
        raise ValueError(f"NLOS probabilities missing columns: {sorted(missing)}")
    ensure_unique(labels, ["run_id", "t", "anchor_id"], "NLOS labels")
    ensure_unique(probabilities, ["run_id", "t", "anchor_id"], "NLOS probabilities")
    joined = labels[["run_id", "t", "anchor_id", "is_nlos"]].merge(
        probabilities[["run_id", "t", "anchor_id", "nlos_probability"]],
        on=["run_id", "t", "anchor_id"], how="left", validate="one_to_one")
    joined["nlos_probability"] = finite_or_nan(joined["nlos_probability"])
    joined = joined.loc[joined["nlos_probability"].notna()].copy()
    if joined.empty:
        return {"n": 0, "threshold": threshold, "precision": None, "recall": None, "f1": None,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "probability_coverage": 0.0}
    if ((joined["nlos_probability"] < 0) | (joined["nlos_probability"] > 1)).any():
        raise ValueError("nlos_probability values must be within [0, 1]")
    result = binary_classification_metrics(joined["is_nlos"], joined["nlos_probability"] >= threshold)
    result["threshold"] = float(threshold)
    result["probability_coverage"] = float(len(joined) / len(labels)) if len(labels) else 0.0
    return result


def uncertainty_calibration(step_errors: pd.DataFrame, n_bins: int = 10) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate sigma as a scale estimate without assuming unverified nominal coverage semantics.

    Reports empirical coverage for error <= k*sigma, and grouped actual RMSE versus predicted
    sigma. It intentionally does not call these confidence levels unless their semantics have
    been established by independent calibration.
    """
    data = step_errors.loc[
        step_errors["error_m"].notna() & step_errors["sigma_m"].notna() & (step_errors["sigma_m"] >= 0)
    ].copy()
    if data.empty:
        return ({"n": 0, "sigma_error_spearman": None, "coverage_by_sigma_multiplier": {},
                 "calibration_bins": []}, pd.DataFrame())
    e = data["error_m"].to_numpy(dtype=float)
    s = data["sigma_m"].to_numpy(dtype=float)
    coverage = {str(mult): float(np.mean(e <= mult * s)) for mult in (0.5, 1.0, 1.5, 2.0, 2.5)}
    corr = data[["error_m", "sigma_m"]].corr(method="spearman").iloc[0, 1]
    # qcut can collapse bins when many uncertainty values are tied; duplicates='drop' handles this.
    try:
        data["sigma_bin"] = pd.qcut(data["sigma_m"], q=min(n_bins, data["sigma_m"].nunique()), duplicates="drop")
    except ValueError:
        data["sigma_bin"] = "all"
    if data["sigma_bin"].nunique(dropna=True) == 0:
        data["sigma_bin"] = "all"
    bins = data.groupby("sigma_bin", observed=True).agg(
        n=("error_m", "size"),
        mean_sigma_m=("sigma_m", "mean"),
        mean_error_m=("error_m", "mean"),
        rmse_error_m=("error_m", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        p90_error_m=("error_m", lambda values: float(np.quantile(values, .90))),
    ).reset_index()
    bins["sigma_bin"] = bins["sigma_bin"].astype(str)
    payload = {
        "n": int(len(data)),
        "sigma_error_spearman": float(corr) if pd.notna(corr) else None,
        "coverage_by_sigma_multiplier": coverage,
        "calibration_bins": bins.to_dict("records"),
        "interpretation_note": "Empirical scale diagnostics only; nominal probability coverage is not asserted unless sigma_m semantics are separately validated.",
    }
    return payload, bins


def forecast_error(forecast_predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Score optional future-position forecasts stored as JSON arrays in a sidecar CSV.

    Expected columns: run_id,t,horizon_steps,x_hat_m,y_hat_m; t is the forecast origin and
    horizon_steps is the positive offset to the truth timestamp t + horizon_steps*dt_s.
    This explicit sidecar format keeps the shared predictions.csv contract unchanged.
    """
    required = {"run_id", "t", "horizon_steps", "x_hat_m", "y_hat_m"}
    if missing := required - set(forecast_predictions.columns):
        raise ValueError(f"Forecast sidecar missing columns: {sorted(missing)}")
    ensure_unique(truth, ["run_id", "t"], "truth")
    ensure_unique(forecast_predictions, ["run_id", "t", "horizon_steps"], "forecast_predictions")
    truth_index = truth[["run_id", "t", "x_m", "y_m"]].copy()
    step = pd.to_numeric(forecast_predictions["horizon_steps"], errors="coerce")
    if step.isna().any() or (step <= 0).any() or not np.allclose(step, np.round(step)):
        raise ValueError("horizon_steps must be positive integers")
    # Contract dt is fixed at 0.5s, so a horizon is represented as a number of steps.
    forecast = forecast_predictions.copy()
    forecast["target_t"] = forecast["t"] + forecast["horizon_steps"].astype(float) * 0.5
    target = truth_index.rename(columns={"t": "target_t", "x_m": "truth_x_m", "y_m": "truth_y_m"})
    joined = forecast.merge(target, on=["run_id", "target_t"], how="left", validate="many_to_one")
    joined["forecast_error_m"] = np.sqrt(
        (pd.to_numeric(joined["x_hat_m"], errors="coerce") - joined["truth_x_m"]) ** 2
        + (pd.to_numeric(joined["y_hat_m"], errors="coerce") - joined["truth_y_m"]) ** 2
    )
    joined["horizon_steps"] = joined["horizon_steps"].astype(int)
    return joined


def main() -> None:
    """Provide a small CLI for direct position-error summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Path to contract truth.csv")
    parser.add_argument("--predictions", required=True, help="Path to contract predictions.csv")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args()
    truth = read_contract_csv(args.truth, TRUTH_COLUMNS, "truth.csv")
    pred = read_contract_csv(args.predictions, PREDICTION_COLUMNS, "predictions.csv")
    result = position_errors(pred, truth)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, summarize_errors(result["error_m"]))
    write_manifest(Path(args.out).with_name("manifest.json"), {
        "script": "metrics.py", "truth_path": args.truth, "predictions_path": args.predictions, "output_path": args.out,
    }, extra={"scored_rows": int(len(result))})


if __name__ == "__main__":
    main()
