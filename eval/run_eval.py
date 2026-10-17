"""Fair, reproducible evaluation runner for B1, B2, B3 and Traceline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Support both direct script execution and package imports.
    from .common import (
        PREDICTION_COLUMNS, build_run_features, filter_split, load_data_bundle,
        parse_run_id, add_run_identity, read_contract_csv, write_json, write_manifest, json_safe,
    )
    from .metrics import (
        nlos_metrics, position_errors, summarize_errors, summarize_tracks,
        track_level_metrics, uncertainty_calibration, forecast_error,
        binary_classification_metrics,
    )
except ImportError:  # pragma: no cover - exercised by standalone CLI
    from common import (
        PREDICTION_COLUMNS, build_run_features, filter_split, load_data_bundle,
        parse_run_id, add_run_identity, read_contract_csv, write_json, write_manifest, json_safe,
    )
    from metrics import (
        nlos_metrics, position_errors, summarize_errors, summarize_tracks,
        track_level_metrics, uncertainty_calibration, forecast_error,
        binary_classification_metrics,
    )

DEFAULT_METHODS = ["B1", "B2", "B3", "Traceline"]


def _finite_metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize a scored frame, including prediction availability."""
    point = summarize_errors(frame["error_m"])
    tracks = track_level_metrics(frame)
    track = summarize_tracks(tracks)
    denominator = len(frame)
    coverage = float(frame["has_prediction"].mean()) if denominator else 0.0
    return {
        **point,
        **track,
        "n_truth_steps": int(denominator),
        "n_missing_predictions": int((~frame["has_prediction"]).sum()),
        "prediction_coverage": coverage,
    }


def _attach_strata(scored: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Attach run-level evaluation strata to per-step scored predictions."""
    return scored.merge(features, on="run_id", how="left", suffixes=("", "_feature"), validate="many_to_one")


def _slice_report(scored: pd.DataFrame, method: str) -> list[dict[str, Any]]:
    """Generate useful robustness slices without changing any shared data files."""
    preferred = [
        "venue_id", "active_anchor_count", "noise_level", "noise_multiplier",
        "drift_condition", "outlier_condition", "dropout_duration_steps", "scenario_id",
    ]
    output: list[dict[str, Any]] = []
    for column in preferred:
        if column not in scored.columns:
            continue
        local = scored.loc[scored[column].notna()]
        if local.empty or local[column].nunique(dropna=True) < 1:
            continue
        for value, group in local.groupby(column, dropna=False, sort=True):
            stats = _finite_metrics_for_frame(group)
            output.append({"method": method, "slice_dimension": column, "slice_value": str(value), **stats})

    # Empirical strata when injection metadata is not available.
    if "nlos_fraction_observed" in scored.columns:
        temp = scored.copy()
        temp["nlos_stratum"] = pd.cut(
            temp["nlos_fraction_observed"], bins=[-0.001, 0.10, 0.30, 1.001],
            labels=["low (<=0.10)", "moderate (0.10-0.30)", "high (>0.30)"], include_lowest=True,
        )
        for value, group in temp.loc[temp["nlos_stratum"].notna()].groupby("nlos_stratum", observed=True):
            output.append({"method": method, "slice_dimension": "nlos_stratum", "slice_value": str(value), **_finite_metrics_for_frame(group)})
    if "packet_loss_rate_observed" in scored.columns:
        temp = scored.copy()
        temp["packet_loss_stratum"] = pd.cut(
            temp["packet_loss_rate_observed"], bins=[-0.001, 0.05, 0.20, 1.001],
            labels=["low (<=0.05)", "moderate (0.05-0.20)", "high (>0.20)"], include_lowest=True,
        )
        for value, group in temp.loc[temp["packet_loss_stratum"].notna()].groupby("packet_loss_stratum", observed=True):
            output.append({"method": method, "slice_dimension": "packet_loss_stratum", "slice_value": str(value), **_finite_metrics_for_frame(group)})
    if "crowd_density_mean_people_m2" in scored.columns:
        temp = scored.loc[scored["crowd_density_mean_people_m2"].notna()]
        for value, group in temp.groupby(pd.cut(temp["crowd_density_mean_people_m2"], bins=[-0.001, 0.5, 2.0, 4.001],
                                                labels=["lower (<=0.5)", "moderate (0.5-2)", "dense (>2)"], include_lowest=True), observed=True):
            output.append({"method": method, "slice_dimension": "crowd_density_stratum", "slice_value": str(value), **_finite_metrics_for_frame(group)})
    return output


def _read_nlos_probs(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    expected = {"run_id", "t", "anchor_id", "nlos_probability"}
    frame = pd.read_csv(path)
    if missing := expected - set(frame.columns):
        raise ValueError(f"{path} missing required NLOS probability columns: {sorted(missing)}")
    return frame


def _choose_nlos_threshold(labels: pd.DataFrame, probs: pd.DataFrame) -> float:
    """Select an NLOS threshold on development data by maximum F1, with deterministic tie-break."""
    joined = labels[["run_id", "t", "anchor_id", "is_nlos"]].merge(
        probs[["run_id", "t", "anchor_id", "nlos_probability"]],
        on=["run_id", "t", "anchor_id"], how="inner", validate="one_to_one")
    if joined.empty:
        return 0.5
    joined = joined.loc[joined["nlos_probability"].notna()]
    if joined.empty:
        return 0.5
    options = np.linspace(0.05, 0.95, 19)
    scored = []
    for threshold in options:
        result = binary_classification_metrics(joined["is_nlos"], joined["nlos_probability"] >= threshold)
        scored.append((result["f1"], result["precision"], float(threshold)))
    # Max F1; if tied prefer higher precision, then higher threshold to reduce false positives.
    return max(scored)[2]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute evaluation for all configured methods and write reports/plots."""
    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_data_bundle(data_dir)
    truth = bundle["truth"]
    assert isinstance(truth, pd.DataFrame)
    truth = truth.copy()
    truth["split"] = truth["run_id"].astype(str).map(lambda value: parse_run_id(value)["split"])
    contains_hidden = bool((truth["split"] == "hidden").any())
    nlos_labels_exist = isinstance(bundle.get("labels_nlos"), pd.DataFrame)
    nlos_sidecars_exist = any((predictions_dir / "nlos_probabilities" / f"{method}.csv").exists() for method in args.methods)
    needs_frozen_nlos_threshold = nlos_labels_exist and nlos_sidecars_exist
    if args.split == "hidden" and not args.confirm_hidden:
        raise ValueError("Hidden-set evaluation is guarded. Add --confirm-hidden only after model/settings freeze and agreement with Part A.")
    if args.split == "hidden" and needs_frozen_nlos_threshold and args.nlos_threshold is None:
        raise ValueError("For hidden evaluation with NLOS scoring, provide the frozen --nlos-threshold selected on development data.")
    if args.split == "all" and contains_hidden and not args.confirm_hidden:
        raise ValueError("Split=all includes hidden rows in this dataset; prefer train/dev or explicitly confirm a post-freeze evaluation.")
    if args.split == "all" and contains_hidden and needs_frozen_nlos_threshold and args.nlos_threshold is None:
        raise ValueError("Split=all includes hidden rows; supply the frozen --nlos-threshold selected on development data.")

    features = build_run_features(bundle)
    truth_subset = filter_split(truth, args.split)
    if truth_subset.empty:
        raise ValueError(f"No truth rows found for split={args.split!r}")
    truth_subset = truth_subset[["run_id", "t", "x_m", "y_m"]].copy()
    feature_subset = features.loc[features["run_id"].isin(truth_subset["run_id"].unique())].copy()

    all_summary: dict[str, Any] = {
        "project": "Traceline Workstream C evaluation",
        "split": args.split,
        "methods": {},
        "settings": {"nlos_threshold_argument": args.nlos_threshold, "sigma_bins": args.sigma_bins},
        "data_notes": [
            "Position error is computed by left-joining predictions to all scored truth steps; missing predictions count against coverage.",
            "All headline error metrics are computed on available finite predictions and must be read alongside coverage.",
            "Synthetic data results are not real-world hardware measurements.",
        ],
    }
    summaries: list[dict[str, Any]] = []
    tracks_frames: list[pd.DataFrame] = []
    slices: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    cdf_frames: list[tuple[str, np.ndarray]] = []
    forecast_rows: list[dict[str, Any]] = []

    all_labels = bundle.get("labels_nlos")
    labels_for_evaluation = None
    labels_for_threshold_selection = None
    if isinstance(all_labels, pd.DataFrame):
        labels_for_evaluation = all_labels.loc[all_labels["run_id"].isin(truth_subset["run_id"].unique())].copy()
        labels_for_evaluation["split"] = labels_for_evaluation["run_id"].astype(str).map(lambda value: parse_run_id(value)["split"])
        labels_for_evaluation = filter_split(labels_for_evaluation, args.split)
        labels_for_threshold_selection = all_labels.copy()
        labels_for_threshold_selection["split"] = labels_for_threshold_selection["run_id"].astype(str).map(lambda value: parse_run_id(value)["split"])
        labels_for_threshold_selection = labels_for_threshold_selection.loc[labels_for_threshold_selection["split"] == "dev"].copy()

    for method in args.methods:
        prediction_path = predictions_dir / f"{method}.csv"
        pred_all = read_contract_csv(prediction_path, PREDICTION_COLUMNS, f"predictions for {method}")
        pred = filter_split(add_run_identity(pred_all), args.split)
        pred = pred[["run_id", "t", "x_hat_m", "y_hat_m", "sigma_m", "flags"]]
        # Left join against the same truth rows for every method; missing predictions remain visible.
        scored = position_errors(pred, truth_subset)
        scored = _attach_strata(scored, feature_subset)
        stats = _finite_metrics_for_frame(scored)
        calibration, calibration_bins = uncertainty_calibration(scored, n_bins=args.sigma_bins)
        method_summary: dict[str, Any] = {**stats, "uncertainty_calibration": calibration}

        nlos_sidecar = predictions_dir / "nlos_probabilities" / f"{method}.csv"
        nlos_result = None
        if isinstance(labels_for_evaluation, pd.DataFrame) and nlos_sidecar.exists():
            probs_all = _read_nlos_probs(nlos_sidecar)
            assert probs_all is not None
            probs_all = add_run_identity(probs_all)
            probs_for_evaluation = probs_all.loc[probs_all["run_id"].isin(truth_subset["run_id"].unique())].copy()
            probs_for_evaluation = filter_split(probs_for_evaluation, args.split)
            threshold = args.nlos_threshold
            if threshold is None:
                assert isinstance(labels_for_threshold_selection, pd.DataFrame)
                probs_for_selection = probs_all.loc[probs_all["split"] == "dev"].copy()
                threshold = _choose_nlos_threshold(labels_for_threshold_selection, probs_for_selection)
            labels_eval = labels_for_evaluation[["run_id", "t", "anchor_id", "is_nlos"]]
            nlos_result = nlos_metrics(labels_eval, probs_for_evaluation, threshold=float(threshold))
            method_summary["nlos_metrics"] = nlos_result
            method_summary["nlos_threshold_used"] = float(threshold)

        forecast_sidecar = predictions_dir / "predicted_positions" / f"{method}.csv"
        if forecast_sidecar.exists():
            forecast_frame = pd.read_csv(forecast_sidecar)
            if "run_id" not in forecast_frame:
                raise ValueError(f"Forecast sidecar {forecast_sidecar} must contain run_id")
            forecast_frame = __import__("common").add_run_identity(forecast_frame)
            forecast_frame = filter_split(forecast_frame, args.split)
            forecast_join = forecast_error(forecast_frame, truth_subset)
            forecast_summary = []
            for horizon, group in forecast_join.groupby("horizon_steps", sort=True):
                valid_forecast = group["forecast_error_m"].dropna()
                row_summary = {"horizon_steps": int(horizon), **summarize_errors(valid_forecast)}
                forecast_summary.append(row_summary)
                forecast_rows.append({"method": method, **row_summary})
            method_summary["forecast_metrics_by_horizon"] = forecast_summary
            if not forecast_join.empty:
                forecast_join.insert(0, "method", method)
                forecast_join.to_csv(reports_dir / f"forecast_errors_{method}.csv", index=False)

        run_tracks = track_level_metrics(scored)
        run_tracks.insert(0, "method", method)
        tracks_frames.append(run_tracks)
        scored.insert(0, "method", method)
        step_frames.append(scored)
        slices.extend(_slice_report(scored, method))
        valid_err = scored["error_m"].dropna().to_numpy(dtype=float)
        cdf_frames.append((method, np.sort(valid_err)))
        summaries.append({"method": method, **{key: value for key, value in stats.items() if not isinstance(value, dict)}})
        all_summary["methods"][method] = method_summary

        if not calibration_bins.empty:
            calibration_bins.insert(0, "method", method)
            calibration_bins.to_csv(reports_dir / f"calibration_bins_{method}.csv", index=False)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(reports_dir / "metrics_summary.csv", index=False)
    all_summary["outputs"] = {
        "metrics_summary_csv": str(reports_dir / "metrics_summary.csv"),
        "metrics_summary_json": str(reports_dir / "metrics_summary.json"),
        "track_metrics_csv": str(reports_dir / "track_metrics.csv"),
        "robustness_slices_csv": str(reports_dir / "robustness_slices.csv"),
        "step_errors_csv": str(reports_dir / "step_errors.csv"),
        "plots_dir": str(plots_dir),
    }
    if forecast_rows:
        all_summary["outputs"]["forecast_metrics_csv"] = str(reports_dir / "forecast_metrics_by_horizon.csv")
        all_summary["outputs"]["forecast_plot"] = str(plots_dir / "forecast_rmse_by_horizon.png")
    write_json(reports_dir / "metrics_summary.json", json_safe(all_summary))
    if tracks_frames:
        pd.concat(tracks_frames, ignore_index=True).to_csv(reports_dir / "track_metrics.csv", index=False)
    if slices:
        pd.DataFrame(slices).to_csv(reports_dir / "robustness_slices.csv", index=False)
    if forecast_rows:
        pd.DataFrame(forecast_rows).to_csv(reports_dir / "forecast_metrics_by_horizon.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        forecast_frame = pd.DataFrame(forecast_rows)
        for method in args.methods:
            rows = forecast_frame.loc[forecast_frame["method"] == method].sort_values("horizon_steps")
            if not rows.empty:
                ax.plot(rows["horizon_steps"], rows["rmse_m"], marker="o", label=method)
        ax.set_xlabel("Forecast horizon (steps of 0.5 s)")
        ax.set_ylabel("Forecast RMSE (m)")
        ax.set_title("Forecast error growth by horizon")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "forecast_rmse_by_horizon.png", dpi=160)
        plt.close(fig)
    if step_frames:
        all_steps = pd.concat(step_frames, ignore_index=True)
        all_steps.to_csv(reports_dir / "step_errors.csv", index=False)
        selected_cols = [c for c in ["run_id", "venue_id", "split", "active_anchor_count", "packet_loss_rate_observed",
                                     "nlos_fraction_observed", "crowd_density_mean_people_m2", "noise_level", "noise_multiplier",
                                     "drift_condition", "outlier_condition", "dropout_duration_steps"] if c in all_steps.columns]
        run_slice = all_steps[selected_cols].drop_duplicates("run_id")
        run_slice.to_csv(reports_dir / "run_condition_features.csv", index=False)

    # Plots are generated only from the scored output files.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in args.methods:
        row = summary_frame.loc[summary_frame["method"] == method]
        if not row.empty and pd.notna(row.iloc[0].get("median_m")):
            ax.bar(method, float(row.iloc[0]["median_m"]))
    ax.set_ylabel("Median Euclidean position error (m)")
    ax.set_title(f"Positioning error by method — split: {args.split}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "median_error_by_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, values in cdf_frames:
        if values.size:
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, label=method)
    ax.set_xlabel("Euclidean position error (m)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title(f"Position-error CDF — split: {args.split}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "position_error_cdf.png", dpi=160)
    plt.close(fig)

    if slices:
        slice_frame = pd.DataFrame(slices)
        anchor_slices = slice_frame.loc[slice_frame["slice_dimension"] == "active_anchor_count"]
        if not anchor_slices.empty:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for method in args.methods:
                rows = anchor_slices.loc[anchor_slices["method"] == method].copy()
                if not rows.empty:
                    rows["sort_key"] = pd.to_numeric(rows["slice_value"], errors="coerce")
                    rows = rows.sort_values("sort_key")
                    ax.plot(rows["slice_value"], rows["p90_m"], marker="o", label=method)
            ax.set_xlabel("Active anchor count")
            ax.set_ylabel("P90 position error (m)")
            ax.set_title("P90 error versus infrastructure density")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / "p90_by_anchor_count.png", dpi=160)
            plt.close(fig)
    all_summary["outputs"] = {
        "metrics_summary_csv": str(reports_dir / "metrics_summary.csv"),
        "metrics_summary_json": str(reports_dir / "metrics_summary.json"),
        "track_metrics_csv": str(reports_dir / "track_metrics.csv"),
        "robustness_slices_csv": str(reports_dir / "robustness_slices.csv"),
        "step_errors_csv": str(reports_dir / "step_errors.csv"),
        "plots_dir": str(plots_dir),
    }
    if forecast_rows:
        all_summary["outputs"]["forecast_metrics_csv"] = str(reports_dir / "forecast_metrics_by_horizon.csv")
        all_summary["outputs"]["forecast_plot"] = str(plots_dir / "forecast_rmse_by_horizon.png")
    write_manifest(reports_dir / "manifest.json", {
        "script": "run_eval.py", "data_dir": str(data_dir), "predictions_dir": str(predictions_dir),
        "output_dir": str(output_dir), "methods": args.methods, "split": args.split,
        "nlos_threshold_argument": args.nlos_threshold, "sigma_bins": args.sigma_bins,
    }, extra={"truth_rows_scored": int(len(truth_subset)), "summary_file": str(reports_dir / "metrics_summary.json")})
    return json_safe(all_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing truth.csv, measurements.csv and optional sidecars")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing B1.csv, B2.csv, B3.csv and Traceline.csv")
    parser.add_argument("--output-dir", default="eval_output", help="Where reports and plots are written")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Prediction files to evaluate, without .csv")
    parser.add_argument("--split", choices=["train", "dev", "hidden", "all"], default="dev")
    parser.add_argument("--nlos-threshold", type=float, default=None, help="Frozen NLOS probability threshold; otherwise selected on dev")
    parser.add_argument("--sigma-bins", type=int, default=10)
    parser.add_argument("--confirm-hidden", action="store_true", help="Required explicit acknowledgement for one-time hidden evaluation")
    args = parser.parse_args()
    if args.sigma_bins < 1:
        parser.error("--sigma-bins must be >= 1")
    result = evaluate(args)
    compact = {"split": result["split"], "summary": {
        method: {key: metrics.get(key) for key in ("median_m", "rmse_m", "p90_m", "p95_m", "prediction_coverage", "n_missing_predictions")}
        for method, metrics in result["methods"].items()
    }, "reports": result["outputs"]}
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
