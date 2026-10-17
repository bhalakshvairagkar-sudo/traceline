"""Select and validate uncertainty-based emergency flags using development trajectories."""
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

try:
    from .common import PREDICTION_COLUMNS, add_run_identity, ensure_unique, read_contract_csv, write_json, write_manifest, json_safe
    from .metrics import binary_classification_metrics, position_errors
except ImportError:  # pragma: no cover - standalone CLI
    from common import PREDICTION_COLUMNS, add_run_identity, ensure_unique, read_contract_csv, write_json, write_manifest, json_safe
    from metrics import binary_classification_metrics, position_errors


def choose_uncertainty_threshold(calibration_rows: pd.DataFrame, high_error_threshold_m: float) -> tuple[float, dict[str, Any]]:
    """Choose uncertainty threshold by maximum F1 on calibration runs only.

    `sigma_m` is used as an uncertainty score, not assumed to be a probability. The high-error
    outcome is an explicitly supplied operational threshold, not a universal safety limit.
    """
    data = calibration_rows.loc[
        calibration_rows["error_m"].notna() & calibration_rows["sigma_m"].notna() & (calibration_rows["sigma_m"] >= 0)
    ].copy()
    if data.empty:
        raise ValueError("No development calibration rows have both a finite error and a valid sigma_m")
    actual = data["error_m"] >= high_error_threshold_m
    class_degenerate = bool(actual.nunique() < 2)
    candidates = np.unique(np.r_[0.0, data["sigma_m"].quantile(np.linspace(0, 1, min(101, max(2, data["sigma_m"].nunique())))).to_numpy(float),
                                 float(data["sigma_m"].max()) + 1e-12])
    results = []
    for threshold in candidates:
        metrics = binary_classification_metrics(actual, data["sigma_m"] >= threshold)
        results.append({"threshold_m": float(threshold), **metrics})
    # Optimize F1; ties prefer higher precision, then the more conservative (higher) threshold.
    best = max(results, key=lambda row: (row["f1"], row["precision"], row["threshold_m"]))
    return float(best["threshold_m"]), {"best_calibration_metrics": best, "threshold_sweep": results,
                                         "selection_rule": "maximum F1; ties prefer precision then higher sigma threshold",
                                         "actual_high_error_prevalence": float(actual.mean()),
                                         "class_degenerate": class_degenerate,
                                         "calibration_warning": ("Calibration subset contains only one actual-error class; the selected threshold is not meaningfully calibrated. "
                                                                 "Use more diverse development trajectories or review the predeclared high-error definition." if class_degenerate else None)}


def evaluate_flags(rows: pd.DataFrame, high_error_threshold_m: float, sigma_threshold_m: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply fixed uncertainty threshold and score its precision/recall on held-out runs."""
    result = rows.copy()
    valid = result["sigma_m"].notna() & (result["sigma_m"] >= 0) & result["error_m"].notna()
    result["actual_high_error"] = result["error_m"] >= high_error_threshold_m
    result["uncertainty_flagged"] = False
    result.loc[valid, "uncertainty_flagged"] = result.loc[valid, "sigma_m"] >= sigma_threshold_m
    result["flag_reason"] = np.where(result["uncertainty_flagged"], "uncertainty_above_dev_threshold", "below_dev_threshold_or_unavailable")
    metrics = binary_classification_metrics(result.loc[valid, "actual_high_error"], result.loc[valid, "uncertainty_flagged"])
    metrics.update({
        "sigma_threshold_m": float(sigma_threshold_m),
        "high_error_threshold_m": float(high_error_threshold_m),
        "n_truth_steps": int(len(result)),
        "n_valid_scored_steps": int(valid.sum()),
        "uncertainty_and_error_coverage": float(valid.mean()) if len(result) else 0.0,
        "flagged_steps": int(result.loc[valid, "uncertainty_flagged"].sum()),
        "warning": "This is a simulated uncertainty-prioritization evaluation, not a certified life-safety decision rule.",
    })
    return result, metrics


def _split_dev_runs(run_ids: list[str], selection_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministically split whole development trajectories into calibration/evaluation groups."""
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be strictly between 0 and 1")
    unique_ids = sorted(set(run_ids))
    if len(unique_ids) < 2:
        raise ValueError("At least two development run IDs are needed for run-separated threshold selection and evaluation")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(unique_ids, dtype=object)[rng.permutation(len(unique_ids))])
    n_selection = min(len(unique_ids) - 1, max(1, int(round(len(unique_ids) * selection_fraction))))
    return set(str(x) for x in shuffled[:n_selection]), set(str(x) for x in shuffled[n_selection:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Path to contract truth.csv")
    parser.add_argument("--predictions", required=True, help="Path to canonical predictions.csv for one method")
    parser.add_argument("--output-dir", default="eval_output/emergency")
    parser.add_argument("--high-error-threshold-m", type=float, required=True,
                        help="Predeclared position error (m) that defines an actual high-error step")
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=["dev", "hidden"], default="dev")
    parser.add_argument("--fixed-uncertainty-threshold-m", type=float, default=None,
                        help="Use a threshold frozen on dev, only for final hidden evaluation")
    parser.add_argument("--confirm-hidden", action="store_true")
    args = parser.parse_args()
    if args.high_error_threshold_m <= 0:
        raise ValueError("high_error_threshold_m must be positive and predeclared")
    if args.split == "hidden" and (not args.confirm_hidden or args.fixed_uncertainty_threshold_m is None):
        raise ValueError("Hidden evaluation requires --confirm-hidden and a frozen --fixed-uncertainty-threshold-m")

    truth = read_contract_csv(args.truth, {"run_id", "t", "x_m", "y_m"}, "truth.csv")
    predictions = read_contract_csv(args.predictions, PREDICTION_COLUMNS, "predictions.csv")
    ensure_unique(predictions, ["run_id", "t"], "predictions")
    identified = add_run_identity(predictions)
    truth_id = add_run_identity(truth)
    predictions = identified.loc[identified["split"] == args.split, list(PREDICTION_COLUMNS)].copy()
    truth = truth_id.loc[truth_id["split"] == args.split, ["run_id", "t", "x_m", "y_m"]].copy()
    if truth.empty:
        raise ValueError(f"No truth data found for split {args.split!r}")
    rows = position_errors(predictions, truth)
    rows = add_run_identity(rows)
    rows["sigma_m"] = pd.to_numeric(rows["sigma_m"], errors="coerce")

    if args.split == "dev":
        selection_ids, evaluation_ids = _split_dev_runs(rows["run_id"].unique().tolist(), args.selection_fraction, args.seed)
        calibration_rows = rows.loc[rows["run_id"].isin(selection_ids)].copy()
        evaluation_rows = rows.loc[rows["run_id"].isin(evaluation_ids)].copy()
        threshold, selection_report = choose_uncertainty_threshold(calibration_rows, args.high_error_threshold_m)
        evaluated, evaluation_metrics = evaluate_flags(evaluation_rows, args.high_error_threshold_m, threshold)
        evaluation_metrics.update({"evaluation_run_count": int(len(evaluation_ids)), "calibration_run_count": int(len(selection_ids)),
                                   "calibration_run_ids": sorted(selection_ids), "evaluation_run_ids": sorted(evaluation_ids)})
        payload = {"split": "dev", "selected_sigma_threshold_m": threshold,
                   "calibration": selection_report, "heldout_development_evaluation": evaluation_metrics,
                   "settings": {"seed": args.seed, "selection_fraction": args.selection_fraction,
                                "high_error_threshold_m": args.high_error_threshold_m}}
    else:
        threshold = float(args.fixed_uncertainty_threshold_m)
        evaluated, evaluation_metrics = evaluate_flags(rows, args.high_error_threshold_m, threshold)
        payload = {"split": "hidden", "selected_sigma_threshold_m": threshold,
                   "heldout_hidden_evaluation": evaluation_metrics,
                   "settings": {"high_error_threshold_m": args.high_error_threshold_m,
                                "note": "Threshold was supplied frozen from development data; no hidden-set threshold selection occurred."}}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(out / "emergency_flag_rows.csv", index=False)
    write_json(out / "emergency_flag_metrics.json", json_safe(payload))
    if args.split == "dev":
        write_json(out / "frozen_emergency_threshold.json", {
            "sigma_threshold_m": threshold,
            "high_error_threshold_m": args.high_error_threshold_m,
            "selected_on_split": "dev",
            "selection_seed": args.seed,
            "selection_run_ids": sorted(selection_ids),
            "purpose": "Reuse unchanged for final hidden evaluation after full model/settings freeze.",
        })
        sweep = pd.DataFrame(selection_report["threshold_sweep"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep["threshold_m"], sweep["precision"], label="Precision")
        ax.plot(sweep["threshold_m"], sweep["recall"], label="Recall")
        ax.plot(sweep["threshold_m"], sweep["f1"], label="F1")
        ax.axvline(threshold, linestyle="--", label="Selected threshold")
        ax.set_xlabel("Uncertainty threshold sigma_m (m)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Emergency-flag threshold selection on calibration development runs")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "emergency_threshold_sweep.png", dpi=160)
        plt.close(fig)
    write_manifest(out / "manifest.json", {
        "script": "emergency_flags.py", "truth_path": args.truth, "predictions_path": args.predictions,
        "high_error_threshold_m": args.high_error_threshold_m, "selection_fraction": args.selection_fraction,
        "seed": args.seed, "split": args.split,
        "fixed_uncertainty_threshold_m": args.fixed_uncertainty_threshold_m,
    }, seed=args.seed if args.split == "dev" else None, extra={"selected_sigma_threshold_m": threshold})
    print(f"Wrote emergency-flag outputs to {out}")
    metrics_key = "heldout_development_evaluation" if args.split == "dev" else "heldout_hidden_evaluation"
    print(json.dumps({"selected_sigma_threshold_m": threshold, "evaluation": payload[metrics_key],
                      "metrics_file": str(out / "emergency_flag_metrics.json")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
