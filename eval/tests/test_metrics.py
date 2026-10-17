from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from metrics import (
    binary_classification_metrics, nlos_metrics, position_errors,
    summarize_errors, track_level_metrics, uncertainty_calibration,
)


def test_summarize_errors_matches_known_values() -> None:
    result = summarize_errors([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean_m"] == pytest.approx(2.0)
    assert result["median_m"] == pytest.approx(2.0)
    assert result["rmse_m"] == pytest.approx(math.sqrt(14 / 3))
    assert result["p90_m"] == pytest.approx(2.8)
    assert result["p95_m"] == pytest.approx(2.9)


def test_summary_empty_is_json_safe() -> None:
    result = summarize_errors([np.nan, np.inf])
    assert result["n"] == 0
    assert result["median_m"] is None


def test_binary_metrics_confusion_and_zero_division() -> None:
    result = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    no_positives = binary_classification_metrics([0, 0], [0, 0])
    assert no_positives["precision"] == 0
    assert no_positives["recall"] == 0
    assert no_positives["f1"] == 0


def test_position_errors_counts_missing_predictions_in_coverage() -> None:
    truth = pd.DataFrame({"run_id": ["hospital-dev-0001"] * 2, "t": [0.0, 0.5], "x_m": [0.0, 1.0], "y_m": [0.0, 0.0]})
    predictions = pd.DataFrame({
        "run_id": ["hospital-dev-0001"], "t": [0.0], "x_hat_m": [3.0], "y_hat_m": [4.0],
        "sigma_m": [1.0], "flags": ["[]"],
    })
    scored = position_errors(predictions, truth)
    assert len(scored) == 2
    assert scored.loc[0, "error_m"] == pytest.approx(5.0)
    assert pd.isna(scored.loc[1, "error_m"])
    tracks = track_level_metrics(scored)
    assert tracks.loc[0, "prediction_coverage"] == pytest.approx(0.5)


def test_nlos_metrics_aligns_anchor_and_time() -> None:
    labels = pd.DataFrame({"run_id": ["hospital-dev-0001"] * 2, "t": [0.0, 0.0], "anchor_id": [0, 1], "is_nlos": [1, 0]})
    probs = pd.DataFrame({"run_id": ["hospital-dev-0001"] * 2, "t": [0.0, 0.0], "anchor_id": [0, 1], "nlos_probability": [0.9, 0.1]})
    result = nlos_metrics(labels, probs, threshold=0.5)
    assert result["n"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_uncertainty_calibration_does_not_assume_semantics() -> None:
    data = pd.DataFrame({"error_m": [1.0, 2.0, 3.0], "sigma_m": [1.0, 1.5, 2.0]})
    report, bins = uncertainty_calibration(data, n_bins=3)
    assert report["n"] == 3
    assert "nominal probability coverage is not asserted" in report["interpretation_note"]
    assert not bins.empty


def test_forecast_error_respects_half_second_horizon() -> None:
    from metrics import forecast_error
    truth = pd.DataFrame({
        "run_id": ["hospital-dev-0001", "hospital-dev-0001"],
        "t": [0.0, 0.5], "x_m": [0.0, 1.0], "y_m": [0.0, 0.0],
    })
    forecasts = pd.DataFrame({
        "run_id": ["hospital-dev-0001"], "t": [0.0], "horizon_steps": [1],
        "x_hat_m": [0.9], "y_hat_m": [0.0],
    })
    result = forecast_error(forecasts, truth)
    assert result.loc[0, "target_t"] == pytest.approx(0.5)
    assert result.loc[0, "forecast_error_m"] == pytest.approx(0.1)


def test_emergency_flag_precision_metrics() -> None:
    from metrics import emergency_flag_precision
    actual = np.array([True, True, False, False])
    flagged = np.array([True, False, True, False])
    metrics = emergency_flag_precision(actual, flagged)
    assert metrics["flag_precision"] == pytest.approx(0.5)
    assert metrics["flag_recall"] == pytest.approx(0.5)
    assert metrics["flag_f1"] == pytest.approx(0.5)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1


def test_position_errors_rejects_duplicate_truth_rows() -> None:
    truth = pd.DataFrame({
        "run_id": ["hospital-dev-0001", "hospital-dev-0001"],
        "t": [0.0, 0.0],
        "x_m": [1.0, 2.0],
        "y_m": [1.0, 2.0],
    })
    pred = pd.DataFrame({
        "run_id": ["hospital-dev-0001"],
        "t": [0.0],
        "x_hat_m": [1.0],
        "y_hat_m": [1.0],
        "sigma_m": [0.1],
        "flags": ["[]"],
    })
    with pytest.raises(ValueError, match="duplicate keys"):
        position_errors(pred, truth)


def test_position_errors_rejects_duplicate_prediction_rows() -> None:
    truth = pd.DataFrame({
        "run_id": ["hospital-dev-0001"],
        "t": [0.0],
        "x_m": [1.0],
        "y_m": [1.0],
    })
    pred = pd.DataFrame({
        "run_id": ["hospital-dev-0001", "hospital-dev-0001"],
        "t": [0.0, 0.0],
        "x_hat_m": [1.0, 1.1],
        "y_hat_m": [1.0, 1.1],
        "sigma_m": [0.1, 0.2],
        "flags": ["[]", "[]"],
    })
    with pytest.raises(ValueError, match="duplicate keys"):
        position_errors(pred, truth)


def test_position_errors_rejects_non_finite_truth() -> None:
    truth = pd.DataFrame({
        "run_id": ["hospital-dev-0001"],
        "t": [0.0],
        "x_m": [np.inf],
        "y_m": [1.0],
    })
    pred = pd.DataFrame({
        "run_id": ["hospital-dev-0001"],
        "t": [0.0],
        "x_hat_m": [1.0],
        "y_hat_m": [1.0],
        "sigma_m": [0.1],
        "flags": ["[]"],
    })
    with pytest.raises(ValueError, match="non-numeric/non-finite"):
        position_errors(pred, truth)
