from __future__ import annotations

import pandas as pd
import pytest

from error_map import apply_error_map, fit_error_map, validate_error_map


def _scored(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    # Each tuple is run_id, x_hat, y_hat, dx, dy with valid prediction.
    records = []
    for run_id, xhat, yhat, dx, dy in rows:
        records.append({"run_id": run_id, "venue_id": "hospital", "x_hat_m": xhat, "y_hat_m": yhat,
                        "dx_m": dx, "dy_m": dy, "has_prediction": True})
    return pd.DataFrame(records)


def test_fit_map_uses_distinct_run_support_not_point_count() -> None:
    scored = _scored([
        ("hospital-train-0001", 1.2, 1.2, -0.2, 0.1),
        ("hospital-train-0001", 1.3, 1.3, -0.2, 0.1),
        ("hospital-train-0002", 1.4, 1.4, -0.3, 0.1),
        ("hospital-train-0003", 1.5, 1.5, -0.1, 0.2),
    ])
    error_map = fit_error_map(scored, cell_size_m=1.0, min_support_runs=3)
    cell = error_map.iloc[0]
    assert cell["support_count"] == 3
    assert cell["observation_count"] == 4
    assert cell["correction_applied_by_default"]


def test_apply_error_map_leaves_unsupported_cells_unchanged() -> None:
    error_map = pd.DataFrame({
        "venue_id": ["hospital"], "cell_x": [1], "cell_y": [1], "support_count": [2],
        "mean_dx_m": [-0.3], "mean_dy_m": [0.2],
    })
    pred = pd.DataFrame({"run_id": ["hospital-dev-0001"], "t": [0.0], "x_hat_m": [1.5], "y_hat_m": [1.5],
                         "sigma_m": [0.2], "flags": ["[]"]})
    corrected = apply_error_map(pred, error_map, cell_size_m=1.0, min_support_runs=3)
    assert corrected.loc[0, "x_hat_corrected_m"] == pytest.approx(1.5)
    assert not bool(corrected.loc[0, "error_map_correction_applied"])


def test_validation_rejects_trajectory_leakage() -> None:
    train = pd.DataFrame({"run_id": ["hospital-train-0001"], "t": [0.0], "x_hat_m": [1.0], "y_hat_m": [1.0], "sigma_m": [0.1], "flags": ["[]"]})
    dev = train.copy()
    truth = pd.DataFrame({"run_id": ["hospital-train-0001"], "t": [0.0], "x_m": [0.0], "y_m": [0.0]})
    with pytest.raises(ValueError, match="Trajectory leakage"):
        validate_error_map(train, dev, truth)
