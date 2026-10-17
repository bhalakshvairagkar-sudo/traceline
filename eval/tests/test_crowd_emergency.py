from __future__ import annotations

import pandas as pd
import pytest

from crowd_analysis import point_in_polygon
from emergency_flags import choose_uncertainty_threshold, evaluate_flags


def test_point_in_polygon_inside_outside_and_boundary() -> None:
    polygon = [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert point_in_polygon(1, 1, polygon)
    assert not point_in_polygon(3, 1, polygon)
    assert point_in_polygon(0, 1, polygon)


def test_emergency_threshold_is_selected_on_supplied_calibration_rows() -> None:
    rows = pd.DataFrame({"error_m": [0.1, 0.2, 0.9, 1.1], "sigma_m": [0.1, 0.2, 0.9, 1.2]})
    threshold, report = choose_uncertainty_threshold(rows, high_error_threshold_m=0.8)
    assert threshold == pytest.approx(0.9)
    assert report["best_calibration_metrics"]["f1"] == 1.0
    scored, metrics = evaluate_flags(rows, high_error_threshold_m=0.8, sigma_threshold_m=threshold)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert scored["uncertainty_flagged"].sum() == 2
