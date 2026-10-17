"""Tests for crowd analysis and emergency flags (Workstream C Module 4)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crowd_analysis import (
    add_crowd_membership, load_crowd_polygons, point_in_polygon,
    paired_comparison, summarize_crowd,
)
from emergency_flags import (
    choose_uncertainty_threshold, evaluate_flags, _split_dev_runs,
)


def test_point_in_polygon_inside_outside_and_boundary() -> None:
    polygon = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    # Strictly inside
    assert point_in_polygon(1.0, 1.0, polygon)
    # Strictly outside
    assert not point_in_polygon(3.0, 1.0, polygon)
    assert not point_in_polygon(-0.5, 1.0, polygon)
    # Boundary vertices and edges
    assert point_in_polygon(0.0, 1.0, polygon)
    assert point_in_polygon(1.0, 0.0, polygon)
    assert point_in_polygon(0.0, 0.0, polygon)
    assert point_in_polygon(2.0, 2.0, polygon)


def test_load_crowd_polygons_from_venue_json(tmp_path: Path) -> None:
    venues_dir = tmp_path / "venues"
    venues_dir.mkdir()
    venue_data = {
        "schema_version": "1.0",
        "venue_id": "hospital",
        "width_m": 40.0,
        "height_m": 30.0,
        "zones": [
            {
                "zone_id": "corridor_1",
                "zone_type": "crowd",
                "polygon": [[5.0, 5.0], [15.0, 5.0], [15.0, 10.0], [5.0, 10.0]]
            },
            {
                "zone_id": "storage_1",
                "zone_type": "restricted",
                "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
            }
        ]
    }
    with open(venues_dir / "hospital.json", "w", encoding="utf-8") as f:
        json.dump(venue_data, f)

    polygons = load_crowd_polygons(venues_dir)
    assert "hospital" in polygons
    assert len(polygons["hospital"]) == 1
    assert polygons["hospital"][0] == [[5.0, 5.0], [15.0, 5.0], [15.0, 10.0], [5.0, 10.0]]


def test_add_crowd_membership_and_summarize() -> None:
    polygons = {
        "hospital": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    }
    scored = pd.DataFrame({
        "run_id": ["hospital-dev-0001", "hospital-dev-0001"],
        "t": [0.0, 0.5],
        "x_m": [5.0, 15.0],
        "y_m": [5.0, 15.0],
        "x_hat_m": [5.2, 15.8],
        "y_hat_m": [5.1, 15.6],
        "error_m": [0.2236, 1.0],
        "sigma_m": [0.3, 0.9],
        "has_prediction": [True, True],
        "method": ["Traceline", "Traceline"],
    })
    labeled = add_crowd_membership(scored, polygons)
    assert bool(labeled.loc[0, "inside_crowd_zone"]) is True
    assert bool(labeled.loc[1, "inside_crowd_zone"]) is False

    summary = summarize_crowd(labeled)
    assert len(summary) == 2
    inside_row = summary.loc[summary["crowd_zone"] == "inside"].iloc[0]
    outside_row = summary.loc[summary["crowd_zone"] == "outside"].iloc[0]
    assert inside_row["n_steps"] == 1
    assert outside_row["n_steps"] == 1
    assert inside_row["mean_error_m"] == pytest.approx(0.2236, abs=1e-3)


def test_paired_comparison_crowd_ablation() -> None:
    scored = pd.DataFrame({
        "run_id": ["hospital-dev-0001", "hospital-dev-0001"],
        "t": [0.0, 0.0],
        "venue_id": ["hospital", "hospital"],
        "inside_crowd_zone": [True, True],
        "method": ["Traceline", "Traceline_no_crowd"],
        "error_m": [0.4, 0.8],
    })
    paired = paired_comparison(scored, with_crowd="Traceline", without_crowd="Traceline_no_crowd")
    assert len(paired) == 1
    # Error with crowd is 0.4, without crowd is 0.8 => benefit is +0.4m
    assert paired.loc[0, "delta_error_m"] == pytest.approx(-0.4)
    assert paired.loc[0, "crowd_input_benefit_m"] == pytest.approx(0.4)


def test_emergency_threshold_is_selected_on_supplied_calibration_rows() -> None:
    rows = pd.DataFrame({
        "run_id": ["hospital-dev-0001"] * 4,
        "t": [0.0, 0.5, 1.0, 1.5],
        "error_m": [0.1, 0.2, 0.9, 1.1],
        "sigma_m": [0.1, 0.2, 0.9, 1.2],
    })
    threshold, report = choose_uncertainty_threshold(rows, high_error_threshold_m=0.8)
    assert threshold == pytest.approx(0.9)
    assert report["best_calibration_metrics"]["f1"] == 1.0
    scored, metrics = evaluate_flags(rows, high_error_threshold_m=0.8, sigma_threshold_m=threshold)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert scored["uncertainty_flagged"].sum() == 2


def test_split_dev_runs_leakage_free() -> None:
    run_ids = [f"hospital-dev-{i:04d}" for i in range(1, 11)]
    calib, eval_ids = _split_dev_runs(run_ids, selection_fraction=0.5, seed=123)
    assert len(calib) == 5
    assert len(eval_ids) == 5
    # Strict disjointness
    assert len(calib & eval_ids) == 0
    # Deterministic across multiple calls with same seed
    calib2, eval2 = _split_dev_runs(run_ids, selection_fraction=0.5, seed=123)
    assert calib == calib2
    assert eval_ids == eval2
