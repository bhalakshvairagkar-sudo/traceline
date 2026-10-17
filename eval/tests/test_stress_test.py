from __future__ import annotations

import pandas as pd
import pytest

from stress_test import generate_conditions, validate_adapter_result


def test_stress_conditions_are_deterministic_unique_and_bounded() -> None:
    first = generate_conditions(5000, seed=4200)
    second = generate_conditions(5000, seed=4200)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5000
    assert first.drop(columns=["condition_id", "condition_seed"]).drop_duplicates().shape[0] == 5000
    assert first["active_anchor_count"].between(3, 8).all()
    assert first["crowd_density_people_m2"].between(0, 4).all()
    assert first["packet_loss_rate"].between(0, 0.5).all()


def test_stress_adapter_results_require_condition_method_p90() -> None:
    conditions = generate_conditions(2, 1)
    with pytest.raises(ValueError, match="missing columns"):
        validate_adapter_result(pd.DataFrame({"condition_id": [conditions.loc[0, "condition_id"]]}), conditions)


def test_stress_adapter_must_report_every_condition_for_all_baselines() -> None:
    conditions = generate_conditions(2, 5)
    rows = []
    for condition_id in conditions["condition_id"]:
        for method in ("B1", "B2", "B3", "Traceline"):
            rows.append({"condition_id": condition_id, "method": method, "p90_error_m": 1.0, "status": "completed"})
    complete = validate_adapter_result(pd.DataFrame(rows), conditions)
    assert len(complete) == 8
    with pytest.raises(ValueError, match="omitted condition/method pairs"):
        validate_adapter_result(pd.DataFrame(rows[:-1]), conditions)
