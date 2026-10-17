from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from make_fixture import create_fixture
from run_eval import evaluate


def test_run_eval_generates_fair_reports(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    create_fixture(fixture, seed=17)
    output = tmp_path / "out"
    result = evaluate(Namespace(
        data_dir=str(fixture), predictions_dir=str(fixture / "predictions"), output_dir=str(output),
        methods=["B1", "B2", "B3", "Traceline"], split="dev", nlos_threshold=None, sigma_bins=5,
        confirm_hidden=False,
    ))
    assert set(result["methods"]) == {"B1", "B2", "B3", "Traceline"}
    assert (output / "reports" / "metrics_summary.csv").exists()
    assert (output / "reports" / "step_errors.csv").exists()
    assert (output / "plots" / "position_error_cdf.png").exists()
    assert result["methods"]["Traceline"]["n_truth_steps"] == 960
