"""Create a small deterministic synthetic fixture for smoke tests only (not performance evidence)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from .common import write_manifest
except ImportError:  # pragma: no cover - standalone CLI
    from common import write_manifest

METHOD_BIAS = {
    "B1": (0.62, -0.28, 0.18),
    "B2": (0.42, -0.18, 0.14),
    "B3": (0.28, -0.11, 0.11),
    "Traceline": (0.15, -0.06, 0.08),
}
ANCHORS = [(0.0, 0.0), (7.0, 0.0), (7.0, 5.0), (0.0, 5.0)]


def create_fixture(root: Path, seed: int = 73) -> None:
    """Write contract-shaped data with known repeatable biases for code-path validation only."""
    rng = np.random.default_rng(seed)
    rows_truth, rows_measurements, rows_nlos, rows_crowd = [], [], [], []
    run_metadata = []
    method_predictions = {method: [] for method in METHOD_BIAS}
    nlos_probs = {method: [] for method in METHOD_BIAS}
    specs = [("train", i) for i in range(1, 5)] + [("dev", i) for i in range(1, 5)]
    n_steps = 240  # 120 s at the frozen 0.5 s time step: contract-compliant smoke trajectories.
    for split, index in specs:
        run_id = f"hospital-{split}-{index:04d}"
        phase = 0.01 * index
        noise_multiplier = 1.0 if index % 2 else 1.5
        run_metadata.append({
            "run_id": run_id,
            "noise_level": "moderate" if index % 2 else "high",
            "noise_multiplier": noise_multiplier,
            "drift_condition": "present" if index == 4 else "absent",
            "drift_rate_m_per_s": 0.005 if index == 4 else 0.0,
            "outlier_condition": "present" if index == 3 else "absent",
            "dropout_duration_steps": 2 if index == 2 else 0,
            "scenario_id": f"fixture-{split}-{index:04d}",
        })
        speed_m_s = 1.0 + 0.02 * index
        perimeter_m = 14.0
        for step in range(n_steps):
            t = step * 0.5
            # Repeat a rectangular walkable loop at a plausible walking speed; turning happens at corners.
            distance_along = (step * 0.5 * speed_m_s) % perimeter_m
            if distance_along < 4.0:
                x, y = 1.0 + distance_along, 1.0
            elif distance_along < 7.0:
                x, y = 5.0, 1.0 + (distance_along - 4.0)
            elif distance_along < 11.0:
                x, y = 5.0 - (distance_along - 7.0), 4.0
            else:
                x, y = 1.0, 4.0 - (distance_along - 11.0)
            y += 0.015 * index
            rows_truth.append({"run_id": run_id, "t": t, "x_m": x, "y_m": y})
            for anchor_id, (ax, ay) in enumerate(ANCHORS):
                true_range = float(np.hypot(x - ax, y - ay))
                is_nlos = int(anchor_id == 2 and 16 <= step <= 19 or (index % 2 == 0 and anchor_id == 1 and step in (96, 97)))
                bias = float(rng.uniform(0.5, 3.0)) if is_nlos else 0.0
                drift_offset = 0.005 * max(0.0, t - 20.0) if index == 4 and anchor_id == 0 else 0.0
                range_noise = (0.15 + 0.02 * true_range) * noise_multiplier
                observed_range = max(0.0, true_range + rng.normal(0, range_noise) + bias + drift_offset)
                rssi = -40.0 - 20.0 * np.log10(max(true_range, 0.5)) + rng.normal(0, 2.5 * noise_multiplier)
                if index == 3 and anchor_id == 1 and step in (70, 71):
                    observed_range += 4.0  # labelled in run_metadata as an injected fixture outlier
                    rssi += 10.0
                missing = bool(rng.random() < 0.05 or (anchor_id == 3 and index == 2 and step in (80, 81)))
                rows_measurements.append({
                    "run_id": run_id, "t": t, "anchor_id": anchor_id,
                    "range_m": np.nan if missing else round(observed_range, 3),
                    "rssi_dbm": np.nan if missing else round(rssi, 1),
                })
                rows_nlos.append({"run_id": run_id, "t": t, "anchor_id": anchor_id, "is_nlos": is_nlos, "bias_m": round(bias, 3)})
                for method, (bx, by, sigma) in METHOD_BIAS.items():
                    nlos_probs[method].append({
                        "run_id": run_id, "t": t, "anchor_id": anchor_id,
                        "nlos_probability": float(np.clip(0.80 if is_nlos else 0.12, 0, 1)),
                    })
            # The density stream is an evaluation side input; geometry membership comes from the venue JSON.
            rows_crowd.append({"run_id": run_id, "t": t, "zone_id": "crowd-a", "density_people_m2": 2.5 if 3.0 <= x <= 4.7 else 0.2})
            for method, (bx, by, sigma) in METHOD_BIAS.items():
                nx, ny = rng.normal(0, sigma * 0.20, size=2)
                method_predictions[method].append({
                    "run_id": run_id, "t": t,
                    "x_hat_m": round(x + bx + nx, 4),
                    "y_hat_m": round(y + by + ny, 4),
                    "sigma_m": round(max(0.02, sigma + rng.normal(0, sigma * 0.08)), 4),
                    "flags": "[]",
                })

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_truth).to_csv(root / "truth.csv", index=False)
    pd.DataFrame(rows_measurements).to_csv(root / "measurements.csv", index=False)
    pd.DataFrame(rows_nlos).to_csv(root / "labels_nlos.csv", index=False)
    pd.DataFrame(rows_crowd).to_csv(root / "crowd.csv", index=False)
    pd.DataFrame(run_metadata).to_csv(root / "run_metadata.csv", index=False)
    pred_dir = root / "predictions"
    nlos_dir = pred_dir / "nlos_probabilities"
    nlos_dir.mkdir(parents=True, exist_ok=True)
    for method, records in method_predictions.items():
        pd.DataFrame(records).to_csv(pred_dir / f"{method}.csv", index=False)
        pd.DataFrame(nlos_probs[method]).to_csv(nlos_dir / f"{method}.csv", index=False)
    venues = root / "venues"
    venues.mkdir(exist_ok=True)
    venue_data = {
        "schema_version": "1.0", "venue_id": "hospital", "width_m": 7.0, "height_m": 5.0,
        "anchors": [{"anchor_id": i, "x_m": x, "y_m": y, "enabled": True, "anchor_type": "fixture"} for i, (x, y) in enumerate(ANCHORS)],
        "walls": [], "obstacles": [],
        "zones": [{"zone_id": "crowd-a", "zone_type": "crowd", "polygon": [[3.0, 0.5], [5.0, 0.5], [5.0, 4.8], [3.0, 4.8]], "name": "Fixture crowd region"}],
        "metadata": {"synthetic_fixture_only": True},
    }
    with (venues / "hospital.json").open("w", encoding="utf-8") as handle:
        json.dump(venue_data, handle, indent=2)
    write_manifest(root / "manifest.json", {
        "script": "make_fixture.py", "seed": seed, "purpose": "synthetic smoke test only",
        "n_train_runs": 4, "n_dev_runs": 4, "steps_per_run": n_steps,
        "methods": sorted(METHOD_BIAS),
    }, seed=seed, extra={"not_performance_evidence": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="fixtures")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    create_fixture(Path(args.output_dir), args.seed)
    print(f"Created synthetic smoke-test fixture at {args.output_dir}; do not use as benchmark evidence.")


if __name__ == "__main__":
    main()
