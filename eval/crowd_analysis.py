"""Measure crowd-zone degradation and compare model variants with/without crowd input."""
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
    from .common import PREDICTION_COLUMNS, add_run_identity, load_data_bundle, read_contract_csv, write_json, write_manifest, json_safe
    from .metrics import position_errors, summarize_errors
except ImportError:  # pragma: no cover - standalone CLI
    from common import PREDICTION_COLUMNS, add_run_identity, load_data_bundle, read_contract_csv, write_json, write_manifest, json_safe
    from metrics import position_errors, summarize_errors


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Return whether a point is inside a polygon using ray crossing; boundary is treated as inside."""
    if len(polygon) < 3:
        return False
    inside = False
    eps = 1e-9
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
        if abs(cross) <= eps and dot <= eps:
            return True
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x <= x_intersect:
                inside = not inside
    return inside


def load_crowd_polygons(venues_dir: str | Path) -> dict[str, list[list[list[float]]]]:
    """Read crowd-zone polygons from contract venue JSON files."""
    venues: dict[str, list[list[list[float]]]] = {}
    for path in sorted(Path(venues_dir).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            venue = json.load(handle)
        venue_id = venue.get("venue_id", path.stem)
        polygons = []
        for zone in venue.get("zones", []):
            if zone.get("zone_type") == "crowd" and isinstance(zone.get("polygon"), list):
                # Contract polygon is [x_m, y_m] coordinate pairs.
                polygons.append([[float(point[0]), float(point[1])] for point in zone["polygon"]])
        venues[venue_id] = polygons
    return venues


def add_crowd_membership(scored: pd.DataFrame, polygons_by_venue: dict[str, list[list[list[float]]]]) -> pd.DataFrame:
    """Label each ground-truth position according to venue crowd polygons."""
    result = add_run_identity(scored)
    membership: list[bool | None] = []
    for row in result.itertuples(index=False):
        polygons = polygons_by_venue.get(str(row.venue_id))
        if polygons is None:
            membership.append(None)
            continue
        membership.append(any(point_in_polygon(float(row.x_m), float(row.y_m), poly) for poly in polygons))
    result["inside_crowd_zone"] = pd.array(membership, dtype="boolean")
    return result


def summarize_crowd(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize position error and uncertainty inside/outside crowd zones by method and venue."""
    rows: list[dict[str, Any]] = []
    data = scored.loc[scored["inside_crowd_zone"].notna()].copy()
    for (method, venue, inside), group in data.groupby(["method", "venue_id", "inside_crowd_zone"], observed=True):
        err = summarize_errors(group["error_m"])
        sigma = pd.to_numeric(group["sigma_m"], errors="coerce").dropna()
        rows.append({
            "method": method, "venue_id": venue,
            "crowd_zone": "inside" if bool(inside) else "outside",
            "n_steps": int(len(group)), "n_tracks": int(group["run_id"].nunique()),
            "mean_error_m": err["mean_m"], "median_error_m": err["median_m"],
            "rmse_error_m": err["rmse_m"], "p90_error_m": err["p90_m"],
            "mean_sigma_m": float(sigma.mean()) if len(sigma) else None,
            "median_sigma_m": float(sigma.median()) if len(sigma) else None,
            "prediction_coverage": float(group["has_prediction"].mean()) if len(group) else None,
        })
    return pd.DataFrame(rows)


def paired_comparison(scored: pd.DataFrame, with_crowd: str, without_crowd: str) -> pd.DataFrame:
    """Compare matched step errors for explicit with-crowd and no-crowd prediction variants."""
    data = scored.loc[scored["method"].isin([with_crowd, without_crowd])].copy()
    pivot = data.pivot_table(index=["run_id", "t", "venue_id", "inside_crowd_zone"], columns="method",
                             values="error_m", aggfunc="first")
    if with_crowd not in pivot.columns or without_crowd not in pivot.columns:
        raise ValueError(f"Need prediction rows for both method variants: {with_crowd!r} and {without_crowd!r}")
    paired = pivot.dropna(subset=[with_crowd, without_crowd]).reset_index()
    paired["error_with_crowd_m"] = paired[with_crowd]
    paired["error_without_crowd_m"] = paired[without_crowd]
    paired["delta_error_m"] = paired["error_with_crowd_m"] - paired["error_without_crowd_m"]
    paired["crowd_input_benefit_m"] = -paired["delta_error_m"]  # positive means with-crowd error is lower
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing contract CSV inputs")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing method prediction CSVs")
    parser.add_argument("--venues-dir", required=True, help="Directory shared/venues with venue JSON files")
    parser.add_argument("--output-dir", default="eval_output/crowd")
    parser.add_argument("--methods", nargs="+", default=["Traceline"])
    parser.add_argument("--split", choices=["train", "dev", "hidden", "all"], default="dev")
    parser.add_argument("--with-crowd-method", default=None, help="Method/file name trained with crowd input")
    parser.add_argument("--without-crowd-method", default=None, help="Ablation method/file name without crowd input")
    parser.add_argument("--confirm-hidden", action="store_true")
    args = parser.parse_args()
    if args.split == "hidden" and not args.confirm_hidden:
        raise ValueError("Hidden evaluation requires --confirm-hidden and must run only after freeze")
    bundle = load_data_bundle(args.data_dir)
    truth = bundle["truth"]
    assert isinstance(truth, pd.DataFrame)
    truth_identified = add_run_identity(truth)
    truth = truth_identified.loc[truth_identified["split"] == args.split].copy() if args.split != "all" else truth_identified.copy()
    polygons = load_crowd_polygons(args.venues_dir)
    all_rows: list[pd.DataFrame] = []
    for method in args.methods:
        pred = read_contract_csv(Path(args.predictions_dir) / f"{method}.csv", PREDICTION_COLUMNS, f"predictions for {method}")
        pred = add_run_identity(pred)
        if args.split != "all":
            pred = pred.loc[pred["split"] == args.split]
        scored = position_errors(pred[["run_id", "t", "x_hat_m", "y_hat_m", "sigma_m", "flags"]],
                                 truth[["run_id", "t", "x_m", "y_m"]])
        scored = add_run_identity(scored)
        scored["method"] = method
        all_rows.append(add_crowd_membership(scored, polygons))
    combined = pd.concat(all_rows, ignore_index=True)
    summary = summarize_crowd(combined)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "crowd_zone_metrics.csv", index=False)
    payload: dict[str, Any] = {"split": args.split, "methods": args.methods,
                               "rows_with_crowd_zone_geometry": int(combined["inside_crowd_zone"].notna().sum()),
                               "summary": summary.to_dict("records"),
                               "interpretation": "Crowd membership uses ground-truth position and contract venue polygons; error is still computed from model predictions."}
    if args.with_crowd_method and args.without_crowd_method:
        # Ensure both named variants were scored even if only those two were not in `--methods`.
        if not {args.with_crowd_method, args.without_crowd_method}.issubset(set(args.methods)):
            raise ValueError("Both crowd input variants must also appear in --methods")
        paired = paired_comparison(combined, args.with_crowd_method, args.without_crowd_method)
        paired.to_csv(out / "crowd_input_paired_comparison.csv", index=False)
        paired_summary = []
        for inside, group in paired.groupby("inside_crowd_zone", dropna=False):
            delta = group["delta_error_m"].dropna()
            paired_summary.append({"crowd_zone": "inside" if bool(inside) else "outside", "n_paired_steps": int(len(group)),
                                   "mean_error_delta_with_minus_without_m": float(delta.mean()) if len(delta) else None,
                                   "median_error_delta_with_minus_without_m": float(delta.median()) if len(delta) else None,
                                   "fraction_with_crowd_improves": float((delta < 0).mean()) if len(delta) else None})
        payload["paired_crowd_ablation"] = paired_summary
    write_json(out / "crowd_analysis.json", json_safe(payload))
    write_manifest(out / "manifest.json", {
        "script": "crowd_analysis.py", "data_dir": args.data_dir, "predictions_dir": args.predictions_dir,
        "venues_dir": args.venues_dir, "methods": args.methods, "split": args.split,
        "with_crowd_method": args.with_crowd_method, "without_crowd_method": args.without_crowd_method,
    }, extra={"rows_with_geometry": int(combined["inside_crowd_zone"].notna().sum())})

    plot_data = summary.groupby(["method", "crowd_zone"], as_index=False)["p90_error_m"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{row.method}\n{row.crowd_zone}" for row in plot_data.itertuples(index=False)]
    values = plot_data["p90_error_m"].fillna(0).tolist()
    ax.bar(labels, values)
    ax.set_ylabel("P90 position error (m)")
    ax.set_title("Positioning degradation inside versus outside crowd zones")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "crowd_zone_p90.png", dpi=160)
    plt.close(fig)
    print(f"Wrote crowd analysis to {out}")


if __name__ == "__main__":
    main()
