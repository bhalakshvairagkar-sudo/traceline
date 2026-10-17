"""Generate reproducible stress conditions and evaluate them through a team-supplied adapter.

This module never fabricates model scores. Without --adapter it generates and manifests the
5,000+ unique scenarios, clearly marking them as not yet scored. An adapter must connect these
settings to the real simulator and predictor owned by Parts A/B.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .common import write_json, write_manifest, json_safe
except ImportError:  # pragma: no cover - standalone CLI
    from common import write_json, write_manifest, json_safe

CONDITION_COLUMNS = [
    "condition_id", "condition_seed", "active_anchor_count", "noise_multiplier",
    "nlos_rate", "nlos_bias_mean_m", "crowd_density_people_m2", "packet_loss_rate",
    "outage_steps", "drift_rate_m_per_s", "outlier_rate", "outlier_scale_m",
]


def generate_conditions(n_conditions: int = 5000, seed: int = 4200) -> pd.DataFrame:
    """Generate n unique, reproducible stress configurations from the documented stress factors."""
    if n_conditions < 1:
        raise ValueError("n_conditions must be positive")
    if n_conditions > 1_000_000:
        raise ValueError("Refusing to generate more than 1,000,000 conditions in one run")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    while len(records) < n_conditions:
        row = {
            "condition_seed": int(rng.integers(0, 2**31 - 1)),
            "active_anchor_count": int(rng.integers(3, 9)),
            "noise_multiplier": round(float(rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])), 3),
            "nlos_rate": round(float(rng.uniform(0.0, 0.8)), 3),
            "nlos_bias_mean_m": round(float(rng.uniform(0.5, 3.0)), 3),
            "crowd_density_people_m2": round(float(rng.uniform(0.0, 4.0)), 3),
            "packet_loss_rate": round(float(rng.uniform(0.0, 0.5)), 3),
            "outage_steps": int(rng.integers(0, 11)),
            "drift_rate_m_per_s": round(float(rng.uniform(0.0, 0.05)), 4),
            "outlier_rate": round(float(rng.uniform(0.0, 0.15)), 3),
            "outlier_scale_m": round(float(rng.uniform(0.5, 8.0)), 3),
        }
        fingerprint = tuple(row[key] for key in row if key != "condition_seed")
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        row["condition_id"] = f"stress-{len(records) + 1:06d}"
        records.append(row)
    return pd.DataFrame.from_records(records, columns=CONDITION_COLUMNS)


def _load_adapter(spec: str) -> Callable[..., pd.DataFrame]:
    """Load an adapter specified as module:function, or 'default'/'reference'."""
    if spec.lower() in {"default", "reference", "builtin"}:
        spec = "stress_adapter:evaluate_conditions"
    if ":" not in spec:
        raise ValueError("--adapter must use module:function syntax (or 'default')")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"Adapter {spec!r} is not a callable function")
    return function


def validate_adapter_result(result: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Validate actual adapter scores; every condition/method pair must be explicitly reported."""
    required = {"condition_id", "method", "p90_error_m"}
    if missing := required - set(result.columns):
        raise ValueError(f"Stress adapter results missing columns: {sorted(missing)}")
    if result.empty:
        raise ValueError("Stress adapter returned no scores")
    if result[["condition_id", "method"]].duplicated().any():
        raise ValueError("Stress adapter returned duplicate condition_id/method rows")
    unknown = set(result["condition_id"]) - set(conditions["condition_id"])
    if unknown:
        raise ValueError(f"Stress adapter returned unknown condition IDs: {sorted(unknown)[:5]}")
    scores = result.copy()
    if "status" not in scores:
        scores["status"] = "completed"
    scores["status"] = scores["status"].fillna("failed").astype(str).str.lower()
    supported_success = {"ok", "success", "completed"}
    scores["p90_error_m"] = pd.to_numeric(scores["p90_error_m"], errors="coerce")
    success = scores["status"].isin(supported_success)
    if scores.loc[success, "p90_error_m"].isna().any() or not np.isfinite(scores.loc[success, "p90_error_m"]).all():
        raise ValueError("Completed adapter rows must return finite p90_error_m")
    # Failures must be represented, never silently dropped; their P90 may be unavailable.
    if "failure_reason" not in scores:
        scores["failure_reason"] = ""
    if scores.loc[~success, "failure_reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Failed adapter rows must provide failure_reason")
    methods = sorted(scores["method"].astype(str).unique())
    expected_methods = {"B1", "B2", "B3", "Traceline"}
    if not expected_methods.issubset(set(methods)):
        raise ValueError(f"Stress adapter must include the complete baseline ladder {sorted(expected_methods)}; got {methods}")
    expected_pairs = {(str(condition_id), method) for condition_id in conditions["condition_id"] for method in methods}
    actual_pairs = set(zip(scores["condition_id"].astype(str), scores["method"].astype(str)))
    missing_pairs = expected_pairs - actual_pairs
    if missing_pairs:
        raise ValueError(f"Stress adapter omitted condition/method pairs (including explicit failed rows): {sorted(missing_pairs)[:5]}")
    if "prediction_coverage" in scores:
        scores["prediction_coverage"] = pd.to_numeric(scores["prediction_coverage"], errors="coerce")
        invalid = scores["prediction_coverage"].notna() & ((scores["prediction_coverage"] < 0) | (scores["prediction_coverage"] > 1))
        if invalid.any():
            raise ValueError("prediction_coverage must be in [0, 1]")
    return scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Create stress matrix and optionally execute a genuine simulator/model adapter."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = generate_conditions(args.n_conditions, args.seed)
    condition_path = output_dir / "stress_conditions.csv"
    conditions.to_csv(condition_path, index=False)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "conditions_requested": args.n_conditions,
        "conditions_generated": int(len(conditions)),
        "unique_conditions": int(conditions.drop(columns=["condition_id", "condition_seed"]).drop_duplicates().shape[0]),
        "factors": [column for column in CONDITION_COLUMNS if column not in {"condition_id", "condition_seed"}],
        "conditions_file": str(condition_path),
        "scored": False,
        "status": "conditions_generated_not_scored",
        "important_note": "No positioning scores are claimed unless a real simulator/predictor adapter is supplied.",
    }
    if args.adapter:
        if args.failure_p90_threshold_m is None or args.failure_p90_threshold_m <= 0:
            raise ValueError("Scored stress testing requires a predeclared positive --failure-p90-threshold-m")
        adapter = _load_adapter(args.adapter)
        result = adapter(conditions.copy(), seed=args.seed, output_dir=output_dir)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Stress adapter must return a pandas DataFrame")
        scores = validate_adapter_result(result, conditions)
        scores["failure_p90_threshold_m"] = float(args.failure_p90_threshold_m)
        scores["failed"] = (~scores["status"].isin({"ok", "success", "completed"})) | (scores["p90_error_m"] > args.failure_p90_threshold_m)
        scores.to_csv(output_dir / "stress_scores.csv", index=False)
        per_method = scores.groupby("method", as_index=False).agg(
            conditions_scored=("condition_id", "nunique"),
            failure_count=("failed", "sum"),
            failure_fraction=("failed", "mean"),
            median_p90_error_m=("p90_error_m", "median"),
            p90_of_p90_error_m=("p90_error_m", lambda v: float(np.quantile(v.dropna(), 0.90)) if len(v.dropna()) else None),
            worst_p90_error_m=("p90_error_m", "max"),
        )
        per_method.to_csv(output_dir / "stress_failure_summary.csv", index=False)
        conditions_with_scores = scores.merge(conditions, on="condition_id", how="left", validate="many_to_one")
        # Summarize failures at each stress factor value/range to identify boundaries.
        boundaries: list[dict[str, Any]] = []
        numeric_factors = [c for c in CONDITION_COLUMNS if c not in {"condition_id", "condition_seed"}]
        for method, subset in conditions_with_scores.groupby("method"):
            for factor in numeric_factors:
                if subset[factor].nunique() < 2:
                    continue
                # Quantile bins make continuous factors readable; integer/categorical counts remain discrete.
                if factor in {"active_anchor_count", "outage_steps"}:
                    groups = subset.groupby(factor, observed=True)
                else:
                    groups = subset.groupby(pd.qcut(subset[factor], q=min(8, subset[factor].nunique()), duplicates="drop"), observed=True)
                for label, part in groups:
                    boundaries.append({
                        "method": method, "factor": factor, "factor_bin": str(label),
                        "n_conditions": int(part["condition_id"].nunique()),
                        "failure_fraction": float(part["failed"].mean()),
                        "median_p90_error_m": float(part["p90_error_m"].median()),
                        "max_p90_error_m": float(part["p90_error_m"].max()),
                    })
        pd.DataFrame(boundaries).to_csv(output_dir / "stress_failure_boundaries.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for method, subset in conditions_with_scores.groupby("method"):
            ax.scatter(subset["noise_multiplier"], subset["p90_error_m"], s=9, alpha=0.45, label=method)
        ax.axhline(args.failure_p90_threshold_m, linestyle="--", label="Predeclared failure threshold")
        ax.set_xlabel("Range/RSSI noise multiplier")
        ax.set_ylabel("P90 position error (m)")
        ax.set_title("Stress-test response to noise")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "stress_noise_failure_plot.png", dpi=160)
        plt.close(fig)

        # Generate 2D failure boundary heatmaps across active_anchor_count vs noise_multiplier
        try:
            methods_list = sorted(scores["method"].unique().tolist())
            fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
            axes_flat = axes.flatten()
            im_last = None
            for idx, method in enumerate(methods_list[:4]):
                ax = axes_flat[idx]
                subset = conditions_with_scores.loc[conditions_with_scores["method"] == method]
                pivot = subset.pivot_table(
                    index="noise_multiplier",
                    columns="active_anchor_count",
                    values="failed",
                    aggfunc="mean",
                ).sort_index(ascending=False)
                im_last = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
                ax.set_title(f"{method} Failure Rate")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"{val:.2f}x" for val in pivot.index])
                if idx in (2, 3):
                    ax.set_xlabel("Active Anchor Count")
                if idx in (0, 2):
                    ax.set_ylabel("Noise Multiplier")
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if pd.notna(val):
                            color = "white" if val > 0.6 else "black"
                            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=8)
            if im_last is not None:
                cbar = fig.colorbar(im_last, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
                cbar.set_label(f"Failure Rate (P90 > {args.failure_p90_threshold_m} m)")
            fig.suptitle(f"Stress Test Failure Boundaries (P90 Threshold: {args.failure_p90_threshold_m} m)", fontsize=13)
            fig.savefig(output_dir / "stress_failure_heatmap.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        manifest.update({
            "scored": True,
            "status": "scored",
            "adapter": args.adapter,
            "failure_p90_threshold_m": float(args.failure_p90_threshold_m),
            "score_rows": int(len(scores)),
            "methods": sorted(scores["method"].unique().tolist()),
            "scores_file": str(output_dir / "stress_scores.csv"),
            "failure_summary_file": str(output_dir / "stress_failure_summary.csv"),
            "failure_boundaries_file": str(output_dir / "stress_failure_boundaries.csv"),
            "heatmap_file": str(output_dir / "stress_failure_heatmap.png"),
        })
    write_json(output_dir / "stress_manifest.json", json_safe(manifest))
    write_manifest(output_dir / "manifest.json", {
        "script": "stress_test.py", "seed": args.seed, "n_conditions": args.n_conditions,
        "adapter": args.adapter, "failure_p90_threshold_m": args.failure_p90_threshold_m,
    }, seed=args.seed, extra={"stress_status": manifest})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval_output/stress")
    parser.add_argument("--n-conditions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--adapter", default=None, help="Callable module:function that runs real simulator + predictor and returns scores")
    parser.add_argument("--failure-p90-threshold-m", type=float, default=None,
                        help="Predeclare the P90 error failure threshold before scoring; required with --adapter")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
