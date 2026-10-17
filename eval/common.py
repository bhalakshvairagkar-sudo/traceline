"""Shared, contract-preserving utilities for Traceline Workstream C."""
from __future__ import annotations

import json
import re
import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

TRUTH_COLUMNS = {"run_id", "t", "x_m", "y_m"}
MEASUREMENT_COLUMNS = {"run_id", "t", "anchor_id", "range_m", "rssi_dbm"}
PREDICTION_COLUMNS = {"run_id", "t", "x_hat_m", "y_hat_m", "sigma_m", "flags"}
NLOS_LABEL_COLUMNS = {"run_id", "t", "anchor_id", "is_nlos", "bias_m"}
CROWD_COLUMNS = {"run_id", "t", "zone_id", "density_people_m2"}
RUN_ID_RE = re.compile(r"^(hospital|warehouse|office|clinic)-(train|dev|hidden)-(\d{4})$")


def read_contract_csv(path: str | Path, required: Iterable[str], name: str) -> pd.DataFrame:
    """Read a CSV and validate required contract columns without changing its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required {name} file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=True)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} contains no rows: {path}")
    return frame


def ensure_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    """Raise a useful error if a table is not unique on its contract key."""
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(5).to_dict("records")
        raise ValueError(f"{name} contains duplicate keys {keys}; examples: {sample}")


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Parse contract run IDs into venue, split and run index when possible."""
    match = RUN_ID_RE.match(str(run_id))
    if match:
        return {"venue_id": match.group(1), "split": match.group(2), "run_index": int(match.group(3))}
    pieces = str(run_id).split("-")
    return {
        "venue_id": pieces[0] if pieces else "unknown",
        "split": pieces[-2] if len(pieces) >= 3 and pieces[-2] in {"train", "dev", "hidden"} else "unknown",
        "run_index": int(pieces[-1]) if pieces and pieces[-1].isdigit() else None,
    }


def add_run_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived run identity columns; never mutates input."""
    result = frame.copy()
    if "run_id" not in result:
        raise ValueError("Expected run_id column")
    parsed = result["run_id"].astype(str).map(parse_run_id)
    result["venue_id"] = parsed.map(lambda row: row["venue_id"])
    result["split"] = parsed.map(lambda row: row["split"])
    result["run_index"] = parsed.map(lambda row: row["run_index"])
    return result


def finite_or_nan(series: pd.Series) -> pd.Series:
    """Convert numeric values and replace infinities with NaN."""
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def load_data_bundle(data_dir: str | Path) -> dict[str, pd.DataFrame | None]:
    """Load shared-contract inputs and optional evaluation sidecars."""
    root = Path(data_dir)
    truth = read_contract_csv(root / "truth.csv", TRUTH_COLUMNS, "truth.csv")
    measurements = read_contract_csv(root / "measurements.csv", MEASUREMENT_COLUMNS, "measurements.csv")
    ensure_unique(truth, ["run_id", "t"], "truth.csv")
    ensure_unique(measurements, ["run_id", "t", "anchor_id"], "measurements.csv")

    for col in ("t", "x_m", "y_m"):
        truth[col] = finite_or_nan(truth[col])
    if truth[["t", "x_m", "y_m"]].isna().any().any():
        raise ValueError("truth.csv contains non-numeric or non-finite time/position values")
    for col in ("t", "anchor_id", "range_m", "rssi_dbm"):
        measurements[col] = finite_or_nan(measurements[col])
    if measurements[["t", "anchor_id"]].isna().any().any():
        raise ValueError("measurements.csv contains invalid t or anchor_id")
    if (measurements["range_m"].dropna() < 0).any():
        raise ValueError("measurements.csv range_m must be non-negative when present")
    measurements["anchor_id"] = measurements["anchor_id"].astype(int)
    truth_runs = set(truth["run_id"].astype(str))
    measurement_runs = set(measurements["run_id"].astype(str))
    if truth_runs != measurement_runs:
        raise ValueError(
            "truth.csv and measurements.csv run IDs must match; "
            f"missing measurements={sorted(truth_runs - measurement_runs)[:5]}, "
            f"missing truth={sorted(measurement_runs - truth_runs)[:5]}"
        )

    labels_path = root / "labels_nlos.csv"
    labels = None
    if labels_path.exists():
        labels = read_contract_csv(labels_path, NLOS_LABEL_COLUMNS, "labels_nlos.csv")
        ensure_unique(labels, ["run_id", "t", "anchor_id"], "labels_nlos.csv")
        labels["is_nlos"] = pd.to_numeric(labels["is_nlos"], errors="coerce")
        if labels["is_nlos"].isna().any() or not labels["is_nlos"].isin([0, 1, True, False]).all():
            raise ValueError("labels_nlos.csv is_nlos must contain only 0/1 labels")
        labels["is_nlos"] = labels["is_nlos"].astype(int)

    crowd_path = root / "crowd.csv"
    crowd = None
    if crowd_path.exists():
        crowd = read_contract_csv(crowd_path, CROWD_COLUMNS, "crowd.csv")
        ensure_unique(crowd, ["run_id", "t", "zone_id"], "crowd.csv")
        crowd["density_people_m2"] = finite_or_nan(crowd["density_people_m2"])
        if crowd["density_people_m2"].isna().any() or (crowd["density_people_m2"] < 0).any():
            raise ValueError("crowd.csv density_people_m2 must be finite and non-negative")

    # Optional evaluation-only metadata sidecar. This does not alter any shared CSV contract.
    metadata_path = root / "run_metadata.csv"
    metadata = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, keep_default_na=True)
        if "run_id" not in metadata:
            raise ValueError("run_metadata.csv must contain run_id")
        ensure_unique(metadata, ["run_id"], "run_metadata.csv")

    return {"truth": truth, "measurements": measurements, "labels_nlos": labels, "crowd": crowd, "run_metadata": metadata}


def build_run_features(bundle: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Build per-run evaluation strata from measurements, labels and optional sidecars."""
    measurements = bundle["measurements"]
    assert isinstance(measurements, pd.DataFrame)
    rows = add_run_identity(pd.DataFrame({"run_id": sorted(measurements["run_id"].astype(str).unique())}))

    measured = measurements.copy()
    measured["range_missing"] = measured["range_m"].isna()
    measured["rssi_missing"] = measured["rssi_dbm"].isna()
    measured["both_missing"] = measured["range_missing"] & measured["rssi_missing"]
    measured["any_missing"] = measured["range_missing"] | measured["rssi_missing"]
    mfeat = measured.groupby("run_id", as_index=False).agg(
        active_anchor_count=("anchor_id", "nunique"),
        measurement_rows=("anchor_id", "size"),
        packet_loss_rate_observed=("both_missing", "mean"),
        any_field_missing_rate=("any_missing", "mean"),
    )
    rows = rows.merge(mfeat, on="run_id", how="left", validate="one_to_one")

    labels = bundle.get("labels_nlos")
    if isinstance(labels, pd.DataFrame):
        lfeat = labels.groupby("run_id", as_index=False).agg(nlos_fraction_observed=("is_nlos", "mean"))
        rows = rows.merge(lfeat, on="run_id", how="left", validate="one_to_one")

    crowd = bundle.get("crowd")
    if isinstance(crowd, pd.DataFrame):
        cfeat = crowd.groupby("run_id", as_index=False).agg(
            crowd_density_mean_people_m2=("density_people_m2", "mean"),
            crowd_density_max_people_m2=("density_people_m2", "max"),
        )
        rows = rows.merge(cfeat, on="run_id", how="left", validate="one_to_one")

    metadata = bundle.get("run_metadata")
    if isinstance(metadata, pd.DataFrame):
        overlap = (set(rows.columns) & set(metadata.columns)) - {"run_id"}
        if overlap:
            raise ValueError(f"run_metadata.csv may not overwrite derived columns: {sorted(overlap)}")
        rows = rows.merge(metadata, on="run_id", how="left", validate="one_to_one")
    return rows


def filter_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter rows to the requested contract split; all includes unrecognized IDs."""
    if split == "all":
        return frame.copy()
    if "split" not in frame:
        frame = add_run_identity(frame)
    return frame.loc[frame["split"] == split].copy()


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON deterministically and normalize NumPy scalar values."""
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=default)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-safe Python primitives, replacing NaN with null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _git_commit() -> str | None:
    """Return current Git commit when executed inside a Git working tree."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            return completed.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def write_manifest(path: str | Path, configuration: dict[str, Any], seed: int | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write reproducibility manifest with config hash, environment, Git commit and UTC date."""
    clean_config = json_safe(configuration)
    canonical = json.dumps(clean_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    package_names = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "pytest": "pytest"}
    versions: dict[str, str | None] = {}
    for label, package in package_names.items():
        try:
            versions[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "configuration": clean_config,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "git_commit": _git_commit(),
    }
    if extra:
        manifest.update(json_safe(extra))
    write_json(path, manifest)
    return manifest
