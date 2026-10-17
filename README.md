# Traceline — Indoor Positioning & Evaluation Engine

> **Track 4 · Problem Statement 2 · Technology for Social Cause**  
> AI/ML-based indoor positioning algorithm estimating 2D positions of moving targets from limited reference points, robust to noise, packet loss, and NLOS errors.

## Repository Architecture

Traceline is structured into four independent, decoupled workstreams:

```text
traceline/
├── eval/       # Workstream C: Evaluation, Error Learning and Stress Testing (OWNED)
├── sim/        # Workstream A: Simulation & Data Engine (traceline_sim + JS port)
├── model/      # Workstream B: Positioning AI, baselines (B1/B2/B3), ONNX export
├── planner/    # Workstream D: Blueprint planner & Anchor Placement Optimizer
├── demo/       # Workstream D: Interactive browser demo application
├── shared/     # Shared contracts, schemas, constants, versions.json (frozen)
└── datasets/   # Reproducible synthetic evaluation fixtures & datasets
```

---

## Workstream C: Evaluation, Error Learning & Stress Testing

Workstream C proves when Traceline's positioning model works, when it fails, and whether its corrections genuinely improve accuracy on unseen trajectories.

### Core Modules in `eval/`

1. **Evaluation Engine (`metrics.py`, `run_eval.py`):**
   - Scores B1 (Weighted Least-Squares), B2 (Residual Rejection), B3 (Extended Kalman Filter), and Traceline neural baseline on identical runs.
   - Point & track-level metrics (mean, median, RMSE, P90, P95, coverage).
   - NLOS classification metrics (precision, recall, F1, confusion matrix).
   - Empirical uncertainty calibration diagnostics across $\sigma$ bins.
   - Robustness slices by venue, active anchor count, noise multiplier, NLOS, packet loss, crowd density, drift, and outages.

2. **Self-Learning Error Map (`error_map.py`):**
   - Divides walkable area into spatial cells using predicted coordinates.
   - Estimates recurring location-specific residual corrections on training trajectories only.
   - Enforces distinct-trajectory support thresholds ($\ge 3$ runs) and zero trajectory leakage.
   - Evaluates generalization on separate held-out development trajectories.

3. **Automatic Stress Testing (`stress_test.py`, `stress_adapter.py`):**
   - Generates 5,000 unique, reproducible seeded parameter combinations across 10 factors.
   - Pre-declared failure threshold ($P_{90} > \text{threshold}$).
   - Integrated reference simulation adapter (`stress_adapter.py`) scoring 20,000 method evaluations in $<2$ seconds.
   - Generates failure boundary summaries and 2D failure rate heatmaps.

4. **Crowd & Emergency Analytics (`crowd_analysis.py`, `emergency_flags.py`):**
   - Polygon ray-casting analysis of error and uncertainty inside vs outside crowd zones.
   - Paired model ablation measuring crowd input benefit (`crowd_input_benefit_m`).
   - Uncertainty flag threshold selection on development calibration runs and precision/recall evaluation on held-out development runs.

---

## Quick Start

### 1. Requirements & Setup

```bash
cd eval
python -m pip install -r requirements.txt
```

### 2. Run Test Suite (26 Automated Tests)

```bash
# On Linux / macOS:
make test

# On Windows PowerShell:
.\run_all.ps1 -Target test
# Or:
python -m pytest -v
```

### 3. Generate Complete Evaluation Report & Visualizations

```bash
# On Linux / macOS:
make report

# On Windows PowerShell:
.\run_all.ps1 -Target report
```

For complete documentation on data contracts, sidecars, and integration guidelines, see [`eval/README.md`](eval/README.md) and [`eval/VALIDATION.md`](eval/VALIDATION.md).
