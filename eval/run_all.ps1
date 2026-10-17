# Traceline Workstream C - Windows PowerShell Execution Script
param (
    [string]$Target = "report",
    [string]$DataDir = "fixtures",
    [string]$PredictionsDir = "fixtures/predictions",
    [string]$OutputDir = "demo_outputs",
    [string]$VenuesDir = "fixtures/venues",
    [string]$HighErrorThresholdM = "0.2",
    [string]$FailureP90M = "3.0"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Traceline Workstream C: $Target ===" -ForegroundColor Cyan

switch ($Target.ToLower()) {
    "test" {
        python -m pytest -v
    }
    "eval" {
        python run_eval.py --data-dir $DataDir --predictions-dir $PredictionsDir --output-dir "$OutputDir/evaluation" --methods B1 B2 B3 Traceline --split dev
    }
    "errormap" {
        python error_map.py --truth "$DataDir/truth.csv" --predictions "$PredictionsDir/Traceline.csv" --output-dir "$OutputDir/error_map" --cell-size-m 1.0 --min-support-runs 3 --train-prediction-provenance unknown
    }
    "crowd" {
        python crowd_analysis.py --data-dir $DataDir --predictions-dir $PredictionsDir --venues-dir $VenuesDir --output-dir "$OutputDir/crowd" --methods Traceline --split dev
    }
    "emergency" {
        python emergency_flags.py --truth "$DataDir/truth.csv" --predictions "$PredictionsDir/Traceline.csv" --output-dir "$OutputDir/emergency" --high-error-threshold-m $HighErrorThresholdM --split dev
    }
    "stress" {
        python stress_test.py --output-dir "$OutputDir/stress" --n-conditions 5000 --seed 4200 --adapter default --failure-p90-threshold-m $FailureP90M
    }
    "report" {
        Write-Host "1/5 Running Positioning Evaluation..." -ForegroundColor Green
        python run_eval.py --data-dir $DataDir --predictions-dir $PredictionsDir --output-dir "$OutputDir/evaluation" --methods B1 B2 B3 Traceline --split dev
        
        Write-Host "2/5 Running Self-Learning Error Map..." -ForegroundColor Green
        python error_map.py --truth "$DataDir/truth.csv" --predictions "$PredictionsDir/Traceline.csv" --output-dir "$OutputDir/error_map" --cell-size-m 1.0 --min-support-runs 3 --train-prediction-provenance unknown
        
        Write-Host "3/5 Running Crowd-Zone Degradation Analysis..." -ForegroundColor Green
        python crowd_analysis.py --data-dir $DataDir --predictions-dir $PredictionsDir --venues-dir $VenuesDir --output-dir "$OutputDir/crowd" --methods Traceline --split dev
        
        Write-Host "4/5 Running Emergency Uncertainty Flags..." -ForegroundColor Green
        python emergency_flags.py --truth "$DataDir/truth.csv" --predictions "$PredictionsDir/Traceline.csv" --output-dir "$OutputDir/emergency" --high-error-threshold-m $HighErrorThresholdM --split dev
        
        Write-Host "5/5 Running 5,000-Condition Stress Matrix & Heatmap..." -ForegroundColor Green
        python stress_test.py --output-dir "$OutputDir/stress" --n-conditions 5000 --seed 4200 --adapter default --failure-p90-threshold-m $FailureP90M

        Write-Host "`nAll Workstream C reports, heatmaps, and validation outputs generated successfully in $OutputDir!" -ForegroundColor Green
    }
    "clean" {
        python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['$OutputDir', '.pytest_cache', '__pycache__', 'tests/__pycache__']]"
        Write-Host "Cleaned temporary files and outputs." -ForegroundColor Yellow
    }
    default {
        Write-Host "Unknown target: $Target. Supported: test, eval, errormap, crowd, emergency, stress, report, clean" -ForegroundColor Red
    }
}
