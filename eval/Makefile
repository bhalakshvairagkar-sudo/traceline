PYTHON ?= python
DATA_DIR ?= fixtures
PREDICTIONS_DIR ?= fixtures/predictions
OUTPUT_DIR ?= demo_outputs
VENUES_DIR ?= fixtures/venues
METHODS ?= B1 B2 B3 Traceline
HIGH_ERROR_THRESHOLD_M ?= 0.2
CELL_SIZE_M ?= 1.0
MIN_SUPPORT_RUNS ?= 3
STRESS_SEED ?= 4200
STRESS_CONDITIONS ?= 5000
STRESS_ADAPTER ?= default
FAILURE_P90_M ?= 3.0

.PHONY: setup test fixture eval errormap crowd emergency stress report clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -v

fixture:
	$(PYTHON) make_fixture.py --output-dir fixtures --seed 73

eval:
	$(PYTHON) run_eval.py --data-dir "$(DATA_DIR)" --predictions-dir "$(PREDICTIONS_DIR)" --output-dir "$(OUTPUT_DIR)/evaluation" --methods $(METHODS) --split dev

errormap:
	$(PYTHON) error_map.py --truth "$(DATA_DIR)/truth.csv" --predictions "$(PREDICTIONS_DIR)/Traceline.csv" --output-dir "$(OUTPUT_DIR)/error_map" --cell-size-m $(CELL_SIZE_M) --min-support-runs $(MIN_SUPPORT_RUNS) --train-prediction-provenance unknown

crowd:
	$(PYTHON) crowd_analysis.py --data-dir "$(DATA_DIR)" --predictions-dir "$(PREDICTIONS_DIR)" --venues-dir "$(VENUES_DIR)" --output-dir "$(OUTPUT_DIR)/crowd" --methods Traceline --split dev

emergency:
	$(PYTHON) emergency_flags.py --truth "$(DATA_DIR)/truth.csv" --predictions "$(PREDICTIONS_DIR)/Traceline.csv" --output-dir "$(OUTPUT_DIR)/emergency" --high-error-threshold-m $(HIGH_ERROR_THRESHOLD_M) --split dev

stress:
ifeq ($(strip $(STRESS_ADAPTER)),)
	$(PYTHON) stress_test.py --output-dir "$(OUTPUT_DIR)/stress" --n-conditions $(STRESS_CONDITIONS) --seed $(STRESS_SEED)
else
	$(PYTHON) stress_test.py --output-dir "$(OUTPUT_DIR)/stress" --n-conditions $(STRESS_CONDITIONS) --seed $(STRESS_SEED) --adapter "$(STRESS_ADAPTER)" --failure-p90-threshold-m $(FAILURE_P90_M)
endif

report: eval errormap crowd emergency stress
	@echo "Workstream C reports and plots generated successfully under $(OUTPUT_DIR)."

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['$(OUTPUT_DIR)', '.pytest_cache', '__pycache__', 'tests/__pycache__']]"
