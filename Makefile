PY ?= python3
EXP = experiments

# Set CSA6G_OUT to write a run into its own folder, for example:
#     CSA6G_OUT=my_run make all   -> my_run/{results,figures}

.PHONY: all quick test clean

all:            ## full-scale run of every experiment
	$(PY) $(EXP)/run_all.py

quick:          ## reduced-scale smoke run of every experiment
	$(PY) $(EXP)/run_all.py --quick

test:           ## run the test suite
	$(PY) -m pytest tests -q

clean:
	rm -rf results figures tables
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
