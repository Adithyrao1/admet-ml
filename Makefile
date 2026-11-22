# Reproduce the whole project. On Windows without `make`, run the commands
# under each target directly -- they are plain Python.

PYTHON ?= python
RESULTS ?= results/results.csv

.PHONY: all install test experiments analyze explain app clean

all: experiments analyze explain

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

# The full headline sweep: 2 datasets x 4 models x 2 splits x 3 seeds.
# Resumable -- re-running skips finished runs.
experiments:
	$(PYTHON) scripts/run_experiment.py --all --results $(RESULTS)

analyze:
	$(PYTHON) scripts/analyze.py --results $(RESULTS)

explain:
	$(PYTHON) scripts/explain.py --dataset bbb_martins --results $(RESULTS)

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

clean:
	rm -rf data/features results/figures results/summary.csv results/tables.md
