PYTHON ?= python3
PYTHONPATH := .

.PHONY: all install test reproduce figures paper clean help

help:
	@echo "Targets:"
	@echo "  install     -- pip install -e ."
	@echo "  test        -- run the unit-test suite (~7s)"
	@echo "  reproduce   -- run every paper experiment (~60s on CPU)"
	@echo "  figures     -- regenerate paper/figures/*.{png,pdf}"
	@echo "  paper       -- build paper/cctm.pdf"
	@echo "  all         -- reproduce + figures + paper"
	@echo "  clean       -- delete LaTeX build artefacts and __pycache__"

install:
	pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/

reproduce: results/synthetic_scm.json results/loan_recourse.json results/fairness_audit.json

results/synthetic_scm.json: experiments/synthetic_scm.py cctm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/synthetic_scm.py

results/loan_recourse.json: experiments/loan_recourse.py cctm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/loan_recourse.py

results/fairness_audit.json: experiments/fairness_audit.py cctm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/fairness_audit.py

figures: reproduce
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_figures.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode cctm.tex \
	  && bibtex cctm \
	  && pdflatex -interaction=nonstopmode cctm.tex \
	  && pdflatex -interaction=nonstopmode cctm.tex

all: paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out paper/*.toc
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
