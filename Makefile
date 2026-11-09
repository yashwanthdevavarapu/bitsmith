.PHONY: install lint format typecheck test test-fast coverage check reproduce reproduce-gpu figures clean

PY ?= python
SRC = src tests examples scripts

install:
	$(PY) -m pip install -e ".[dev,plot]"

lint:
	ruff check $(SRC)
	ruff format --check $(SRC)

format:
	ruff check --fix $(SRC)
	ruff format $(SRC)

typecheck:
	mypy

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

coverage:
	pytest -q --cov --cov-report=term --cov-report=html

check: lint typecheck test

reproduce:
	$(PY) scripts/reproduce.py

reproduce-gpu:
	$(PY) scripts/reproduce.py --gpu

figures:
	$(PY) scripts/make_figures.py

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
