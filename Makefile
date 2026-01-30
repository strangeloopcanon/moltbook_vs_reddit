.PHONY: setup format check test all

export PYTHONPATH := src

setup:
	uv sync --dev

format:
	uv run ruff format .

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run python -m compileall -q src tests

test:
	uv run pytest -q

all: check test
