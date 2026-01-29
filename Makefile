.PHONY: all setup check lint typecheck test clean help

UV=uv
UVX=uvx
PYTEST=pytest
LINTER=ruff
TYPECHECKER=ty

all: clean setup check test

setup:
	@echo '=== Setup ==='
	$(UV) pip uninstall .
	$(UV) sync
	$(UV) pip install -e .

check: lint typecheck

lint:
	@echo '=== Linter ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checker ==='
	$(UVX) $(TYPECHECKER) check

test:
	@echo '=== Tests ==='
	$(UV) run $(PYTEST)

clean:
	@echo '=== Cleaning up ==='
	rm -rf **/__pycache__ **/*.pyc hyperbench.egg-info .pytest_cache .coverage

help:
    @echo "Usage: make [target]"
    @echo "Targets:"
	@echo "  all       - Setup, lint, typecheck, test"
	@echo "  setup     - Install dependencies"
	@echo "  lint      - Run linter"
	@echo "  typecheck - Run type checker"
	@echo "  test      - Run tests"
	@echo "  check     - Run lint and typecheck"
	@echo "  clean     - Remove build/test artifacts"
