COLOR ?= auto
PYTEST_FLAGS ?= -v
PYTEST_PARALLEL ?= auto # overwritten in CI

.PHONY: setup
setup:
	pip install -r requirements/python-test.txt
	pre-commit install

.PHONY: lint
lint: format
	mypy modules

.PHONY: format
format:
	pre-commit run --all-files --show-diff-on-failure
