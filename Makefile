# Modern Physical AI Engineer — companion repo tasks.
# The offline core targets (test, demo) need no GPU and no API key.

.PHONY: help test demo capstone lint install install-all

help:
	@echo "make install      - install the offline core (editable, dev extras)"
	@echo "make install-all  - install everything incl. sim/vla/api extras (GPU recommended)"
	@echo "make test         - run the offline test suite (no GPU, no key)"
	@echo "make demo         - score the oracle vs a random baseline, in/out-of-distribution"
	@echo "make capstone     - run the end-to-end capstone on the toy env (added in cycle 3)"
	@echo "make lint         - ruff check"

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	python -m pytest -q

demo:
	python tools/demo.py

capstone:
	python -m capstone.run "put the red block on the bowl"

lint:
	ruff check physicalai tests tools
