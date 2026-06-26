# Modern Physical AI Engineer — companion repo tasks.
# The offline core targets (test, demo) need no GPU and no API key.

.PHONY: help test demo ch16 ch20 ch36 capstone lint install install-all

help:
	@echo "make install      - install the offline core (editable, dev extras)"
	@echo "make install-all  - install everything incl. sim/vla/api extras (GPU recommended)"
	@echo "make test         - run the offline test suite (no GPU, no key)"
	@echo "make demo         - score the oracle vs a random baseline, in/out-of-distribution"
	@echo "make ch16         - imitation: behavior cloning + the covariate-shift gap"
	@echo "make ch20         - RL: tabular Q-learning control on GridWorld"
	@echo "make ch36         - VLA: a VLM as a high-level policy (offline-safe, key-optional)"
	@echo "make capstone     - run the end-to-end capstone on the toy env"
	@echo "make lint         - ruff check"

ch16:
	python chapters/ch16-imitation/reproduce.py

ch20:
	python chapters/ch20-rl/reproduce.py

ch36:
	python chapters/ch36-vla-finetune/reproduce.py

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
