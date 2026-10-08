.PHONY: test test-all lint figures check-claims reproduce-figures replay-check pubmed-pool llm-up llm-down llm-check helper-bench

UV ?= uv run --quiet
OUT ?= outputs

test:
	$(UV) pytest -q

test-all:
	$(UV) pytest -q -m ""

lint:
	$(UV) ruff check src tests experiments serving/llm_check.py serving/helper_bench.py

# Figures, tables, and every number the paper quotes, from the shipped results (seconds).
figures:
	$(UV) python experiments/figures.py --out $(OUT)

# Compare the regenerated numbers with the ones the paper quotes.
check-claims:
	$(UV) python experiments/check_claims.py $(OUT)/claims.yaml reference/claims.yaml

reproduce-figures: figures check-claims

# Fetch the PubMed predicate pool (third-party, not redistributed). Needed before re-running
# experiments on PubMed.
pubmed-pool:
	$(UV) python -m semviews.workloads.datasets --pubmed-pool

# Replay the paper's operator on PubMed (E2, 20 workload orders) from the shipped tapes into
# experiments/results/rerun/ and compare it query by query with the shipped results. No model
# access needed. Every other experiment: docs/REPRODUCE.md, section 4. Never write reruns into
# experiments/results/ or results/op2/ under new names: figures.py reads every file there.
replay-check: configs/predicates/pubmed.yaml
	$(UV) python experiments/run_all.py --only E2 --outdir rerun --tag main --methods semviews semviews_noviews --datasets pubmed
	$(UV) python experiments/compare_replay.py experiments/results/rerun/E2__pubmed__main.parquet experiments/results/op2/E2__main.parquet

configs/predicates/pubmed.yaml:
	$(UV) python -m semviews.workloads.datasets --pubmed-pool

# Only for recording new tapes: local helper model and LiteLLM proxy (see docs/REPRODUCE.md).
llm-up:
	./serving/llm_up.sh

llm-down:
	./serving/llm_down.sh

llm-check:
	$(UV) python serving/llm_check.py

helper-bench:
	$(UV) python serving/helper_bench.py
