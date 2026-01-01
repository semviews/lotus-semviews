# semviews

This repository accompanies the paper *Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries* (see [Citation](#citation)). It contains the
full implementation, the recorded model outputs ("tapes") that every experiment replays, the
result files behind every figure and table, and the scripts that regenerate them.

`CertifiedFilter` answers a semantic filter (a natural-language predicate evaluated by an LLM
"oracle") while meeting user precision and recall targets with probability 1 - δ. It orders the
rows using signals it already has for free: the label columns that earlier filters left behind
(*semantic views*), a local sentence embedding, and a small helper model. It refines that order
with labels it has to pay for anyway, fixes a reject region and an accept region, and certifies
both regions with fresh oracle samples and exact finite-population (hypergeometric) bounds. Views
affect only cost, never validity: a stale or misleading view costs oracle calls, not accuracy.
The operator ships as a [LOTUS](https://github.com/lotus-data/lotus) optimizer extension and
needs no changes to LOTUS.

## Quick start

Requirements: macOS on Apple silicon (the lock file targets `darwin/arm64`), Python 3.12, and
[uv](https://docs.astral.sh/uv/). Replaying the experiments needs **no model access and no API
keys**. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for details.

```bash
uv sync
make test
```

**1. Regenerate every figure, table, and number in the paper from the shipped results**
(seconds):

```bash
make reproduce-figures
```

This writes `outputs/figures/*.pdf`, `outputs/tables/*.tex`, and `outputs/claims.yaml`. Each
number the paper quotes appears there with its source file. The target then compares them,
number by number, with `reference/claims.yaml`, the values printed in the paper.

**2. Re-run experiments by replaying the recorded tapes:**

```bash
make replay-check
```

This fetches the third-party PubMed predicate pool, which is not redistributed. It then replays
the paper's operator on the PubMed workloads into `experiments/results/rerun/` and checks every
query's oracle calls and outcome against the shipped results. The exact command for every other
experiment, and how to compare its output, is in
[docs/REPRODUCE.md](docs/REPRODUCE.md#4-level-b-re-run-the-experiments-from-the-tapes).

**3. Record new tapes with live models.** This needs a local helper model, a LiteLLM proxy, and
an OpenAI-compatible endpoint for the oracle models, and it costs money. See
[docs/REPRODUCE.md](docs/REPRODUCE.md#5-level-c-re-record-the-tapes-with-live-models).

## Use with LOTUS

```python
from lotus.ast import LazyFrame
from lotus.types import CascadeArgs
from semviews.catalog import Catalog
from semviews.adapters.lotus.optimizer import ViewOptimizer

catalog = Catalog("reviews", 0, path="reviews.semviews")   # DuckDB file beside the data
args = CascadeArgs(precision_target=0.9, recall_target=0.9, failure_probability=0.05)
pipeline = LazyFrame().sem_filter("{text} reports a hardware defect", cascade_args=args)
result = pipeline.optimize([ViewOptimizer(catalog)]).execute(reviews_df)
catalog.save()                                             # keep the views for later queries
```

The optimizer rewrites every `SemFilterNode`; prompts, models, and other nodes are unchanged.
Configure `lotus.settings.lm` (the oracle) and, optionally, `lotus.settings.helper_lm` as usual
in LOTUS.

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the system works: concepts, the `CertifiedFilter` algorithm step by step with pointers into the code, why validity holds regardless of view quality, configuration, the oracle/tape/cost design, the LOTUS integration, baselines, module map, extension points |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | How to repeat every experiment: setup, the three levels of reproduction, the command and output of each experiment, a map from each figure and table in the paper to its script and result files, seeds, re-recording tapes |
| [docs/DATA.md](docs/DATA.md) | Datasets and their sources, predicate pools and workloads, tapes and their schema, result-file schema, models and prices |
| [docs/AI_USE.md](docs/AI_USE.md) | Use of generative AI tools in writing the code and the paper (ACM policy disclosure) |

## Repository layout

```
src/semviews/
  oracle/          tape, replay oracle, cost meter (the only reader of oracle labels)
  certify/         Clopper-Pearson and hypergeometric bounds, decision procedures
  operators/       CertifiedFilter (region_filter.py, the paper's operator); ReuseFilter (strata variant)
  catalog/         semantic views with per-label provenance (oracle / derived), DuckDB persistence
  relate/          predicate and row embeddings
  baselines/       oracle-all, LOTUS cascade, BARGAIN, caches, proxies, inferred reuse, SemWeave reimpl.
  workloads/       dataset preparation, experiment runner, synthetic data
  adapters/lotus/  LOTUS optimizer, taped LM, tape builders, LLM judge, rewording generator
experiments/       run_all.py (main runner), figures.py (figures, tables, claims), other studies
  results/         per-query results (op2/ holds runs of the paper's operator)
configs/           frozen experiment configuration, model aliases, prices, predicate pools
data/              datasets (5,000 rows; two 100,000-row tables for E13), predicate relations, tapes
serving/           local helper server and LiteLLM proxy scripts (only for recording tapes)
reference/         claims.yaml: every number as printed in the paper
tests/             unit, property, coverage-simulation, import-boundary, and LOTUS contract tests
```

## Data and licences

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). The datasets in
`data/datasets/` are fixed samples of public datasets. Their sources and licences are listed in
[docs/DATA.md](docs/DATA.md), and they remain under their upstream terms. The PubMed predicate
pool comes from a third-party repository without a licence, so it is not included. `make
pubmed-pool` fetches it and checks its hash against the pool the experiments used.

## Citation

```bibtex
@unpublished{hassanzadeh2026semviews,
  title  = {Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries},
  author = {Hassanzadeh, Oktie and Subramanian, Dharmashankar},
  note   = {Under review},
  year   = {2026},
  url    = {https://github.com/semviews/lotus-semviews}
}
```
