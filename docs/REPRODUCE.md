# Reproducing the experiments

This guide explains how to repeat every experiment in the paper. There are three levels of
reproduction. Each one goes further back toward the raw model calls:

| Level | What you do | Needs | Time |
| --- | --- | --- | --- |
| **A** | Regenerate every figure, table, and quoted number from the shipped result files | `uv sync` | under a minute |
| **B** | Re-run the experiments by replaying the shipped tapes (recorded oracle labels and helper scores) | `uv sync`; network access once to fetch the PubMed predicate pool and the sentence encoder | minutes to hours per experiment, on CPU |
| **C** | Re-record the tapes from scratch with live models | a local llama.cpp server, a LiteLLM proxy, an OpenAI-compatible hosted endpoint of your choice, and an API budget | many hours, plus API cost |

Levels A and B need no model access and no API keys. Every experiment in the paper replays the
tapes, so oracle-call counts are exact and repeatable. The only exception is the live LOTUS check
(E8), which has its own recorded tape.

Related documents: [ARCHITECTURE.md](ARCHITECTURE.md) describes the code, and [DATA.md](DATA.md)
describes the datasets, predicate pools, and tapes.

---

## 1. Requirements, installation, and tests

**Platform.** The code was developed and run only on macOS on Apple silicon (arm64).
`pyproject.toml` declares
`required-environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'"]`, so the lock
file is guaranteed to resolve for that platform. Other platforms are untested. For level C, the
helper start script `serving/llm_up.sh` passes `--device MTL0` (Apple Metal) to `llama-server`.
Change that flag if you use other hardware.

**Software.**

- [uv](https://docs.astral.sh/uv/). It installs the pinned Python and all dependencies from
  `uv.lock`.
- Python 3.12 (`requires-python = ">=3.12,<3.13"`). uv fetches it if it is missing.
- Git and network access during `uv sync`. One dependency, BARGAIN, is installed from its public
  Git repository at a pinned commit (`[tool.uv.sources]` in `pyproject.toml`).
- Pinned key versions: `lotus-ai==1.2.4` and `litellm==1.103.2`.
- No LaTeX toolchain is needed. Figures are written as PDF by matplotlib, and tables as LaTeX
  fragments.

**Install.**

```bash
uv sync            # creates .venv with the runtime and dev dependencies (pytest, ruff, matplotlib)
```

**Tests.**

```bash
make test          # uv run pytest -q ; default addopts deselect tests marked `live`
make test-all      # uv run pytest -q -m "" ; also runs `live` tests
make lint          # ruff
```

The test suite contains:

- coverage simulations for every confidence bound (`tests/test_certify.py`);
- validity and leakage tests of both operators on synthetic tapes (`tests/test_region_filter.py`,
  `tests/test_reuse_filter.py`);
- a boundary test that the core packages do not import LOTUS (`tests/test_boundaries.py`);
- LOTUS contract tests that use an offline fake LM (`tests/test_lotus_contract.py`).

`pyproject.toml` defines two markers: `live` (needs the local LiteLLM proxy and model access) and
`slow` (long simulation tests). No current test carries either marker, so `make test` and
`make test-all` run the same tests.

---

## 2. What the release contains

| Path | Contents |
| --- | --- |
| `configs/experiments/sprint.yaml` | **The frozen experiment configuration**: targets (γ_P = γ_R = 0.9, δ = 0.05), the paper's operator settings (`operator:`), datasets and workloads, method lists, `orders`, `validity_seeds`, sensitivity grids |
| `configs/models.yaml` | Model aliases (`oracle-llama`, `oracle-gptoss`, `judge`, `helper`), sampling parameters, helper GGUF path, embedding model |
| `configs/budgets.yaml` | Spending cap and per-alias token prices used by the tape recorder |
| `configs/predicates/*.yaml` | Predicate pools. `pubmed.yaml` is **not shipped** (see §4.1). |
| `data/datasets/*.parquet` | The four 5,000-row samples (`row_id`, `text`, human labels where available) |
| `data/tapes/<dataset>__<alias>[__rw].parquet` | Recorded oracle labels and helper scores, one row per (tuple, predicate) |
| `data/tapes/judge.sqlite` | Recorded judge responses (relation judgments and rewordings) |
| `data/tapes/e8_live.sqlite` | Recorded responses of the live LOTUS run (E8) |
| `data/relations/*.json` | Pairwise predicate relations from the LLM judge (used by SemWeave and inferred reuse) |
| `experiments/results/` | Baseline runs and runs of the **first-draft operator** |
| `experiments/results/op2/` | Runs of the **paper's operator** |
| `reference/claims.yaml` | Frozen copy of every number the paper quotes |

Tape sizes:

| Tape | Rows |
| --- | --- |
| `oracle-llama`, four pools | 725,000 (40 × 5,000 for GoEmotions, DBpedia, and Reviews; 25 × 5,000 for PubMed) |
| `oracle-llama__rw`, reworded predicates | 150,000 (15 × 5,000 each for Reviews and GoEmotions) |
| `oracle-gptoss` | 40,000 (40 predicates × the first 1,000 GoEmotions tuples) |
| `helper` / `helper__rw` | One score per oracle row of the same pool |

### `results/` and `results/op2/`

The paper's operator, `semviews` in the result files, is the configuration under `operator:` in
`configs/experiments/sprint.yaml`. It uses refit rounds, sampling without replacement with
hypergeometric bounds, a planning margin, and a δ split. The runner `src/semviews/workloads/runner.py`
applies these settings to every `semviews*` method.

- **`experiments/results/`** holds the baselines, which are deterministic replays that do not
  depend on the operator. It also holds earlier runs of the operator in its first-draft
  configuration (no refit rounds, Clopper–Pearson bounds with replacement). When
  `experiments/figures.py` loads these files, it renames their operator rows: `semviews` becomes
  `semviews_v1` (the "first draft" row of the ablation table), and every other `semviews_*` except
  `semviews_strata` becomes `v1/semviews_*`.
- **`experiments/results/op2/`** holds all runs of the paper's operator. Files here are read as
  they are.

`figures.py` loads an experiment `X` as the concatenation of `results/X.parquet`,
`results/X__*.parquet`, `results/op2/X.parquet`, and `results/op2/X__*.parquet`. Any extra file
whose name matches these patterns is therefore counted too. See the warnings in §4.2.

---

## 3. Level A: regenerate figures, tables, and claims

```bash
make figures        # runs experiments/figures.py with outputs/ as the output directory
make check-claims   # compares outputs/claims.yaml with reference/claims.yaml
```

`make figures` is equivalent to:

```bash
uv run python experiments/figures.py --out outputs
```

It writes:

- `outputs/figures/`: `e2_cumulative.pdf`, `e2_breakdown.pdf`, `e3_tradeoff.pdf`,
  `e4_sensitivity.pdf`, `e5_partial.pdf`
- `outputs/tables/`: `datasets.tex`, `e1_relations.tex`, `views.tex`, `e3_validity.tex`,
  `rephrased.tex`, `e9_overhead.tex`, `e7_robustness.tex`, `ablation.tex`, `e13_scale.tex`
- `outputs/claims.yaml`: 206 claims, including every number the paper quotes

It reads only the shipped result files, the tapes, and the reworded predicate pools. It does not
need `configs/predicates/pubmed.yaml`, the network, or a GPU. On the development machine it took
about 6 seconds and reproduced the paper's claims and tables exactly. It also prints the cost
(E2) and validity (E3) summary tables to stdout.

---

## 4. Level B: re-run the experiments from the tapes

### 4.1 One-time preparation

**PubMed predicate pool.** The 25 PubMed predicates come from the ScaleDoc predicate file. The
source repository has no licence, so the file is not redistributed. Fetch it at runtime:

```bash
make pubmed-pool     # = uv run python -m semviews.workloads.datasets --pubmed-pool
```

The command prints the SHA-256 of the fetched pool and whether it matches the pool the
experiments used (`POOL_SHA256` in `src/semviews/workloads/datasets.py`).

This command downloads the query file from the public ScaleDoc repository on GitHub. The URL is
in the function and is recorded in the `source:` field of the output. It then writes
`configs/predicates/pubmed.yaml` with ids `q00`, `q01`, … and `ext..`, and drops the
"This is a random …" entries of the extended set. Every experiment that touches PubMed needs this
file (E1, E2, E3, E4, E5, E9, E10, E11, E12, sessions, headroom).

The file is fetched from the repository's `main` branch. Check that its predicate ids match the
tape:

```bash
uv run python - <<'EOF'
import pandas as pd, yaml
ids = {"pubmed/" + p["id"] for p in yaml.safe_load(open("configs/predicates/pubmed.yaml"))["predicates"]}
tape = set(pd.read_parquet("data/tapes/pubmed__oracle-llama.parquet", columns=["predicate_id"]).predicate_id)
print(len(ids), len(tape), ids == tape)   # expect 25 25 True
EOF
```

**Sentence encoder and embedding cache.** Row and predicate embeddings come from
`sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU. The model is downloaded on first
use. The embedding cache (`data/cache/`) is not shipped, so the first run of each dataset embeds
its 5,000 rows and writes `data/cache/rowemb_<dataset>.npy`. The shipped results were computed
with the development machine's cache. Recomputing embeddings with other library versions or
hardware could change floating-point values slightly. This has not been tested.

### 4.2 How the runner works

`experiments/run_all.py` is the main runner. Each job replays one workload order for one method
on one dataset, and one job runs per worker process.

```text
uv run python experiments/run_all.py [--config CONFIG] [--only E2 E3 ...] [--workers N]
                                     [--datasets D ...] [--methods M ...] [--tag TAG]
                                     [--outdir SUBDIR] [--semviews-only]
```

| Flag | Effect |
| --- | --- |
| `--config` | Default `configs/experiments/sprint.yaml` |
| `--only` | Experiments to run, out of `E2 E9 E3 E4 E5 E11 E12 E10 E7` (all if omitted) |
| `--workers` | Worker processes; default is CPU count − 2. Results do not depend on it (every job is seeded independently) |
| `--datasets` | Restrict datasets. This also adds `__<d1>_<d2>…` to the output file name |
| `--methods` | Replace the method lists of E2 and E3 (`methods_main` and `methods_validity`) and drop the E2 ablations |
| `--tag` | Add `__<tag>` to the output file name |
| `--outdir` | Write to `experiments/results/<outdir>/` instead of `experiments/results/` |
| `--semviews-only` | Keep only `semviews*` jobs, except `semviews_strata` |

Each experiment writes one file, `experiments/results/[<outdir>/]<E>[__<datasets>][__<tag>].parquet`,
and prints `<E>: <jobs> runs, <rows> rows, <seconds>s`.

> **Warning: avoid double counting.** Because of how `figures.py` loads files (§2), do not leave
> extra files in `experiments/results/` or `op2/` whose names match the shipped ones. In
> particular, `run_all.py` with no `--datasets`, `--tag`, or `--outdir` writes un-suffixed `results/E2.parquet`,
> `results/E3.parquet`, and so on. `figures.py` would load these **in addition to** the shipped
> per-dataset files `results/E2__<dataset>.parquet`. It would also label their `semviews` rows
> as the first-draft operator, although they come from the paper's operator. To check a replay,
> either regenerate an `op2/` file **under its exact shipped name** (§4.3), or write to a separate
> directory with `--outdir rerun` and compare (§4.8). `figures.py` never reads subdirectories
> other than `op2/`.

### 4.3 The paper's operator (`experiments/results/op2/`)

These commands regenerate every `op2/` file that the runner produces, under the shipped file
names. Run them from the repository root.

```bash
# E2: oracle calls over random-order workloads (20 orders), operator with and without views
uv run python experiments/run_all.py --only E2 --outdir op2 --tag main \
    --methods semviews semviews_noviews
#   -> experiments/results/op2/E2__main.parquet

# E2 ablations of the operator (rows of the ablation table and the shaping-size panel of the sensitivity figure)
uv run python experiments/run_all.py --only E2 --outdir op2 --tag abl \
    --methods semviews_allviews semviews_noemb semviews_nohelper semviews_noviews_noemb semviews_shape100 semviews_shape400
#   -> experiments/results/op2/E2__abl.parquet

# E3: validity over 1,000 workload orders per dataset (validity_seeds)
uv run python experiments/run_all.py --only E3 --outdir op2 --tag nv --methods semviews_noviews
#   -> experiments/results/op2/E3__nv.parquet
uv run python experiments/run_all.py --only E3 --outdir op2 --tag reused --methods semviews
#   -> experiments/results/op2/E3__reused.parquet   (see note below)

# E4 sensitivity, E5 partial views, E7 oracle change, E9 unrelated control,
# E10 adversarial catalogs, E11 appends: the operator's own methods only
uv run python experiments/run_all.py --only E4 E5 E7 E9 E10 E11 --outdir op2 --tag run --semviews-only
#   -> experiments/results/op2/{E4,E5,E7,E9,E10,E11}__run.parquet

# Reworded workload W_R (Reviews and GoEmotions, original pool plus 15 rewordings each)
uv run python experiments/rephrased.py cost     # 20 orders  -> experiments/results/op2/ER.parquet
uv run python experiments/rephrased.py valid    # 200 orders -> experiments/results/op2/ER_valid.parquet

# Headroom (perfect score, fully warm catalog) and tiled table-size simulation, paper operator
uv run python experiments/headroom.py paper     # -> experiments/results/op2/HEADROOM.parquet

# E13 (appendix): real relations of 100,000 tuples. Building them needs the upstream sources
# (DBpedia train split; first 250,000 lines of Electronics.jsonl in data/raw/); the shipped
# tables, tapes, and results make this unnecessary for replay.
uv run python experiments/scale_real.py data     # -> data/datasets/dbpedia_100k.parquet
uv run python experiments/scale_real.py reviews  # -> data/datasets/reviews_100k.parquet
uv run python experiments/scale_real.py          # human-label oracle -> experiments/results/op2/E13_scale.parquet
uv run python experiments/scale_real.py llm      # oracle tapes -> experiments/results/op2/E13_llm__{dbpedia,reviews}_100k.parquet

# Exploration-session workloads W_S (not quoted in the paper)
uv run python experiments/sessions.py           # -> experiments/results/op2/ES.parquet
```

Note on `op2/E3__reused.parquet`: the shipped file was not written by `run_all.py`. It holds the
validity runs of the operator configuration `best` from `experiments/improvements.py valid` (1,000
orders per dataset; those runs are also stored as `results/IMP_E3__<dataset>.parquet`),
relabeled as method `semviews`, experiment `E3`, with each dataset's workload name. At 5,000 rows,
that configuration equals the paper's operator. The file's oracle calls and target outcomes match
the `best` rows of `IMP_E3__*.parquet` exactly. The `run_all.py` command above runs the same
configuration on the same orders and seeds. For E2, the corresponding rerun through `run_all.py`
(`op2/E2__main.parquet`) reproduced the `improvements.py` run query by query. A rerun of E3
through `run_all.py` has not been compared with the shipped file.

Contents of each `op2/` file:

| File | Experiment | Methods | Orders (seeds) |
| --- | --- | --- | --- |
| `E2__main` | E2 | `semviews`, `semviews_noviews` | 20 |
| `E2__abl` | E2 ablations | `semviews_allviews`, `_noemb`, `_nohelper`, `_noviews_noemb`, `_shape100`, `_shape400` | 20 |
| `E3__nv`, `E3__reused` | E3 | `semviews_noviews`, `semviews` | 1,000 |
| `E4__run` | E4 | `semviews`, `semviews_noviews` × targets {0.8/0.8, 0.9/0.9, 0.95/0.95, 0.95/0.8, 0.8/0.95}, δ {0.01, 0.05, 0.1, 0.2}, rows {1,000, 2,500, 5,000} | 20 |
| `E5__run` | E5 | `semviews` × view coverage {0.1, 0.25, 0.5, 0.75, 1.0} | 20 |
| `E7__run` | E7 | `semviews`, `semviews_noviews` × {same model, cross-model views}; GoEmotions, first 1,000 tuples, oracle `oracle-gptoss` | 20 |
| `E9__run` | E9 | `semviews`, `semviews_noviews` on the 8-predicate control workload W0 | 20 |
| `E10__run` | E10 | `semviews`, `semviews_noviews` × {adversarial, inverted} | 200 |
| `E11__run` | E11 | `semviews`, `semviews_noviews`, views on the first 80% of tuples | 20 |
| `ER` | W_R cost | `oracle_all`, `semviews`, `semviews_noviews`, `bargain_pr`, `lotus_cascade`, `exact_cache`, `sim_cache@{0.7,0.8,0.9}`, `semweave` | 20 |
| `ER_valid` | W_R validity | `semviews`, `semviews_noviews`, `bargain_pr`, `lotus_cascade`, `sim_cache@{0.8,0.9}`, `semweave` | 200 |
| `ES` | W_S sessions | `semviews`, `semviews_noviews`, `bargain_pr`, `oracle_all` | 20 |
| `HEADROOM` | headroom | variant `paper`; modes `perfect`, `warm`, `scale` (k = 1, 4, 20), `scale_shape` (k = 4, 20) | 4 |
| `E13_scale` | E13, DBpedia 100k, human-label oracle, 40 predicates | `semviews`, `semviews_noviews` (helper off) × rows {5,000, 20,000, 100,000} | 4 |
| `E13_llm__dbpedia_100k`, `E13_llm__reviews_100k` | E13, LLM oracle, 8 predicates each | `semviews`, `semviews_noviews` (helper off) × rows {5,000, 20,000, 100,000} | 20 |
| `E8_live__2000` | E8 | see §4.6 | n/a |

### 4.4 Baselines and the first-draft operator (`experiments/results/`)

The shipped files were produced dataset by dataset with commands of this form. `<d>` is one of
`pubmed`, `goemotions`, `dbpedia`, `reviews`.

| Shipped file(s) | Command |
| --- | --- |
| `E2__<d>`, `E3__<d>`, `E4__<d>`, `E5__<d>`, `E9__<d>` | `uv run python experiments/run_all.py --datasets <d> --only E2 E9 E4 E5 E3` |
| `E2__<d>__sw`, `E3__<d>__sw` (SemWeave) | `uv run python experiments/run_all.py --datasets <d> --only E2 E3 --tag sw --methods semweave` |
| `E2__goemotions__bv`, `E2__pubmed__bv` (BARGAIN on the operator's score) | `uv run python experiments/run_all.py --datasets <d> --only E2 --tag bv --methods bargain_views` |
| `E7__goemotions` | `uv run python experiments/run_all.py --datasets goemotions --only E7` |
| `E10__dbpedia_reviews`, `E10__goemotions_pubmed` | `uv run python experiments/run_all.py --datasets dbpedia reviews --only E10` (and `goemotions pubmed`) |
| `E11`, `E12` | `uv run python experiments/run_all.py --only E11 E12` |

The method lists in `configs/experiments/sprint.yaml` now include `bargain_views` and `semweave`
in `methods_main`. A fresh `--only E2` run therefore produces those methods in the main file. When
the shipped GoEmotions and PubMed files were written, those methods were not yet in the list,
which is why separate `__bv` and `__sw` files exist.

Two caveats apply to re-running these files:

1. **Baseline rows** (`oracle_all`, `lotus_cascade*`, `bargain_*`, `proxy_lr`, `exact_cache`,
   `sim_cache@*`, `sim_proxy`, `inferred_reuse`, `semweave`, `lotus_strict*`) do not depend on the
   operator configuration and can be compared directly.
2. **Operator rows** (`semviews*` other than `semviews_strata`) in these files come from the
   first-draft configuration. The current runner applies the paper's operator to every `semviews*`
   method, so re-running the commands above produces paper-operator rows under the old names.
   Do **not** write such a rerun back into `experiments/results/`; use `--outdir rerun` (§4.8).
   The method name `semviews_v1` (defined in `runner.py` as the paper's operator with refit
   rounds, without-replacement sampling, planning margin, and δ split all switched off) is meant to
   reproduce the first-draft `semviews` rows. It has not been compared with the shipped rows.
   No command-line switch reproduces the first-draft variants of the other `semviews_*` methods.

### 4.5 Relation analyses (E1)

```bash
uv run python experiments/e1_relations.py        # -> results/E1_pairs.parquet, results/E1_summary.parquet
uv run python experiments/e1_predictability.py   # -> results/E1_predictability.parquet
```

Both scripts take optional dataset names as arguments. The default is all four. They read only
the oracle tapes, the predicate pools (including human-label definitions), and
`data/relations/<dataset>.json`. They write into `experiments/results/` and overwrite the shipped
files.

### 4.6 Live LOTUS run (E8), replayed

`experiments/e8_live.py` runs three LazyFrame `sem_filter` queries on the first N Reviews tuples
through LOTUS. It runs them twice: natively with the LOTUS cascade, and with `ViewOptimizer`
(`CertifiedFilter` with the paper's operator). It uses `TapedLM`, a record-and-replay LM backed by
`data/tapes/e8_live.sqlite`, which holds 15,000 recorded responses.

```bash
uv run python experiments/e8_live.py 2000        # -> experiments/results/op2/E8_live__2000.parquet
```

A request that is on the tape is replayed. A request that is not on the tape goes live through the
proxy, which fails if no proxy is running. Use N = 2000, the size of the shipped run. The script's
default, N = 500, writes `op2/E8_live__500.parquet` and is not covered by the shipped run. A replay
without a proxy has not been tested.

`experiments/results/E8_live.parquet` is an earlier 500-tuple run with the first-draft adapter,
written by an earlier version of the script. The current script cannot regenerate it. E8 is an
integration check: the paper quotes no number from it, and `figures.py` does not read it.

### 4.7 Development studies shipped for completeness

These files record how the paper's operator configuration was chosen, as described in the
paper's reproducibility appendix. Tuning used only GoEmotions and PubMed, each round on fresh
workload orders. `figures.py` reads none of these files.

```bash
uv run python experiments/improvements.py tune    # orders 0-5,   pilot datasets -> results/IMP_tune.parquet
uv run python experiments/improvements.py tune2   # orders 6-11,  pilot datasets -> results/IMP_tune2.parquet
uv run python experiments/improvements.py tune3   # orders 12-17, pilot datasets -> results/IMP_tune3.parquet
uv run python experiments/improvements.py full    # all datasets, 20 orders      -> results/IMP_E2.parquet
uv run python experiments/improvements.py valid   # 1,000 / 200 orders           -> results/IMP_E3__<dataset>.parquet
uv run python experiments/headroom.py             # first-draft and candidate configs -> results/HEADROOM.parquet
uv run python experiments/headroom.py fix         # tuning round 3, tiled check  -> results/HEADROOM_fix.parquet
```

`improvements.py valid` accepts dataset names as extra arguments.

### 4.8 Checking a replay against a shipped file

`make replay-check` does this for the paper's operator on PubMed (E2 main, 20 orders): it writes
`experiments/results/rerun/E2__pubmed__main.parquet` and runs
`experiments/compare_replay.py <rerun file> <shipped file>`, which matches rows on
(dataset, method, seed, query_idx, predicate_id), requires identical oracle calls, outcomes,
precision, and recall, and exits non-zero on any difference. Use the same script for any other
rerun, for example:

```bash
uv run python experiments/run_all.py --only E9 --outdir rerun --tag run --semviews-only --datasets goemotions
uv run python experiments/compare_replay.py experiments/results/rerun/E9__goemotions__run.parquet experiments/results/op2/E9__run.parquet
```

For files in `results/` (not `op2/`), compare only baseline rows, since operator rows there are
from the first-draft operator (§4.4). Manually:

```bash
uv run python experiments/run_all.py --datasets pubmed --only E9 --outdir rerun
uv run python - <<'EOF'
import pandas as pd
key = ["dataset", "method", "seed", "query_idx", "predicate_id"]
a = pd.read_parquet("experiments/results/E9__pubmed.parquet")
b = pd.read_parquet("experiments/results/rerun/E9__pubmed.parquet")
baseline = ~a.method.str.startswith("semviews")          # operator rows here are first-draft (§4.4)
m = a[baseline].merge(b, on=key, suffixes=("_shipped", "_rerun"))
print(len(a[baseline]), len(m),
      (m.oracle_calls_shipped == m.oracle_calls_rerun).mean(),
      (m.met_shipped == m.met_rerun).mean())
EOF
```

For `op2/` files, compare all methods. The `wall_ms` and `git_commit` columns are expected to
differ.

### 4.9 Recorded runtimes

Runtimes were recorded only for the first-draft runs in `experiments/results/`. Those runs used
an 18-core Apple-silicon machine with the default 16 worker processes. Some ran while helper
tapes were being recorded on the same machine.

| Experiment, one dataset | Jobs | Recorded wall-clock |
| --- | --- | --- |
| E2 (all methods and ablations) | 400–420 | 225–583 s |
| E3 (1,000 orders for operator methods, 200 for baselines) | 3,800 | 1,236–4,249 s |
| E4 | 720 | 131–2,071 s |
| E5 | 100 | 27–1,007 s |
| E9 | 100 | 18–28 s |
| E7 (GoEmotions, 1,000 tuples) | 200 | 24 s |
| E10 (two datasets) | 1,760 | 419 s |

Runtimes for the `op2/` runs, the reworded and session workloads, and the headroom study were not
recorded. The paper's operator adds refit rounds and spends a median of about 0.85 s of
single-threaded compute per query (claim `e3 semviews ms per query`). Expect the E3 runs to
dominate. Memory use was not recorded.

### 4.10 Optional: replaying the judge

`data/relations/*.json` and `configs/predicates/*_rephrased.yaml` were produced by the judge
model through `TapedLM` and recorded in `data/tapes/judge.sqlite`. The same commands as in §5.5
should replay from that tape without model access when every request is on the tape. They
overwrite the shipped files. This replay has not been tested without a proxy.

---

## 5. Level C: re-record the tapes with live models

> **Cost warning.** Re-recording calls a hosted 70B-class model 875,000 times. At the token prices
> in `configs/budgets.yaml`, the shipped tapes correspond to about **$136**:
>
> | Tapes | Labels | USD at `budgets.yaml` prices |
> | --- | --- | --- |
> | `oracle-llama`, four pools | 725,000 | 114.57 |
> | `oracle-llama__rw` | 150,000 | 18.91 |
> | `oracle-gptoss` | 40,000 | 2.60 |
> | `helper` (local) | — | 0 |
>
> This figure is computed from the token counts stored in the tapes. It excludes the judge, the
> noise check, and the E8 live run. `configs/budgets.yaml` sets `usd_cap_total: 150` and
> `max_live_calls_total: 700000`. Only the tape recorder `build_tape` enforces a cap: before every
> chunk it prices all recorded part files and stops (exit code 2) once `usd_cap_total` is reached.
> `max_live_calls_total` is not enforced by any script. The judge, `rephrase`, `noise_check`,
> `e8_live.py`, and `fast_helper` have no cap. Set the prices in `budgets.yaml` to your endpoint's
> prices and lower the cap before you start.

### 5.1 Models and serving

All model calls go through a local LiteLLM proxy, addressed by alias:

| Alias | Model | Role | Sampling (`configs/models.yaml`) |
| --- | --- | --- | --- |
| `oracle-llama` | Llama 3.3 70B Instruct | oracle for all tapes | temperature 0, max_tokens 8 |
| `oracle-gptoss` | gpt-oss-120b | second oracle (E7) | temperature 0, max_tokens 1024, reasoning_effort low |
| `judge` | gpt-oss-120b | relation judge, rewordings | temperature 0, max_tokens 2048, reasoning_effort medium |
| `helper` | Llama 3.2 3B Instruct Q4_K_M GGUF, served by llama.cpp | helper scores (log-probabilities) | temperature 0, max_tokens 4 |

The embedding model, `sentence-transformers/all-MiniLM-L6-v2`, runs in-process.

Setup:

1. **Helper model.** Install llama.cpp so that `llama-server` is on your `PATH`. Place the
   Q4_K_M GGUF of Llama 3.2 3B Instruct at `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` (path from
   `helper_gguf` in `configs/models.yaml`; override it with `HELPER_GGUF`). `serving/llm_up.sh`
   starts it on `127.0.0.1:8081` with `HELPER_PARALLEL` slots (default 16). Helper tape requests go
   directly to this server (`direct_routes.helper` in `configs/models.yaml`), not through the proxy.
2. **Proxy.** `serving/llm_up.sh` starts the proxy from `serving/.venv/bin/litellm`. That is a
   separate environment with LiteLLM installed from the hash-locked `serving/requirements.lock`
   (`litellm[proxy]==1.103.2`). The repository does not script this step. One way to do it:

   ```bash
   uv venv serving/.venv --python 3.12
   uv pip install --python serving/.venv/bin/python --require-hashes -r serving/requirements.lock
   ```

3. **Proxy configuration.** Copy `serving/litellm.example.yaml` to `serving/litellm.yaml`. Fill in
   the routes for `oracle-llama`, `oracle-gptoss`, and `judge`, pointing to an OpenAI-compatible
   hosted endpoint of your choice configured in LiteLLM. Keep the helper route, caching off, and no
   fallbacks.
4. **Secrets.** Copy `.env.example` to `.env` in the repository root and fill it in. Never commit or print it. It must define
   `LITELLM_MASTER_KEY` and any variables your `serving/litellm.yaml` reads. `llm_up.sh` loads it
   into the proxy's environment only.
5. Start, check, and stop the services:

   ```bash
   make llm-up      # helper on :8081, proxy on :4000; logs in serving/logs/
   make llm-check   # one answer per alias, temperature-0 repeatability, helper log-probabilities, latency
   make llm-down
   ```

   `make helper-bench` (`serving/helper_bench.py`) benchmarks helper models.

### 5.2 Datasets (optional)

The shipped `data/datasets/*.parquet` are the reference samples. To rebuild them:

```bash
uv run python -m semviews.workloads.datasets                  # all four; --n 5000 --seed 20261001 by default
uv run python -m semviews.workloads.datasets goemotions dbpedia
```

The builder downloads public source datasets, normalizes whitespace, caps text length (700
characters; 1,200 for PubMed), removes duplicates, draws a seeded sample, and assigns content-hash
row ids. Sources are listed in the module docstring and in [DATA.md](DATA.md). Reviews also needs
`data/raw/electronics_head.jsonl` (the first 60,000 lines of the source corpus's Electronics
review file), which is not shipped. A rebuild matches the shipped files only if the upstream
sources are unchanged.

### 5.3 Oracle tapes

```bash
for d in pubmed goemotions dbpedia reviews; do
  uv run python -m semviews.adapters.lotus.build_tape --dataset $d --model oracle-llama
done
# second oracle, first 1,000 GoEmotions tuples (E7)
uv run python -m semviews.adapters.lotus.build_tape --dataset goemotions --model oracle-gptoss --rows 1000
```

`build_tape` uses LOTUS's own `sem_filter` prompt, one tuple per prompt. Flags: `--predicates`,
`--rows`, `--chunk` (default 500), `--batch` (default 32), `--pool`, `--tag`, `--consolidate`.
It writes one resumable part file per predicate to `data/tapes/parts/<dataset>__<alias>/`, retries
transport errors with back-off, and finally consolidates the parts into
`data/tapes/<dataset>__<alias>.parquet`. `serving/run_tape.sh <alias> "<datasets>"` wraps the
loop and reads the `CHUNK` and `BATCH` environment variables. During development, recorded oracle
throughput ranged from 0.5 to 74 calls per second per predicate, depending on the endpoint's rate
limits.

### 5.4 Helper tapes

```bash
for d in pubmed goemotions dbpedia reviews; do
  uv run python -m semviews.adapters.lotus.fast_helper --dataset $d --threads 24
  uv run python -m semviews.adapters.lotus.build_tape --dataset $d --model helper --consolidate
done
```

`fast_helper` builds each prompt with LOTUS's `filter_formatter`, posts the same chat-completions
request to the local llama.cpp server (with `logprobs` and `top_logprobs=10`), and computes the
score with LOTUS's `format_logprobs_for_filter_cascade`. It writes only part files, so run
`build_tape --model helper --consolidate` afterwards. `serving/run_fast_helper.sh` loops over the
four datasets. `serving/helper_watchdog.sh` restarts `llama-server` every 40 minutes while it
runs, because llama.cpp throughput degraded over hours of load. During development the recorded
helper throughput was about 14–16 tuples per second per predicate with 24 threads.
`build_tape --model helper` (without `--consolidate`) also works but is slower.

### 5.5 Judge relations and the reworded workload

```bash
# pairwise relations for SemWeave and inferred reuse -> data/relations/<dataset>.json
for d in pubmed goemotions dbpedia reviews; do
  uv run python -m semviews.adapters.lotus.judge --dataset $d
done

# reworded predicates (seeded choice of 15 per pool, --seed 0 by default) -> configs/predicates/<d>_rephrased.yaml
for d in reviews goemotions; do
  uv run python -m semviews.adapters.lotus.rephrase --dataset $d --n 15
  uv run python -m semviews.adapters.lotus.build_tape --dataset $d --model oracle-llama --pool ${d}_rephrased --tag rw
  uv run python -m semviews.adapters.lotus.fast_helper --dataset $d --pool ${d}_rephrased --tag rw
  uv run python -m semviews.adapters.lotus.build_tape --dataset $d --model helper --tag rw --consolidate
  uv run python -m semviews.adapters.lotus.judge --dataset $d --extra ${d}_rephrased   # -> data/relations/<d>_rephrased.json
done
```

`judge` and `rephrase` record through `TapedLM` into `data/tapes/judge.sqlite`. A request already
on the tape is replayed, not re-asked. To record fresh responses, move the shipped
`judge.sqlite` aside first.

### 5.6 Noise check

```bash
uv run python -m semviews.adapters.lotus.noise_check --dataset goemotions --pairs 2000 --repeats 2
#   -> experiments/results/noise_check_goemotions.parquet  (claim `noise agreement`)
```

This command re-labels 2,000 random (tuple, predicate) pairs live, twice each (sample
`random_state=7`). It samples from the **part files** in `data/tapes/parts/goemotions__oracle-llama/`.
Those files are not shipped, so it runs only after §5.3 has been done locally.

### 5.7 Live E8

Move `data/tapes/e8_live.sqlite` aside to force live calls. Then run
`uv run python experiments/e8_live.py 2000`. The run records into a new tape and writes
`experiments/results/op2/E8_live__2000.parquet`.

After re-recording, run the level-B commands (§4) and level A (§3). New tapes give new labels, so
the numbers will differ from the paper's. The oracle is not perfectly deterministic at
temperature 0 (see the noise check).

---

## 6. Where each table and figure comes from

The commands are in §4.3–§4.6. Paths are relative to `experiments/results/`. "Results read" lists
what `figures.py` loads. A claim key prefix such as `e2 breakdown` covers every key that starts
with it in `claims.yaml`.

| Paper item | Experiment | Script / command | Results read | Output | `claims.yaml` keys |
| --- | --- | --- | --- | --- | --- |
| Table: datasets and predicate pools (pool sizes, selectivity) | tapes | `build_tape` (§5.3) | `data/tapes/*__oracle-llama.parquet` | `tables/datasets.tex` | — |
| Setup text: number of recorded labels, second oracle, label repeatability | tapes, noise check | §5.3, §5.6 | tapes; `noise_check_goemotions.parquet` | — | `tape oracle calls`, `tape gptoss calls`, `noise agreement` |
| Table: containment and exclusion among predicates, judged implications | E1 | `e1_relations.py` | `E1_summary.parquet` (per-pair detail in `E1_pairs.parquet`) | `tables/e1_relations.tex` | `e1 pairs`, `e1 contain 0.1 pairs`, `e1 judged implies`, `e1 judged implies bad share` |
| Discussion: how well other predicates' labels predict a predicate (AUROC) | E1b | `e1_predictability.py` | `E1_predictability.parquet` | — | `e1 auroc <dataset>` |
| Figure: cumulative oracle calls as a share of exact evaluation | E2 (dotted lines from E3) | §4.3 `E2 --tag main`; §4.4 | `E2.parquet`, `E2__*` in `results/` and `op2/`; E3 files | `figures/e2_cumulative.pdf` | `e2 saving all`, `e2 share all`, `e2 ratio valid`, `e2 vs bargain`, `e2 view gain`, `e2 short`, `e2 semviews worst query share`, `e2 semweave share`, `e2 <dataset> …` |
| Table: calls saved by views (related, unrelated, all, last 10) | E2 | as above | E2 files; oracle tapes (correlations) | `tables/views.tex` | `views …`, `e2 semviews saving`, `e2 bargain saving`, `e2 saving ratio`, `e2 late …`, `views v1 related gain` |
| Text: cost by selectivity | E2 | as above | E2 files; oracle tapes | — | `sel …` |
| Text: BARGAIN on the operator's unrefined score; refinement gain | E2 | §4.4 (`__bv`), §4.3 | E2 files (`bargain_views`, `semviews_v1`) | — | `e2 vs bargain views`, `e2 vs bargain views v1`, `e2 refine cut` |
| Figure: where the oracle calls go (breakdown by step) | E2 | §4.3 `E2 --tag main` | E2 files (`stats` column) | `figures/e2_breakdown.pdf` | `e2 breakdown …`, `e2 refit rounds`, `e2 close steps` |
| Setup text: dollar cost agrees with call counts | E2 | as above | E2 files (`usd` column) | — | `e2 usd call gap` |
| Table: share of queries that miss a target, per method and dataset | E3 | §4.3 `E3 --tag nv/reused`; §4.4 | `E3__*` in `results/` and `op2/` | `tables/e3_validity.tex` | `e3 …` (`e3 semviews max miss`, `e3 semviews seeds`, `e3 lotus …`, `e3 simcache …`, `e3 semweave …`, …) |
| Figure: cost against validity | E3 | as above | E3 files | `figures/e3_tradeoff.pdf` | `e3 cheaper than semviews min miss` |
| Text: LOTUS cascade with stricter internal targets | E12 | §4.4 `--only E11 E12` | `E12.parquet` | — | `e12 strict lotus …` |
| Table: reworded workload, calls and misses on repeated questions | W_R | `rephrased.py cost`, `rephrased.py valid` | `op2/ER.parquet`, `op2/ER_valid.parquet`; `*__oracle-llama__rw` tapes; `configs/predicates/*_rephrased.yaml` | `tables/rephrased.tex` | `wr …` |
| Figure: sensitivity to targets, δ, table size, shaping-sample size | E4 (+ E2 ablations for the shaping-size panel) | §4.3 `--tag run`, `E2 --tag abl`; §4.4 | `E4__*`, E2 files | `figures/e4_sensitivity.pdf` | `e4 rows small share`, `e4 rows large share` |
| Table: control workloads of nearly uncorrelated predicates | E9 | §4.3 `--tag run`; §4.4 | `E9__*` | `tables/e9_overhead.tex` | `e9 worst overhead` |
| Text: adversarial and inverted views | E10 | §4.3 `--tag run`; §4.4 | `E10__*` | — | `e10 …` |
| Text: appends | E11 | §4.3 `--tag run`; §4.4 | `E11*` | — | `e11 …` |
| Table: views recorded under one oracle, reused under another | E7 | §4.3 `--tag run`; §4.4 | `E7__*` | `tables/e7_robustness.tex` | `e7 …` |
| Figure: partial views | E5 (+ E2 no-views baseline) | §4.3 `--tag run` | `E5__*`, E2 files | `figures/e5_partial.pdf` | `e5 …` |
| Table: ablations relative to the operator | E2 ablations | §4.3 `E2 --tag main/abl`; §4.4 (first draft `semviews_v1`, `semviews_strata`) | E2 files | `tables/ablation.tex` | `e2 refine cut inv` |
| Text: latency drivers | E3, E2 | as above | E3, E2 files (`wall_ms`, `stats`) | — | `e3 semviews ms per query`, `e2 refit rounds`, `e2 close steps` |
| Proposition on the certification floor | closed form | `figures.py` | — | — | `floor coef`, `floor draws at one pct` |
| Discussion: headroom (perfect score, warm catalog) and tiled scale | HEADROOM | `headroom.py paper` | `op2/HEADROOM.parquet` | — | `headroom …` |
| Not in the paper | W_S sessions; E8 live; tuning studies | `sessions.py`; `e8_live.py 2000`; §4.7 | `op2/ES`, `op2/E8_live__2000`, `E8_live`, `IMP_*`, `HEADROOM*` in `results/`, `E1_pairs` | — | — |

Experiment ids and workloads used in the result files:

- **E1**: relations
- **E2**: cost
- **E3**: validity
- **E4**: sensitivity
- **E5**: partial views
- **E7**: oracle change
- **E8**: live LOTUS
- **E9**: unrelated control
- **E10**: adversarial catalogs
- **E11**: appends
- **E12**: strict LOTUS
- **ER / ER_valid**: reworded workload W_R
- **ES**: sessions W_S

Workload column values:

- `W1`: PubMed, external pool
- `W2a`: Reviews, analyst-style pool
- `W3`: GoEmotions and DBpedia, taxonomy-derived pools
- `W0`: control, 8 predicates
- `WR`: reworded workload
- `WS`: session workload

---

## 7. Seeds and determinism

All randomness is seeded from the configuration or fixed in the code:

| What | Seed |
| --- | --- |
| Dataset samples | `--seed 20261001` (`semviews.workloads.datasets`) |
| Workload order for seed `s` | `numpy.random.default_rng(10_000 + s)` shuffle of the pool (`order_for` in `run_all.py`) |
| Session order (W_S) for seed `s` | `default_rng(20_000 + s)` (`sessions.py`) |
| Method randomness for seed `s` | `default_rng(s)` per job; partial-view masking `default_rng(s + 7_919)` (`runner.py`) |
| Adversarial and inverted catalogs (E10) | `default_rng(99)` |
| Row-embedding PCA | `random_state=0` |
| Reworded predicate choice | `rephrase --seed 0` |
| Noise-check sample | `random_state=7` |
| E8 | `cascade_IS_random_seed=0`, `ViewOptimizer(seed=0)` |

Counts come from `configs/experiments/sprint.yaml`:

- `orders: 20` sets the workload orders for E2, E4, E5, E7, E9, and W_S.
- `validity_seeds: 1000` sets the E3 orders for `semviews*` methods.
- `validity_seeds_baselines: 200` sets the E3 orders for the other methods.

Some counts are fixed in the scripts:

- E10: 200 orders for `semviews*` methods, 20 for the others.
- E11: 20 orders.
- E12: 100 orders.
- W_R: 20 orders for cost, 200 for validity.
- Headroom: 4 orders.

Each job seeds itself, so results do not depend on `--workers` or on job scheduling.
`run_all.py` limits each worker to one BLAS thread. All models run at temperature 0, and every
experiment except the live E8 run reads its labels from the tapes. A replay is therefore expected
to reproduce oracle-call counts and target outcomes exactly. Two independent runs of the E2
configuration of the paper's operator produced identical call counts query by query (§2). Bitwise
reproduction on other platforms or library versions has not been tested; see the embedding-cache
note in §4.1. Hyperparameters were fixed on pilot data before the full runs and are frozen in the
experiment configuration.

---

## 8. Metadata in every result row

Runner outputs (E2–E12, ER, ES) contain one row per query, with these columns:

- `experiment`, `workload`, `dataset`, `method`, `seed`, `query_idx`, `predicate_id`
- `oracle_calls`; `oracle_calls_cert` (certification draws); `cum_oracle_calls`; `helper_calls`;
  `usd` (priced with `configs/budgets.yaml`)
- `precision`, `recall`, `precision_target`, `recall_target`, `delta`, `met` (both targets met)
- `n_rows`, `n_returned`, `selectivity`, `wall_ms`
- `stats`: JSON with operator or baseline internals, for example shaping size, refit labels,
  oracle-region size, gap-closing steps, cache hits, reused rows
- reproducibility metadata:
  - `git_commit`: short commit hash of the code that produced the row
  - `config_hash`: first 12 hex digits of a SHA-256 over the method name, the targets, and any
    per-job configuration overrides
  - `model_ids`: oracle alias
  - `lotus_version`: installed `lotus-ai` version (1.2.4 in all shipped files)

Notes:

- The `git_commit` values refer to the development history of the code, not to commits of this
  release. A release has no matching history, and a rerun records `unknown` if the directory is
  not a Git repository.
- `config_hash` does not include the `operator:` block of the experiment configuration. First-draft
  and paper-operator rows of the same method can therefore share a hash. Use the directory
  (`results/` or `results/op2/`) to tell them apart.
- `model_ids` holds the alias (`oracle-llama` or `oracle-gptoss`). The alias-to-model mapping is
  in `configs/models.yaml` and in §5.1. Each tape row also records `model`, `model_id` (as returned
  by the endpoint), `prompt_version` (`lotus-1.2.4-sem_filter-default`), `sampling`, token counts,
  latency, and a timestamp.
- `HEADROOM`, `E1_*`, `E8_live*`, and `noise_check_*` files have their own smaller schemas and
  carry no reproducibility metadata columns.

---

## 9. Verifying the numbers

Every number the paper quotes is an entry of `claims.yaml`, written by `experiments/figures.py`:

```yaml
claims:
  e3 semviews max miss:
    value: 0.7                                   # the number
    fmt: '{:.1f}\%'                              # Python format string used to typeset it
    source: experiments/results/E3.parquet       # the results it is computed from
    script: experiments/figures.py               # the script that computed it
```

The `source` field names the experiment's results: `experiments/results/E3.parquet`,
`experiments/results/E3*.parquet`, and similar patterns refer to everything `figures.py` loads
for that experiment, which includes `results/E3__*.parquet` and `results/op2/E3__*.parquet` (§2).
A few sources are tapes (`data/tapes/*__oracle-llama.parquet`) or `experiments/figures.py
(closed form)`.

To verify:

```bash
make figures         # writes outputs/claims.yaml (and figures and tables)
make check-claims    # compares outputs/claims.yaml with reference/claims.yaml
```

To compare by hand:

```bash
uv run python - <<'EOF'
import yaml
new = yaml.safe_load(open("outputs/claims.yaml"))["claims"]
ref = yaml.safe_load(open("reference/claims.yaml"))["claims"]
diff = {k: (ref.get(k, {}).get("value"), new.get(k, {}).get("value"))
        for k in sorted(set(new) | set(ref)) if new.get(k) != ref.get(k)}
print(len(ref), "reference claims;", len(diff), "differ")
for k, (r, n) in diff.items():
    print(f"{k}: reference={r} regenerated={n}")
EOF
```

After level A, no claim should differ. After a level-B rerun written under the shipped file names,
differences point to a non-reproducing replay. Use §4.8 to find which rows differ. After level C,
differences are expected because the labels are new.
