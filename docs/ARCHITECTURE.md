# Architecture

This document explains how the `semviews` code works: the concepts it implements, the
`CertifiedFilter` operator step by step, its configuration, the oracle/tape/cost machinery used
by the experiments, the LOTUS integration, the baselines, and where to extend the system. It is
written for a reader who has the paper and wants to map it onto the code. It describes the code
as it is; where a docstring or comment disagrees with the behavior, the behavior is described.

Related documents: [`README.md`](../README.md) (quick start), [`docs/REPRODUCE.md`](REPRODUCE.md)
(how to rerun the experiments), [`docs/DATA.md`](DATA.md) (datasets, tapes, predicate pools).

---

## 1. Concepts

### Semantic filter, oracle, helper

A **semantic filter** `sem_filter(q)` keeps the rows of a table for which a natural-language
predicate `q` (a LOTUS *langex* such as `"{text} reports a hardware defect"`) holds. The
**oracle** is the expensive model whose yes/no answer *defines* the ground truth for `q`; every
guarantee in this repository is relative to the oracle's labels, not to human labels. The oracle
is reached only through an object with a `label(predicate_id, rows, *, purpose)` method (the
`Oracle` protocol in `src/semviews/oracle/tape.py`). Every label a method uses is paid for
through that call.

The **helper** is a small model that gives a cheap probability score per (row, predicate), as in
LOTUS's model cascade. In the experiments it is Llama 3.2 3B Instruct (GGUF, Q4_K_M) served
locally, and its score is the probability of the positive answer token computed by LOTUS's own
`format_logprobs_for_filter_cascade`. Helper scores are free in oracle terms; they are counted
separately as `helper_calls`.

Model endpoints are referred to only by alias (`configs/models.yaml`):

| Alias           | Role                                   | Public model                 |
|-----------------|----------------------------------------|------------------------------|
| `oracle-llama`  | primary oracle (all tapes)             | Llama 3.3 70B Instruct       |
| `oracle-gptoss` | second oracle (cross-model experiment) | gpt-oss-120b                 |
| `judge`         | relation judge, predicate rewording    | gpt-oss-120b                 |
| `helper`        | cascade helper scores                  | Llama 3.2 3B Instruct (GGUF) |
| (local)         | predicate and row embeddings           | all-MiniLM-L6-v2             |

All model traffic for recording goes through a local LiteLLM proxy by alias (`serving/`); the
experiments themselves replay recorded tapes and need no model access.

### Semantic views and provenance

A **semantic view** (`View` in `src/semviews/catalog/__init__.py`) is a label column over one
table, bound to a predicate id, its text, and the oracle model. Each entry is `1`, `0`, or
`UNKNOWN` (`-1`), so views can be partial. Each entry also carries a **provenance** code:

| Code      | Value | Meaning                                                             |
|-----------|-------|---------------------------------------------------------------------|
| `NONE`    | 0     | no label                                                            |
| `ORACLE`  | 1     | the label came from a paid oracle call for this predicate           |
| `DERIVED` | 2     | the row was accepted or rejected as part of a certified region, without a call |

The rule that makes reuse safe is enforced by construction:

* Only labels with provenance `ORACLE`, **for the predicate being answered**, are used exactly
  (`Catalog.known_oracle`). They are returned as-is and never treated as samples.
* Only **fresh** oracle labels, drawn uniformly at random from a region fixed *before* the
  draws, enter a confidence bound.
* `DERIVED` labels, and every label of every *other* predicate's view, are used only to **shape**
  the computation (as features of a score model that orders the rows). They never enter a bound
  and are never returned as if they were oracle labels.

A **catalog** (`Catalog`) holds the views of one table in a fixed canonical row order. Answering a
query registers its view, so views are a by-product of query answering. `Catalog.register`
merges a new view into an existing one row by row: an `ORACLE` label always wins; a non-oracle
label only fills an entry that has no oracle label.

### Targets

A query comes with `Targets(precision, recall, delta)` (`src/semviews/operators/reuse_filter.py`;
defaults 0.9, 0.9, 0.05). The returned set must have precision at least `precision` and recall at
least `recall` with respect to the oracle's labels, with probability at least `1 - delta` over the
operator's own random draws. The probability is over the operator's sampling only; it holds for
any table, any workload order, and any catalog content.

### Regions

`CertifiedFilter` orders the rows it does not yet know by a score and cuts the order into three
contiguous **regions**:

* **X** (reject): the lowest-scored rows. Returned only if a row was labeled positive.
* **O** (oracle): the middle rows. Labeled in full; returned exactly.
* **A** (accept): the highest-scored rows. Returned except for rows labeled negative.

Certification bounds the number of positives in X from above (missed positives, so recall) and
the number of positives in A from below (true positives returned, so precision and recall).

---

## 2. The `CertifiedFilter` algorithm

`CertifiedFilter` (`src/semviews/operators/region_filter.py`) is the paper's operator. Its entry
point is

```python
CertifiedFilter(oracle, catalog, texts, helper=None, embedder=None, cfg=CertConfig(),
                row_features=None).run(pid, targets, rng, given_views=None, universe=None)
```

* `texts` maps predicate ids to langex strings (used to match views by text).
* `helper` exposes `has(pid)` and `score(pid, rows)`.
* `embedder` is a `semviews.relate.Embedder` (local sentence embeddings).
* `row_features` is an `(n_rows, d)` matrix of free per-row features (PCA-reduced row
  embeddings in all experiments and in the LOTUS adapter).
* `universe` restricts the filter to a subset of catalog positions (used by the LOTUS adapter,
  where the input DataFrame may be a subset of the catalog's rows). Rows outside it are treated
  as known negatives during the run and restored before the view is registered.

It returns a `FilterResult(mask, stats)`: a boolean mask over catalog rows and a dictionary of
diagnostics (`views`, `n_shape`, `n_refit`, `plan_x`, `plan_o`, `plan_a`, `n_x`, `n_a`,
`close_steps`, `ux`, `la`).

All oracle labels go through `CertifiedFilter._label`, which asks the oracle only for rows whose
label is still unknown in the local array `known`, and tags each request with a *purpose* string
(see Section 4).

### Step 0: known labels

`run` starts from `known`, an array of `UNKNOWN`, and fills in every `ORACLE`-provenance label of
the same predicate id already in the catalog (`Catalog.known_oracle`). A repeated predicate is
therefore answered from the labels it already paid for, without trusting any derived decision.

### Step 1: match candidate views (`_candidates`)

If `use_views` is set, candidate views are the catalog's views of other predicates
(`Catalog.others`). When an embedder is present, there are more than `n_candidates` views, and
`view_selection == "embed"`, the operator keeps the `n_candidates` views whose predicate text
(after `predicate_text` strips the `{column}` placeholders) has the highest cosine similarity to
the query's text. With `view_selection == "all"` (or no embedder) every view is a candidate.
`given_views` bypasses matching (the experiments use this only for the "ideal views" analysis).

Matching never decides a relation between predicates. The score model in step 2 decides how much
each view's labels are worth, including negative relations (exclusions).

### Step 2: shaping sample and score (`_features`, `_fit_score`)

The **shaping sample** is drawn uniformly without replacement from the unknown rows:
`n_shape = min(shape_n, shape_frac_max * |unknown|, |unknown|)`. Its labels (purpose `shape`) are
exact and are returned, but they are never used in a bound.

The feature matrix (`_features`) has, per row:

* for each candidate view, two indicators: the view labels the row `1`, and the view does not
  know the row (`UNKNOWN`). "Unknown" is a feature value, so partial views and appended rows
  need no special handling;
* if `use_helper` and the helper covers the predicate, `logit(helper score) / 4`;
* if `use_rowemb` and `row_features` was given, the row-feature columns. In the experiments these
  are a 32-dimensional PCA projection of all-MiniLM-L6-v2 row embeddings, scaled to standard
  deviation 0.5 (`Bench.row_pca` in `src/semviews/workloads/runner.py`; `row_features` in the
  LOTUS adapter does the same on the input DataFrame).

`_fit_score` fits an L2-regularized logistic regression (scikit-learn, `C = cfg.l2`) on every
labeled row inside the universe and predicts a probability `p_hat` for every row. If there are no
features, fewer than ten labels, or only one class among them, it falls back to a constant
(add-0.5 smoothed) rate with a tiny tie-breaker from the first feature column. With `shrink > 0`
the probabilities are shrunk toward the base rate with pseudo-count `shrink`.

### Step 3: plan the regions (`_fit_and_plan`, `_plan`)

`_fit_and_plan` sorts the unknown rows by ascending score (`order`) and forms the prefix sums
`cum_p` of `p_hat` along that order, so that the expected number of positives in any contiguous
block is one subtraction. `_plan` then searches for the plan `(i_x, j_a, n_x, n_a, f_x)`:

* `X = order[:i_x]`, `O = order[i_x:j_a]`, `A = order[j_a:]`;
* boundaries `i_x <= j_a` are taken from `q_grid + 1` evenly spaced positions over the order;
* sample sizes `n_x`, `n_a` come from `n_grid` and must be below 70% of the region they sample
  (a non-empty region with no admissible size is not considered, so X and A, when present, have
  more than `min(n_grid) / 0.7` rows);
* `f_x` is the share of `delta` given to X when both X and A are non-empty, taken from
  `delta_splits`; with a single non-empty region it gets all of `delta` (`_alphas`).

For each candidate the planner *predicts* what certification would conclude by plugging
expected counts into tabulated one-sided Clopper-Pearson bounds (`_tables`, cached per level and
grid):

* expected positives among the draws are `rate * n`; with `plan_margin > 0` they are shifted by
  `plan_margin` standard deviations in the unfavorable direction (`_pred_x`: up for X, down for A);
* with `without_replacement`, the Clopper-Pearson margin is shrunk by the finite-population
  factor `sqrt((N - n) / (N - 1))` (`_fpc`) as an approximation of the hypergeometric bound that
  certification will actually use;
* predicted true positives `T = k_pos + E[pos in O] + lower(A)`, predicted missed positives
  `max(upper(X) - expected positives drawn from X, 0)`, predicted returned size
  `k_pos + E[pos in O] + |A|`, where `k_pos` is the number of known positives.

A candidate is feasible if predicted recall and precision meet the targets. The planner returns
the feasible plan with the fewest predicted calls `|O| + n_x + n_a`, or the all-O plan if nothing
beats `|U|` (the operator then degenerates to exact evaluation). The loops prune any candidate
whose partial cost already reaches the incumbent.

The plan depends on the catalog, the helper, the embeddings, and labels drawn so far; it depends
on no label drawn later.

### Step 3b: refit rounds (optional, on in the paper's configuration)

With `active_rounds > 0`, the operator repeats, up to `active_rounds` times: take the planned O
region; label the `chunk` rows of O whose `p_hat` is closest to 0.5 (purpose `residual`); refit
the score and re-plan. The batch is `chunk = max(active_chunk, active_chunk_frac * |universe|)`,
and the loop stops early once O has at most `chunk` rows or nothing is unknown. Refit labels are
rows the plan would label anyway, they are used exactly, and all of them are drawn before the
regions are fixed, so validity is unaffected. `stats["n_refit"]` records how many labels the
rounds bought.

With `calibrate_uniform` (off in every configuration used in the paper), `_recalibrate` refits a
ridge-penalized Platt map from `logit(p_hat)` to the labels of the uniform shaping sample only, so
that the planner's expected counts are not inflated by refit labels chosen near the boundary.

### Step 4: certify

The last plan is fixed. Then (lines around the `# 4. certify` comment in `run`):

1. O is labeled in full (purpose `residual`).
2. `a_x, a_a = _alphas(delta, |X|, |A|, f_x)`; `a_x + a_a = delta` when both regions exist.
3. From X, `n_x` fresh rows are drawn and labeled (purpose `sample`); likewise `n_a` from A.
   * `without_replacement=True`: draws are `rng.choice(..., replace=False)`, and the bounds are
     the exact hypergeometric count bounds `hg_upper_count` and `hg_lower_count`
     (`src/semviews/certify/bounds.py`): the largest (smallest) number of positives `K` in a
     region of `N` rows for which the observed count has tail probability above `alpha`.
   * `without_replacement=False`: draws are uniform with replacement, and the bounds are
     one-sided Clopper-Pearson rate bounds (`cp_upper`, `cp_lower`) scaled by the region size.
4. The result is `ux`, an upper bound on the positives in X, and `la`, a lower bound on the
   positives in A. Both hold together with probability at least `1 - delta` (union bound).

### Step 5: close the gap (`_check`)

The planner works with predictions, so the certified condition may fail. `_check` computes, for
the current state:

* `exact_pos`: known positives in the universe outside A (known block, O, and positives found in X);
* `d_x`, `unk_x`: known positives and still-unknown rows in X;
* `e_a`: known negatives in A; `la_eff = max(la, known positives in A)`;
* `T = exact_pos + la_eff` (a lower bound on true positives returned);
* `missed = min(max(ux - d_x, 0), unk_x)` (an upper bound on positives not returned);
* certified recall `T / (T + missed)` and certified precision `T / (exact_pos + |A| - e_a)`.

While recall is not certified, the operator labels the next `close_chunk` unknown rows at the
*top* of X (highest score first); each positive found is returned and lowers `missed`. While
precision is not certified, it labels the next `close_chunk` unknown rows at the *bottom* of A;
each negative found is dropped from the answer. All gap-closing labels have purpose `close`. When
both regions are fully labeled the answer is exact, so the loop always terminates with both
conditions met.

### Step 6: answer and register the view (`_finish`)

The answer is every known positive plus every row of A not known to be negative. The view
registered for the predicate has every label obtained in this run (and before) with provenance
`ORACLE`, the still-unknown rows of A as derived `1`, the still-unknown rows of X as derived `0`,
and everything else unknown. The view's `meta` records the targets.

### Why validity does not depend on view quality

The certified statement in `_check` is monotone in the two unknown quantities it bounds: recall
and precision both increase with the true number of positives in A, and recall decreases with the
true number of positives in X. So on the event that `ux` and `la` both cover (probability at
least `1 - delta`), the computed lower bounds on recall and precision are valid for **every**
state the algorithm can stop in, including states reached by data-dependent gap closing.

Views, helper scores, embeddings, shaping labels, and refit labels influence only the score,
hence only where the X/O/A boundaries fall and how many samples are planned. Those choices are
all made before the certification draws, and the draws are uniform within the fixed regions, so
the bounds are exact for whatever regions were chosen. A misleading or adversarial view therefore
moves the boundaries to bad places and costs oracle calls (larger O, more gap closing), never
correctness. The paper's validity theorem states this formally.

### Tests that check this

| Test | What it checks |
|------|----------------|
| `tests/test_certify.py::test_cp_one_sided_coverage` | each Clopper-Pearson bound covers at rate `>= 1 - alpha` (10,000 trials) |
| `tests/test_certify.py::test_hypergeometric_count_coverage` | the hypergeometric count bounds cover at rate `>= 1 - alpha` and are never looser than scaled Clopper-Pearson |
| `tests/test_certify.py::test_simultaneous_coverage_bonferroni` | Bonferroni-split bounds hold jointly |
| `tests/test_certify.py::test_milp_matches_brute_force`, `test_theorem1_end_to_end_simulation` | the strata decision procedure (used by `ReuseFilter`) is optimal and valid |
| `tests/test_region_filter.py::test_validity_over_seeds` | end to end, `CertifiedFilter` misses a target on at most a `delta` share of (seed, query) pairs on a synthetic tape |
| `tests/test_region_filter.py::test_leakage` | flipping every tape label the operator never paid for leaves its output unchanged |
| `tests/test_region_filter.py::test_views_save_calls` | views reduce calls on the synthetic family |
| `tests/test_reuse_filter.py` | the same three properties for `ReuseFilter` |

The synthetic tapes come from `src/semviews/workloads/synthetic.py`: latent binary attributes,
predicates that are attributes, unions, and intersections, with a small label-flip rate so that
containment holds only approximately; helper scores are noisy logits of the labels.

Note that the end-to-end simulation in `tests/test_region_filter.py` runs `CertConfig()`
defaults (draws with replacement, no refit rounds). The paper's configuration is covered at the
bound level by `test_hypergeometric_count_coverage` and at the integration level by
`tests/test_lotus_contract.py`; its end-to-end coverage on the real tapes is measured by the
validity experiment (`E3` in `experiments/run_all.py`).

---

## 3. Configuration

### `CertConfig` fields

`CertConfig` is a dataclass in `src/semviews/operators/region_filter.py`. The paper's operator is
`CertConfig` defaults overridden by the `operator:` block of
`configs/experiments/sprint.yaml`. That block is loaded once into `OPERATOR` in
`src/semviews/workloads/runner.py` (lists become tuples) and is also what
`semviews.adapters.lotus.optimizer.paper_config()` returns.

| Field | Default | Paper | Meaning |
|-------|---------|-------|---------|
| `use_views` | `True` | `True` | use catalog views as features |
| `use_helper` | `True` | `True` | use the helper score as a feature |
| `n_candidates` | `10` | `10` | views kept by embedding similarity |
| `shape_n` | `200` | `200` | shaping sample size |
| `shape_frac_max` | `0.08` | `0.08` | cap on the shaping sample as a share of unknown rows |
| `n_grid` | `(100, 200, 400, 800, 1600)` | same | candidate certification sample sizes |
| `q_grid` | `40` | `40` | number of score-quantile intervals for region boundaries |
| `close_chunk` | `50` | `50` | rows labeled per gap-closing step |
| `l2` | `1.0` | `1.0` | passed as scikit-learn `C` (inverse regularization strength) |
| `view_selection` | `"embed"` | `"embed"` | `"embed"`: top `n_candidates` by text similarity; `"all"`: every view |
| `shrink` | `0.0` | `0.0` | pseudo-count shrinking `p_hat` toward the base rate |
| `use_rowemb` | `True` | `True` | append the row-feature matrix |
| `active_rounds` | `0` | **`8`** | refit rounds before the regions are fixed |
| `active_chunk` | `250` | **`50`** | floor of the refit batch |
| `active_chunk_frac` | `0.0` | **`0.05`** | refit batch as a share of the filtered rows (250 at 5,000 rows) |
| `without_replacement` | `False` | **`True`** | hypergeometric certification without replacement |
| `plan_margin` | `0.0` | **`0.5`** | planner's pessimism, in standard deviations |
| `delta_splits` | `(0.5,)` | **`(0.5, 0.7, 0.9)`** | candidate shares of `delta` for X |
| `calibrate_uniform` | `False` | `False` | Platt recalibration on the shaping labels |

The same file fixes the experiment targets (`precision 0.9, recall 0.9, delta 0.05`), the
dataset-to-workload map, the number of workload orders and validity seeds, and the method lists.

### Method names and variants

`Bench.run` in `src/semviews/workloads/runner.py` maps method names to implementations:

* Any name in `CERT_VARIANTS` builds `CertifiedFilter` with
  `CertConfig(**{**OPERATOR, **CERT_VARIANTS[name], **cfg_overrides})`:

  | Method | Change on top of the paper's operator |
  |--------|----------------------------------------|
  | `semviews` | none (the paper's operator) |
  | `semviews_v1` | the earlier operator: `active_rounds=0, active_chunk_frac=0.0, without_replacement=False, plan_margin=0.0, delta_splits=(0.5,)` |
  | `semviews_noviews` | `use_views=False` (same operator, no views) |
  | `semviews_nohelper` | `use_helper=False` |
  | `semviews_noemb` | `use_rowemb=False` |
  | `semviews_noviews_noemb` | `use_views=False, use_rowemb=False` |
  | `semviews_allviews` | `view_selection="all"` |
  | `semviews_shape100`, `semviews_shape400` | `shape_n=100` / `400` |
  | `semviews_shrink50` | `shrink=50.0` |
  | `semviews_c10`, `semviews_c01` | `l2=10.0` / `0.1` |

* Any other `semviews*` name in `SEMVIEWS_VARIANTS` builds the per-signature strata operator
  `ReuseFilter` with a `ReuseConfig` (`semviews_strata` is its default configuration; the others
  change `use_views`, `use_helper`, `decision`, `adaptive`, `k_views`, or `selection`).
* A `_ideal` suffix (for example `semviews_ideal`) passes views chosen by mutual information on
  the *true* labels (`Bench.ideal_views`); it is an analysis upper bound, not a method.
* `bargain_views` is handled in `Bench._bargain_views` (Section 6).
* Every other name is looked up in `semviews.baselines.METHODS`.

`experiments/improvements.py` evaluates further `CertConfig` overrides relative to `V1`; its
results are written to separate files and do not change the paper's configuration.

### The strata variant (`ReuseFilter`)

`src/semviews/operators/reuse_filter.py` holds an earlier operator kept as the `semviews_strata`
ablation. It selects up to `k_views` views by greedy conditional mutual information on a paid
pilot sample (`relate.select_views`), stratifies unknown rows by the joint signature of those
views' labels (split further by helper-score quantiles, at most `max_strata` strata, strata
smaller than `min_stratum` labeled outright), draws with replacement on a fixed look schedule
(`looks`) with a Bonferroni split of `delta` over 2 sides x strata x looks (weighted by
`sqrt(size)` by default), and assigns each stratum accept / reject / oracle with the exact
branch-and-bound solver `certify.decide_exact` (or `decide_greedy`). An optional settle phase
labels oracle strata one at a time and re-plans with their exact counts. The validity argument
is the same: strata and Bonferroni weights are fixed before the draws.

---

## 4. Oracle, tape, replay, and cost accounting

### Tapes

Experiments never call a model. Every (predicate, row) label of the oracle was recorded once into
a **tape**, and methods replay it.

* Oracle and helper tapes are Parquet files `data/tapes/<dataset>__<model>[__<tag>].parquet`,
  assembled from per-predicate part files `data/tapes/parts/<dataset>__<model>[__<tag>]/<id>.parquet`
  (the `rw` tag holds reworded predicates). Columns include `dataset, row_id, predicate_id, model,
  model_id, prompt_version, raw_output, parse_ok, tokens_in, tokens_out, latency_ms, timestamp,
  sampling, label`, and `score` for helper tapes.
* `load_tape` (`src/semviews/oracle/tape.py`) reads the part files if present (otherwise the
  consolidated file) and keeps only predicates recorded for every row.
* LLM judge and rewording calls are taped in a SQLite file (`data/tapes/judge.sqlite`) by
  `TapedLM`, keyed by a hash of the model, the messages, and the sampling arguments.

`src/semviews/oracle/` is the only package that opens tape files;
`tests/test_boundaries.py::test_only_oracle_reads_tapes` checks that no other package outside
`adapters/` references `data/tapes`, and that `baselines/` never calls `ground_truth`.

### Replay oracle

`TapeOracle(tape, model, row_ids)` loads the tape into a dense (predicate x row) `int8` matrix with
per-entry token counts. `label(pid, rows, purpose=...)` returns the recorded labels and raises
`TapeMiss` if a pair is not on the tape. `ground_truth(pid)` returns a full column and is for the
experiment harness only: `Bench` reads it to score a result *after* the method returns and never
hands it to a method. `clone()` shares the label matrix but starts a fresh meter, so parallel
workload orders do not share charges.

`HelperScores(tape, row_ids, meter)` is the analogous replay of helper scores. Each `score` call
adds the number of rows scored to `meter.helper_calls`.

### Cost meter

`CostMeter` (`src/semviews/oracle/meter.py`) is the only source of cost numbers. It counts
`calls`, `tokens_in`, `tokens_out`, `helper_calls`, and `calls_by_purpose`, and derives `usd` from
per-token prices loaded from `configs/budgets.yaml` (`price_per_mtok`; the helper is priced at
zero).

What counts as a call: `TapeOracle.label` charges each **distinct (predicate, row) pair at most
once per meter scope**, adding the pair's recorded tokens. `meter.scope(name)` opens a scope;
`Bench.run` opens one scope per query. Asking for the same row twice within a query is free; a
label for the same predicate in a later query is not re-requested by `CertifiedFilter` because it
is already in the catalog with `ORACLE` provenance. Baselines that re-request labels already
bought in the same query (for example `bargain_with_scores`) are not charged twice.

Purposes are free-form tags on each request; the ones in use are:

| Purpose | Used by |
|---------|---------|
| `shape` | `CertifiedFilter` shaping sample; shaping sample in `bargain_views` |
| `residual` | `CertifiedFilter` refit rounds and the O region; `ReuseFilter` oracle strata; labeled rows of `inferred_reuse`, `semweave`, and the cascade's middle band |
| `sample` | certification draws (`CertifiedFilter`, `ReuseFilter`); LOTUS cascade importance sample |
| `close` | `CertifiedFilter` gap closing |
| `pilot`, `tiny` | `ReuseFilter` view-selection pilot and tiny strata |
| `baseline` | `oracle_all` |
| `train` | `proxy_lr` training sample |
| `bargain` | BARGAIN baselines |

Each result row written by `Bench.run` records `oracle_calls`, `oracle_calls_cert` (the sum of
the `sample`, `pilot`, and `train` purposes), `cum_oracle_calls`, `helper_calls`, `usd`, achieved
precision and recall, whether both targets were met, and provenance metadata (`git_commit`,
`config_hash`, `model_ids`, `lotus_version`).

### Import boundary

The algorithmic core does not depend on LOTUS. `tests/test_boundaries.py::test_core_does_not_import_lotus`
parses every file under `src/semviews/{oracle, catalog, relate, certify, operators}` and fails if
any imports `lotus`. LOTUS is imported by:

* `src/semviews/adapters/lotus/`: the optimizer, the taped LM wrappers, and the tape builders;
* `src/semviews/baselines/`: to run LOTUS's own cascade code (`importance_sampling`,
  `learn_cascade_thresholds`, `calibrate_llm_logprobs`) and `CascadeArgs` for the cascade
  baselines.

No LOTUS source is edited or patched; the adapter uses LOTUS's public optimizer extension point
and subclasses `lotus.models.LM`, overriding only the uncached-call path.
`tests/test_lotus_contract.py` pins `lotus-ai==1.2.4` and checks the extension points offline
with a keyword-matching fake LM.

---

## 5. LOTUS integration

### Usage

```python
import lotus
from lotus.ast import LazyFrame
from lotus.types import CascadeArgs
from semviews.catalog import Catalog
from semviews.adapters.lotus.optimizer import ViewOptimizer

catalog = Catalog("reviews", 0, path="reviews.semviews")   # loads the DuckDB file if it exists
args = CascadeArgs(precision_target=0.9, recall_target=0.9, failure_probability=0.05)
pipeline = LazyFrame().sem_filter("{text} reports a hardware defect", cascade_args=args)
result = pipeline.optimize([ViewOptimizer(catalog)]).execute(reviews_df)
catalog.save()                                              # persist the views
```

`ViewOptimizer(catalog, cfg=None, seed=0, embedder=None)` implements LOTUS's `BaseOptimizer`.
Its `optimize` replaces every node whose type is exactly `SemFilterNode` with a `ViewFilterNode`
that copies every `SemFilterNode` field and adds the catalog, the configuration, the seed, and a
shared `Embedder`. Other nodes are untouched.

### What `ViewFilterNode` does

When the node executes on a DataFrame (`ViewFilterNode.__call__` in
`src/semviews/adapters/lotus/optimizer.py`):

1. **Fallback.** If the node asks for explanations, raw outputs, few-shot examples, or a
   reasoning strategy, it runs LOTUS's native `SemFilterNode` path unchanged.
2. **Row identity.** Each row gets a content hash of the values of the columns the langex reads
   (`semviews.ids.row_id`), and `Catalog.ensure_rows` maps hashes to catalog positions,
   appending unseen rows as `UNKNOWN` in every existing view. Appended rows therefore need no
   view maintenance.
3. **Predicate identity.** The predicate id is `view_key(langex, prompt_version, model)`, a hash
   of the langex, the LOTUS `sem_filter` prompt version, and the oracle model. A change of oracle
   model or LOTUS version yields a new id, so old labels are never used exactly, but old views
   remain usable as features.
4. **Targets.** Taken from `cascade_args` (`precision_target`, `recall_target`,
   `failure_probability`) when a recall target is set; otherwise `Targets()` defaults.
5. **Helper.** If `cascade_args.proxy_model` is `HELPER_LM` and `lotus.settings.helper_lm` is
   configured, `LotusHelperScores` scores the input rows once with LOTUS's `sem_filter` and
   log-probabilities.
6. **Oracle.** `LotusOracle` labels requested rows with LOTUS's own `sem_filter` on
   `lotus.settings.lm`, passing the node's `default`, `system_prompt`, and `output_tokens`. Its
   meter counts calls and purposes (not tokens).
7. **Operator.** With `cfg=None` (the default), `CertifiedFilter` runs with `paper_config()` and
   row features computed by `row_features` (all-MiniLM-L6-v2 embeddings of the langex columns,
   PCA to at most 32 dimensions, scaled as in the experiments). Passing a `ReuseConfig` selects
   `ReuseFilter` instead; passing any `CertConfig` uses it as given. The run is restricted to the
   input rows via `universe`.
8. **Output.** Returns the kept rows (or, with `return_all`, all rows with a boolean column), and
   stores diagnostics including oracle calls by purpose in `node.last_stats`. The query's view is
   registered in the catalog.

`NoOpOptimizer` in the same file returns the node list unchanged; it exists to show that the
extension point works without edits to LOTUS.

### Catalog persistence

`Catalog(table, n_rows, path)` loads `path` if it exists. `Catalog.save(path=None)` writes a
DuckDB file with four tables: `info` (table name, row count), `rows` (content-hash row ids in
catalog order), `labels` (only known entries: `predicate_id, row, label, prov`), and `views`
(`predicate_id, text, model, meta` as JSON). Nothing saves automatically; the caller decides when
to persist. In memory a view costs two bytes per row (label and provenance).

---

## 6. Baselines

All baselines live in `src/semviews/baselines/__init__.py` and receive a shared `Ctx` (oracle,
catalog, predicate texts, helper, predicate embeddings, row embeddings, judged relations). They
see labels only through `Ctx.oracle.label`, and most register a view (oracle labels as `ORACLE`,
the rest of the answer as `DERIVED`) so that later reuse methods can find it.

| Method | What the code does |
|--------|--------------------|
| `oracle_all` | Labels every row with the oracle and returns the positives. |
| `lotus_cascade` | `cascade_with_scores` on the taped helper scores: LOTUS's `calibrate_llm_logprobs`, `importance_sampling` (default 10% sample), and `learn_cascade_thresholds`; rows at or above the upper threshold are accepted, rows between the thresholds are labeled. |
| `lotus_cascade_full` | Same, with the importance sample allowed to range over all rows (`cascade_IS_max_sample_range = n`). |
| `lotus_strict`, `lotus_strict2` | The cascade asked for stricter internal targets (+0.05 with `delta=0.01`, or +0.08 with `delta=0.001`), to test whether stricter settings make it valid. |
| `exact_cache` | If a previously answered predicate has identical text, return its stored labels; otherwise `lotus_cascade`. |
| `sim_cache@0.7/0.8/0.9` | If the most similar past predicate (cosine of predicate-text embeddings) is at least `tau`, return its labels unchecked; otherwise `lotus_cascade`. |
| `sim_proxy` | Cascade proxy score `0.5 * [nearest view says 1] + 0.5 * helper + 0.25 * [nearest view unknown]`, then the LOTUS cascade. |
| `proxy_lr` | Labels a uniform 5% sample (at least 50 rows), fits logistic regression on full row embeddings, and uses its probabilities as the cascade proxy. |
| `bargain_pr` | BARGAIN's `BARGAIN_PR` (pinned release) with taped helper scores as the proxy. It takes one target, so `max(precision, recall)` is passed. |
| `bargain_views` | (in `Bench._bargain_views`) Pays a shaping sample, fits the same logistic score `CertifiedFilter` would (views, helper, row PCA, `CertConfig()` defaults), and gives that score to `BARGAIN_PR` via `bargain_with_scores`. Isolates the certification step. |
| `inferred_reuse` | Uses judged relations without certification: for the strongest judged relation to a past view, *equivalent* returns its labels, *q implies p* labels only p's positives, *p implies q* accepts p's positives and labels the rest; otherwise `lotus_cascade`. |
| `semweave` | A reimplementation of SemWeave's reuse rule: among the top-5 most similar past predicates, use judged relations (equivalent, implied-by, implies, exclusive) to decide rows from stored results, after dropping exclusions that contradict an implication chain; every undecided row goes to the oracle. No targets, no certification. |

Judged relations are produced once by `src/semviews/adapters/lotus/judge.py` (one judge call per
unordered predicate pair, stored in both directions in `data/relations/<dataset>.json`) and loaded
by `experiments/run_all.py::relations`. The operator itself never uses them.

---

## 7. Module map

| Path | Contents |
|------|----------|
| `src/semviews/operators/region_filter.py` | `CertConfig`, `CertifiedFilter` (the paper's operator) |
| `src/semviews/operators/reuse_filter.py` | `Targets`, `FilterResult`, `ReuseConfig`, `ReuseFilter` (strata variant) |
| `src/semviews/certify/bounds.py` | `cp_lower`, `cp_upper`, `hg_upper_count`, `hg_lower_count`, `bonferroni_alpha`, `required_n_for_upper` |
| `src/semviews/certify/decide.py` | accept/reject/oracle per stratum: `plan_bounds`, `decide_exact` (branch and bound), `decide_greedy` |
| `src/semviews/oracle/tape.py` | `Oracle` protocol, `TapeOracle`, `HelperScores`, `load_tape`, `load_prices`, `TapeMiss` |
| `src/semviews/oracle/meter.py` | `CostMeter` with charging scopes |
| `src/semviews/catalog/__init__.py` | `View`, `Catalog`, provenance codes, DuckDB persistence |
| `src/semviews/relate/__init__.py` | `Embedder` (cached local embeddings), `predicate_text`, `mutual_information`, `select_views` |
| `src/semviews/ids.py` | `row_id` (content hash), `view_key`, prompt-version template |
| `src/semviews/baselines/__init__.py` | baselines and the `METHODS` registry |
| `src/semviews/workloads/datasets.py` | builds the four datasets (`BUILDERS`); writes the external PubMed predicate pool at runtime |
| `src/semviews/workloads/runner.py` | `Bench` replay harness, `OPERATOR`, `V1`, `CERT_VARIANTS`, `SEMVIEWS_VARIANTS` |
| `src/semviews/workloads/synthetic.py` | synthetic tapes for tests |
| `src/semviews/adapters/lotus/optimizer.py` | `ViewOptimizer`, `ViewFilterNode`, `LotusOracle`, `LotusHelperScores`, `paper_config`, `row_features`, `NoOpOptimizer` |
| `src/semviews/adapters/lotus/taped_lm.py` | `RecordingLM`, `TapedLM` (SQLite record and replay), `proxy_lm` |
| `src/semviews/adapters/lotus/build_tape.py` | records oracle and helper tapes with LOTUS's `sem_filter` prompt; resumable; enforces the spend cap |
| `src/semviews/adapters/lotus/fast_helper.py` | faster helper-tape recorder with LOTUS's prompt formatter and score semantics |
| `src/semviews/adapters/lotus/judge.py` | relation judge over predicate pairs |
| `src/semviews/adapters/lotus/rephrase.py` | rewords a seeded sample of predicates with the judge model |
| `src/semviews/adapters/lotus/noise_check.py` | re-labels taped pairs live to measure oracle self-agreement |
| `experiments/run_all.py` | the experiment driver (`--only E2 E3 ...`) |
| `experiments/sessions.py`, `experiments/rephrased.py` | exploration-session and reworded-question workloads |
| `experiments/headroom.py`, `experiments/improvements.py` | headroom and scale studies; opt-in operator variants |
| `experiments/e1_relations.py`, `experiments/e1_predictability.py` | statistical containment and predictability analyses |
| `experiments/e8_live.py` | one live LOTUS run with and without `ViewOptimizer`, through `TapedLM` |
| `experiments/figures.py` | regenerates figures, tables, and the claims file from result Parquet files |
| `configs/` | models, budgets, predicate pools, frozen experiment configuration |
| `tests/` | bounds, operators, LOTUS contract, repository boundaries |

---

## 8. Extension points

### Adding a dataset

1. Add a builder to `BUILDERS` in `src/semviews/workloads/datasets.py` that returns a DataFrame
   with a `text` column (and any human labels); `_finish` normalizes whitespace, truncates,
   de-duplicates, samples `n` rows with the seed, and adds the content-hash `row_id`. Run
   `python -m semviews.workloads.datasets <name>` to write `data/datasets/<name>.parquet`.
2. Write a predicate pool `configs/predicates/<name>.yaml` (next section).
3. Record tapes through the local proxy:
   `python -m semviews.adapters.lotus.build_tape --dataset <name> --model oracle-llama` and
   `--model helper` (or `python -m semviews.adapters.lotus.fast_helper --dataset <name>`), then
   `--consolidate`. For the relation-based baselines, run
   `python -m semviews.adapters.lotus.judge --dataset <name> --record "<noun>"`.
4. Add the dataset to `datasets:` in an experiment configuration; `Bench(<name>)` picks up the
   dataset, tapes, pool, and relations by name. Row embeddings are cached on first use in
   `data/cache/rowemb_<name>.npy`.

### Adding a predicate pool

A pool is YAML with `dataset`, `column`, and a `predicates` list of `{id, langex, ...}` entries,
where `langex` uses LOTUS placeholders (`{text}`). Predicate ids become `<dataset>/<id>`. A second
pool for the same dataset (such as `reviews_rephrased.yaml`) is recorded with
`build_tape --pool <pool> --tag <tag>` into a separate tape; `Bench(..., rephrased=True)` shows
how such a tape and pool are merged with the main one. `Bench` uses only predicates present on
the oracle tape and, when a helper tape exists, on the helper tape.

### Adding a method or baseline

* A **baseline** is a function `f(ctx, pid, targets, rng) -> FilterResult`. Obtain every label
  through `ctx.oracle.label(pid, rows, purpose=...)`, register the result with `_register` if
  later methods should see it, and add it to `METHODS`. It is then available by name in
  `experiments/run_all.py --methods ...`.
* An **operator variant** that is a configuration change is an entry in `CERT_VARIANTS` (or
  `cfg_overrides` / `cfg` in a job), which is applied on top of the paper's `OPERATOR`.
* A **new operator** should follow the `CertifiedFilter` contract: take an object with the
  `Oracle` protocol and a `Catalog`, return `FilterResult`, register its view with provenance,
  and keep anything that enters a bound to fresh uniform oracle draws inside regions fixed
  before the draws. Add a validity-over-seeds and a leakage test modeled on
  `tests/test_region_filter.py`, and keep it free of LOTUS imports.
* To expose a new operator through LOTUS, extend `ViewFilterNode.__call__`; the rest of the
  LOTUS plan stays unchanged.
