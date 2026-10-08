# Data

This document describes every data file shipped with the repository and every external source
the code reads. All experiments replay recorded model outputs ("tapes"), so the paper's numbers
can be recomputed without any model access.

| Directory | Contents |
|---|---|
| `data/datasets/` | The four 5,000-row tables the experiments filter, and two 100,000-row tables for the appendix scale check (E13) |
| `configs/predicates/` | Predicate pools (the natural-language filter conditions) |
| `data/relations/` | LLM-judge logical relations between predicate pairs |
| `data/tapes/` | Recorded oracle labels and helper scores, plus two SQLite response caches |
| `experiments/results/` | Per-query result rows of every experiment |
| `configs/models.yaml`, `configs/budgets.yaml` | Model aliases, sampling settings, prices |

Not shipped: `data/raw/` (raw upstream downloads), `data/cache/` (sentence embeddings, recomputed
on first use with `sentence-transformers/all-MiniLM-L6-v2`), and `configs/predicates/pubmed.yaml`
(see [W1](#w1-pubmed-external-pool-from-scaledoc)).

---

## 1. Datasets (`data/datasets/*.parquet`)

Built by `python -m semviews.workloads.datasets` (`src/semviews/workloads/datasets.py`). Each table
is a seeded sample from a public source:

1. Whitespace runs are collapsed and text is truncated to `max_chars` characters.
2. Rows with fewer than 20 characters are dropped, then duplicate texts are dropped.
3. `n = 5000` rows are drawn with `DataFrame.sample(random_state=20261001)`.
4. `row_id` is prepended: the first 16 hex characters of the SHA-256 of the JSON list `[text]`
   (`semviews.ids.row_id`). Row ids are unique within each table.

| File | Rows | Upstream source (Hugging Face) | Config / split | `max_chars` | Columns |
|---|---|---|---|---|---|
| `pubmed.parquet` | 5,000 | [`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization) | `section`, `test`; the `abstract` field only | 1,200 | `row_id`, `text` |
| `goemotions.parquet` | 5,000 | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | `simplified`, `train` | 700 | `row_id`, `text`, `emotions` |
| `dbpedia.parquet` | 5,000 | [`DeveloperOats/DBPedia_Classes`](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes) | `test` | 700 | `row_id`, `text`, `l1`, `l2`, `l3` |
| `reviews.parquet` | 5,000 | [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | file `raw/review_categories/Electronics.jsonl`, first 60,000 lines | 700 | `row_id`, `text`, `rating` |

Column meanings:

| Column | Type | Meaning |
|---|---|---|
| `row_id` | string | 16-hex content hash of the text (see above). Joins to the tapes. |
| `text` | string | The record the predicates are evaluated on. |
| `emotions` | string | GoEmotions only. JSON list of the sorted human emotion labels of the comment, e.g. `["disapproval"]`. |
| `l1`, `l2`, `l3` | string | DBpedia only. Human class labels at three levels of the DBpedia ontology, e.g. `Event` / `SportsEvent` / `FootballMatch`. |
| `rating` | float | Reviews only. Star rating (1 to 5) from the source. |

Dataset-specific preparation:

- **Reviews.** The builder reads a local copy of the first 60,000 lines of the upstream
  `Electronics.jsonl` from `data/raw/electronics_head.jsonl` (not shipped). Reviews whose body is
  shorter than 80 characters are skipped; `text` is `"<title>. <body>"` (or the body when the
  title is empty). Only `text` and `rating` are kept; no user, product, or time fields.
- **PubMed.** Abstracts only, so `text` is longer (mean 1,022 characters versus 73 for GoEmotions,
  449 for DBpedia and 374 for Reviews).

### Large relations for the scale check (E13)

Built by `experiments/scale_real.py` (`data` and `reviews` subcommands) with the same cleaning
and `row_id` as above, seed 20261001:

| File | Rows | Source | Disjoint from the 5,000-row table because |
|---|---|---|---|
| `dbpedia_100k.parquet` | 100,000 | `DeveloperOats/DBPedia_Classes`, `train` split (240,942 rows) | the 5,000-row table comes from the `test` split |
| `reviews_100k.parquet` | 100,000 | first 250,000 lines of the same `Electronics.jsonl` (`data/raw/electronics_head250k.jsonl`, not shipped; its first 60,000 lines equal `electronics_head.jsonl`) | every text of `reviews.parquet` is removed before sampling |

Both have the columns of their 5,000-row counterparts. Row embeddings are cached in
`data/cache/rowemb_<name>.npy` (not shipped; recomputed on first use).

### Upstream licences

| Source | Licence | How verified |
|---|---|---|
| GoEmotions (`google-research-datasets/go_emotions`) | Apache-2.0 | Dataset card metadata `license: apache-2.0`, and the card's licensing section says the GitHub repository that houses the dataset has an Apache License 2.0. https://huggingface.co/datasets/google-research-datasets/go_emotions |
| DBpedia Classes (`DeveloperOats/DBPedia_Classes`) | CC0-1.0 | Dataset card metadata `license: cc0-1.0`. The card says the data comes from Wikipedia via the DBpedia project. https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes |
| Amazon Reviews 2023 (`McAuley-Lab/Amazon-Reviews-2023`) | The card states no licence. See the upstream dataset card. | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| PubMed summarization (`ccdv/pubmed-summarization`) | The card states no licence. See the upstream dataset card. The card says it was adapted from https://github.com/armancohan/long-summarization. | https://huggingface.co/datasets/ccdv/pubmed-summarization |

The excerpts are provided only so the experiments can be reproduced. Before reusing them, check
the upstream terms.

---

## 2. Predicate pools (`configs/predicates/*.yaml`)

Each file has a header (`dataset`, `column`, and for GoEmotions also `human_label_column:
emotions` and `human_label_format: json_list`) and a `predicates` list. Each predicate has:

| Field | Meaning |
|---|---|
| `id` | Short id. Across the code and data a predicate is named `<dataset>/<id>`, e.g. `goemotions/anger`. |
| `langex` | LOTUS language expression; `{text}` is replaced by the record. This is the prompt condition passed to LOTUS `sem_filter`. |
| `level` | Taxonomy level for W3 pools (0 = coarsest); `null` when the predicate is not taken from a taxonomy. |
| `human` | W3 only: mapping from a dataset column to the human label values that make the predicate true (true if any listed value is present), e.g. `{l3: [SoccerPlayer]}`. `null` when there is no human label. Used only for secondary analyses (e.g. `human_eps` in E1), never as ground truth for the guarantees. |
| `rephrase_of` | Reworded pools only: the `id` of the original predicate. |

| File | Predicates | Workload | Levels | With `human` |
|---|---|---|---|---|
| `pubmed.yaml` (generated, not shipped) | 25 | W1 | all `null` | 0 |
| `goemotions.yaml` | 40 | W3 | 0: 3, 1: 7, 2: 27, `null`: 3 | 37 |
| `dbpedia.yaml` | 40 | W3 | 1: 8, 2: 22, 3: 10 | 40 |
| `reviews.yaml` | 40 | W2a | all `null` | 0 |
| `goemotions_rephrased.yaml` | 15 | WR | taken from the original | 0 |
| `reviews_rephrased.yaml` | 15 | WR | taken from the original | 0 |
| `dbpedia_100k.yaml` | 8 | W3 (E13) | taken from the original | 8 |
| `reviews_100k.yaml` | 8 | W2a (E13) | all `null` | 0 |

The frozen experiment configuration (`configs/experiments/sprint.yaml`, key `datasets`) assigns
the workloads: `pubmed: W1`, `goemotions: W3`, `dbpedia: W3`, `reviews: W2a`.

### W1: PubMed, external pool from ScaleDoc

This is the predicate pool that ScaleDoc released for PubMed, used verbatim. The upstream
repository (https://github.com/Seurgul/ScaleDoc) has no licence: the GitHub API reports
`license: null`. For that reason the pool is **not redistributed**. Fetch it at runtime:

```bash
python -m semviews.workloads.datasets --pubmed-pool
```

This calls `scaledoc_pubmed_predicates()` in `src/semviews/workloads/datasets.py`, which downloads
`https://raw.githubusercontent.com/Seurgul/ScaleDoc/main/dataset/query.json` and writes
`configs/predicates/pubmed.yaml`. The ids are:

- `q00`–`q19`: the 20 questions under the `pubmed` key.
- `ext01`–`ext05`: the 5 questions under the `pubmed_ext` key. The placeholder question
  `"This is a random question."` is dropped.

Each question becomes the langex `"Question about the document in {text}: <question>"`. The
function prints the SHA-256 of `[[id, langex], ...]` and compares it with the hash of the pool the
experiments used (`POOL_SHA256` in the same file). A mismatch means the upstream file changed and
the shipped PubMed tapes no longer correspond to it. The PubMed tapes and relations still name
predicates only by id (`pubmed/q04`), so replaying the experiments does not need the pool text.

### W2a: Reviews, analyst-style pool

These are 40 conditions an analyst might apply to electronics reviews: battery complaints,
defects, shipping damage, wrong items, customer service, and similar. They were written for this
study with AI assistance and do not come from an external source. They have no human labels.

### W3: GoEmotions and DBpedia, taxonomy-derived pools with human labels

- **GoEmotions.** Predicates follow the dataset's own emotion taxonomy:
  - level 2: the 27 fine-grained emotions, e.g. `anger`, `admiration`;
  - level 1: the Ekman groups (`ekman_anger`, `ekman_joy`, …) plus `neutral`;
  - level 0: the sentiment groups (`sent_positive`, `sent_negative`, `sent_ambiguous`).

  Each predicate's `human` field lists the fine labels it covers. Three extra predicates with no
  human label are added: `is_question`, `sarcastic`, `mentions_person`.
- **DBpedia.** Predicates are classes from the three-level DBpedia ontology carried in `l1`, `l2`
  and `l3`, e.g. `agent` (`l1: Agent`) or `soccer_player` (`l3: SoccerPlayer`).

Because these pools come from a taxonomy, the human labels give the true containment between
predicates.

### WR: reworded predicates (`*_rephrased.yaml`)

`python -m semviews.adapters.lotus.rephrase --dataset <d> --n 15` (seed 0) samples 15 predicates
from a pool and asks the `judge` model, at temperature 0, to reword each one without changing its
meaning. The call goes through the `judge.sqlite` cache. The reworded predicates get ids
`rw_<id>`, keep the original `level`, and record `rephrase_of`. Because they are labeled into
separate `__rw` tapes, the original tapes are unchanged. `experiments/rephrased.py` issues each
pool together with its rewordings as workload `WR`.

### E13: predicates for the 100,000-row relations (`*_100k.yaml`)

Eight predicates of `dbpedia.yaml` and of `reviews.yaml`, chosen with
`numpy.random.default_rng(20261006).choice(40, 8, replace=False)`, text unchanged. Only these are
labeled by the oracle on the 100,000-row tables (tapes `dbpedia_100k__oracle-llama.parquet`,
`reviews_100k__oracle-llama.parquet`). E13 also runs all 40 DBpedia predicates on
`dbpedia_100k.parquet` with the human ontology labels (through each predicate's `human` mapping)
as the oracle; that run reads no tape.

### Other workload codes found in result files

| Code | Meaning |
|---|---|
| `W0` | Unrelated control (E9). From predicates with oracle selectivity in [0.02, 0.6], 8 are chosen greedily to minimize the largest pairwise \|phi\| correlation (`w0_subset` in `experiments/run_all.py`). |
| `WS` | Exploration sessions (`experiments/sessions.py`). The same pools, issued in sessions of 8; each next predicate is drawn from the 3 unused predicates whose text is most similar. |
| `W` | Tuning runs of opt-in operator settings (`IMP_*` files). |

---

## 3. Predicate relations (`data/relations/*.json`)

Produced by `python -m semviews.adapters.lotus.judge --dataset <d>`
(`src/semviews/adapters/lotus/judge.py`). The script makes one `judge` call per unordered pair of
predicates in a pool. The system prompt asks which logical relationship holds between two yes/no
conditions A and B on records of a given kind ("social media comment", "encyclopedia entry",
"product review", "biomedical paper abstract"). The reply must be exactly one of `equivalent`,
`A_implies_B`, `B_implies_A`, `exclusive`, `overlapping`, `unrelated`. Replies are mapped to
`equivalent`, `implies`, `implied_by`, `exclusive`, `overlapping`, `unrelated`. A reply matching
no label is mapped by substring, and defaults to `unrelated`. Calls are cached in
`data/tapes/judge.sqlite`.

Format: a flat JSON object. The key `"<dataset>/<A>||<dataset>/<B>"` holds the relation of A to B,
so `implies` means A ⇒ B. Both directions are stored, with `implies` and `implied_by` swapped in
the reverse key.

| File | Keys (ordered pairs) | Contents |
|---|---|---|
| `pubmed.json` | 600 | all pairs of the 25-predicate pool |
| `goemotions.json` | 1,560 | all pairs of the 40-predicate pool |
| `dbpedia.json` | 1,560 | all pairs of the 40-predicate pool |
| `reviews.json` | 1,560 | all pairs of the 40-predicate pool |
| `goemotions_rephrased.json` | 1,410 | pairs involving at least one reworded predicate (`--extra goemotions_rephrased`) |
| `reviews_rephrased.json` | 1,410 | pairs involving at least one reworded predicate (`--extra reviews_rephrased`) |

`experiments/run_all.py` loads these relations for the `inferred_reuse` baseline.
`experiments/e1_relations.py` compares them with containment measured on the oracle tape (E1).
The SemViews operator does not use them.

---

## 4. Tapes (`data/tapes/`)

### 4.1 Parquet tapes: `<dataset>__<alias>[__rw].parquet`

Each row is one model call: one predicate evaluated on one table row. Oracle tapes are recorded
by `python -m semviews.adapters.lotus.build_tape --dataset <d> --model <alias>`. Helper tapes are
recorded by `build_tape --model helper` or by the faster equivalent
`python -m semviews.adapters.lotus.fast_helper --dataset <d>`. Both use LOTUS's own `sem_filter`
prompt (lotus-ai 1.2.4). The suffix `__rw` marks the tapes of the reworded pools
(`--pool <d>_rephrased --tag rw`).

| File | Rows | Predicates | Table rows | `score` |
|---|---:|---:|---:|---|
| `pubmed__oracle-llama.parquet` | 125,000 | 25 | 5,000 | no |
| `pubmed__helper.parquet` | 125,000 | 25 | 5,000 | yes |
| `goemotions__oracle-llama.parquet` | 200,000 | 40 | 5,000 | no |
| `goemotions__oracle-gptoss.parquet` | 40,000 | 40 | 1,000 (the first 1,000 rows of the table) | no |
| `goemotions__helper.parquet` | 200,000 | 40 | 5,000 | yes |
| `goemotions__oracle-llama__rw.parquet` | 75,000 | 15 | 5,000 | no |
| `goemotions__helper__rw.parquet` | 75,000 | 15 | 5,000 | yes |
| `dbpedia__oracle-llama.parquet` | 200,000 | 40 | 5,000 | no |
| `dbpedia__helper.parquet` | 200,000 | 40 | 5,000 | yes |
| `reviews__oracle-llama.parquet` | 200,000 | 40 | 5,000 | no |
| `reviews__helper.parquet` | 200,000 | 40 | 5,000 | yes |
| `reviews__oracle-llama__rw.parquet` | 75,000 | 15 | 5,000 | no |
| `reviews__helper__rw.parquet` | 75,000 | 15 | 5,000 | yes |
| `dbpedia_100k__oracle-llama.parquet` | 800,000 | 8 | 100,000 | no |
| `reviews_100k__oracle-llama.parquet` | 800,000 | 8 | 100,000 | no |

Every oracle and helper tape is complete: each (predicate, row) pair of its pool and table
appears exactly once.

| Column | Type | Meaning |
|---|---|---|
| `dataset` | string | Dataset name. |
| `row_id` | string | Joins to `data/datasets/<dataset>.parquet`. |
| `predicate_id` | string | `<dataset>/<id>`. |
| `model` | string | Model alias (`oracle-llama`, `oracle-gptoss`, `helper`). |
| `model_id` | string | Model name reported in the response. Oracle tapes hold the alias. Helper tapes hold `helper` (rows recorded through LOTUS's LM) or `helper-local` (rows recorded by `fast_helper`). Both are the same local model with the same prompt and settings. |
| `prompt_version` | string | `lotus-1.2.4-sem_filter-default`: LOTUS's default `sem_filter` prompt in lotus-ai 1.2.4. |
| `raw_output` | string | The model's answer text, truncated to 64 characters, e.g. `Answer: False`. |
| `parse_ok` | bool | Whether `raw_output` contains `true` or `false` (case-insensitive). |
| `tokens_in`, `tokens_out` | int32 | Prompt and completion tokens reported for the call. Used to price oracle labels. |
| `latency_ms` | float32 | Call latency. With batched LOTUS calls this is the batch time divided by its size. |
| `timestamp` | string | UTC ISO time at which the chunk was recorded. |
| `sampling` | string | JSON of the sampling settings from `configs/models.yaml`. |
| `score` | float32 | Helper tapes only: the helper's probability that the answer is True (see below). |
| `label` | int8 | 1 if the predicate holds, 0 otherwise, as parsed by LOTUS. |

**Labels.** LOTUS's `filter_postprocess` lowercases the answer and sets the label to 1 if it
contains `true`, else to 0 if it contains `false`. Otherwise the label falls back to LOTUS's
default, which is True (1). Such rows have `parse_ok = false`. They are rare: the share of
unparsed rows is at most 0.6% on the helper tapes (GoEmotions, mostly refusals such as
`I cannot assess the …`) and at most 0.01% on the oracle tapes. The oracle tape's `label` is the
ground truth for all precision and recall numbers. The experiments compare against the oracle,
not against human labels.

**Helper `score`.** Helper calls request log-probabilities with `top_logprobs = 10`.
LOTUS's `LM.format_logprobs_for_filter_cascade` then goes through the answer tokens. At the first
token whose top-10 candidates include both `True` and `False`, it returns
`p(True) / (p(True) + p(False))`. If no token has both candidates, it returns 1 when `True` was
generated and 0 otherwise. Helper calls are free in the cost accounting.

**Access rule.** Only `semviews.oracle.tape` opens these files. Methods obtain labels through
`TapeOracle.label()`, which charges every distinct (predicate, row) pair once per query at the
pair's taped token counts. The full label column (`ground_truth`) is used only by the harness to
score results.

### 4.2 SQLite response caches: `judge.sqlite`, `e8_live.sqlite`

These are record-and-replay caches of full chat-completion responses, written by `TapedLM`
(`src/semviews/adapters/lotus/taped_lm.py`). Each holds one table:

```sql
CREATE TABLE tape (k TEXT PRIMARY KEY, response TEXT);
```

- `k` is the SHA-256 hex digest of `json.dumps([model, messages, {sorted call kwargs}],
  sort_keys=True)`. The call is identified by the model route, the exact message list, and the
  sampling arguments.
- `response` is the response serialized with `model_dump_json()`. On replay it is rebuilt as a
  `ModelResponse`. In replay-only mode a missing key raises an error, so a replay never makes a
  live call by accident.

| File | Entries | Written by |
|---|---:|---|
| `judge.sqlite` | 4,080 | `judge` calls of `semviews.adapters.lotus.judge` (pair relations) and `semviews.adapters.lotus.rephrase` (rewordings) |
| `e8_live.sqlite` | 15,000 | `oracle-llama` and `helper` calls of the live end-to-end LOTUS run, `experiments/e8_live.py` |

In the released files, a provider-specific warning field (the `system` key) has been removed from
the stored JSON responses. Replay does not read that field.

---

## 5. Result files (`experiments/results/`)

### 5.1 Two directories

- **`experiments/results/op2/`** holds the runs of the paper's operator, which uses the `operator`
  block of the frozen experiment configuration (`configs/experiments/sprint.yaml`).
- **`experiments/results/`** (top level) holds:
  - runs of the first-draft operator, before refit rounds and hypergeometric bounds;
  - the baseline runs;
  - the auxiliary analyses listed below.

`experiments/figures.py` (`load`) concatenates both directories. In memory it renames top-level
operator rows: `semviews` becomes `semviews_v1`, and other `semviews_*` variants (except
`semviews_strata`) get a `v1/` prefix. The files on disk are not modified.

### 5.2 Common per-query schema

Most files have one row per query issued (`experiments/run_all.py`, `semviews.workloads.runner`).
For example, `op2/E2__main.parquet` has 5,800 rows and these 27 columns:

| Column | Meaning |
|---|---|
| `experiment` | Experiment id (table below). |
| `workload` | Workload code (W0, W1, W2a, W3, WR, WS, W). |
| `dataset` | Dataset name. |
| `method` | Method id, e.g. `semviews`, `semviews_noviews`, `bargain_pr`, `lotus_cascade`, `sim_cache@0.8`, `semweave`, `oracle_all`. `LABEL` in `experiments/figures.py` gives the display name of each. |
| `seed` | Workload order / random seed. The query order is a shuffle seeded with `10000 + seed`. |
| `query_idx` | Position of the query in the workload order (0-based). |
| `predicate_id` | The query predicate, `<dataset>/<id>`. |
| `oracle_calls` | Distinct oracle labels paid for by this query. |
| `oracle_calls_cert` | The subset of `oracle_calls` drawn as certification or estimation samples (purposes `sample`, `pilot`, `train`). For SemViews these are the fresh uniform draws behind the confidence bounds. Shaping, refit, residual, and gap-closing labels are excluded. |
| `cum_oracle_calls` | Running sum of `oracle_calls` over the workload so far. |
| `helper_calls` | Helper scores used by this query (free). |
| `usd` | Cost of this query's oracle calls: taped tokens × the prices in `configs/budgets.yaml`. |
| `precision`, `recall` | Of the returned rows, against the full oracle label column (1.0 when the denominator is 0). |
| `precision_target`, `recall_target`, `delta` | Requested targets and failure probability (default 0.9 / 0.9 / 0.05). |
| `met` | `precision >= precision_target and recall >= recall_target` for this query. Validity is the share of runs in which this holds. |
| `n_rows` | Rows in the table. |
| `n_returned` | Rows returned by the filter. |
| `selectivity` | True share of positive rows (oracle). |
| `wall_ms` | Wall time of the method on the replayed tape. This is not live model latency. |
| `stats` | JSON object of method diagnostics (below). |
| `git_commit` | Short commit of the development history the run came from. It does not refer to a commit of this release. |
| `config_hash` | First 12 hex characters of SHA-256 over `[method, targets, config overrides]`. |
| `model_ids` | Oracle alias whose tape was replayed (`oracle-llama`, or `oracle-gptoss` for E7). |
| `lotus_version` | Installed lotus-ai version (1.2.4). |

Some files add a `variant` column that names the setting varied, e.g. `delta=0.01`,
`targets=0.95,0.8`, `rows=2500`, `coverage=0.5`, `adversarial`, `inverted`, `appended rows`,
`cross-model views`.

**`stats` for the SemViews operator** (`CertifiedFilter`, `src/semviews/operators/region_filter.py`):

| Key | Meaning |
|---|---|
| `views` | Predicate ids of the candidate views used as features. |
| `n_shape` | Size of the uniform shaping sample (labels are exact but never enter a bound). |
| `n_refit` | Oracle labels taken in the refit rounds before the regions were fixed. |
| `plan_x`, `plan_a`, `plan_o` | Sizes of the reject region X, accept region A, and oracle region O. |
| `n_x`, `n_a` | Planned certification sample sizes in X and A. |
| `close_steps` | Gap-closing steps taken after certification. |
| `ux` | Upper confidence bound on the number of positives in X. |
| `la` | Lower confidence bound on the number of positives in A. |

Baselines write their own keys.

### 5.3 Files

| Experiment | Script | Paper operator (`op2/`) | Top level |
|---|---|---|---|
| E1: containment, exclusion, judge agreement | `experiments/e1_relations.py` | none | `E1_pairs`, `E1_summary` |
| E1b: predictability of a predicate from the others' labels (3-fold AUROC) | `experiments/e1_predictability.py` | none | `E1_predictability` |
| E2: cumulative oracle calls per method | `run_all.py --only E2` | `E2__main`, `E2__abl` | `E2__<d>`, `E2__<d>__sw` (SemWeave), `E2__<d>__bv` (BARGAIN + our proxy) |
| E3: validity over many seeds | `run_all.py --only E3` | `E3__reused`, `E3__nv` | `E3__<d>`, `E3__<d>__sw` |
| E4: sensitivity (targets, delta, table size) | `run_all.py --only E4` | `E4__run` | `E4__<d>` |
| E5: partial views | `run_all.py --only E5` | `E5__run` | `E5__<d>` |
| E7: views from one oracle reused under another (GoEmotions, `oracle-gptoss`, 1,000 rows) | `run_all.py --only E7` | `E7__run` | `E7__goemotions` |
| E8: live end-to-end LOTUS run on Reviews | `experiments/e8_live.py` | `E8_live__2000` | `E8_live` |
| E9: unrelated control W0 | `run_all.py --only E9` | `E9__run` | `E9__<d>` |
| E10: adversarial catalog | `run_all.py --only E10` | `E10__run` | `E10__dbpedia_reviews`, `E10__goemotions_pubmed` |
| E11: appended rows | `run_all.py --only E11` | `E11__run` | `E11` |
| E12: LOTUS cascade with stricter internal targets | `run_all.py --only E12` | none | `E12` |
| ER: reworded questions | `experiments/rephrased.py cost` / `valid` | `ER`, `ER_valid` | none |
| ES: exploration sessions | `experiments/sessions.py` | `ES` | none |
| E13: 100,000-row relations (appendix) | `experiments/scale_real.py` (human-label oracle) / `scale_real.py llm` | `E13_scale`, `E13_llm__<d>_100k` | none |
| Headroom and scale studies | `experiments/headroom.py` | `HEADROOM` | `HEADROOM`, `HEADROOM_fix` |
| Tuning of opt-in operator settings (pilot data) | `experiments/improvements.py` | none | `IMP_tune*`, `IMP_E2`, `IMP_E3__<d>` |
| Oracle self-agreement (2,000 GoEmotions pairs relabeled live twice) | `python -m semviews.adapters.lotus.noise_check` | none | `noise_check_goemotions` |

All files are `.parquet`, and `<d>` stands for a dataset name. Files that do not follow the
common schema:

- `E1_pairs`: `dataset, p, q, sel_p, sel_q, eps, excl, judge, human_eps`, where
  `eps = P[q=0 | p=1]` and `excl = P[q=1 | p=1]` on the oracle tape.
- `E1_summary`: one row per dataset, with counts at eps thresholds 0.01, 0.05, 0.1.
- `E1_predictability`: `dataset, predicate_id, selectivity, auroc`.
- `E8_live*`: `mode, query, oracle_calls, wall_s, precision, recall, n_rows, positives`.
- `HEADROOM*`: `dataset, variant, mode, k, seed, query_idx, predicate_id, oracle_calls, n_rows,
  selectivity, met`.
- `E13_scale` (human-label oracle, no tape): `dataset, method, n_rows, seed, query_idx, predicate_id,
  oracle_calls, selectivity, positives, precision, recall, met, wall_s`, plus the provenance columns.
  `E13_llm__<d>_100k` follow the common schema.
- `noise_check_goemotions`: `dataset, predicate_id, row_id, tape, rep0, rep1`, i.e. the taped
  label and two fresh labels.

---

## 6. Models, sampling, and prices

### Model aliases (`configs/models.yaml`)

All calls go through a local LiteLLM proxy (`serving/litellm.yaml`) by alias, with caching and
fallbacks off. Each alias maps to exactly one model.

| Alias | Role | Model | Sampling |
|---|---|---|---|
| `oracle-llama` | Oracle; labels every tape | Llama 3.3 70B Instruct | temperature 0, `max_tokens` 8 |
| `oracle-gptoss` | Second oracle for E7 (GoEmotions, first 1,000 rows) | gpt-oss-120b | temperature 0, `max_tokens` 1024, `reasoning_effort` low |
| `judge` | Pair relations and rewordings | gpt-oss-120b | temperature 0, `max_tokens` 2048, `reasoning_effort` medium |
| `helper` | Cheap proxy scores | Llama 3.2 3B Instruct, GGUF Q4_K_M, served locally by llama.cpp (`llama-server`) | temperature 0, `max_tokens` 4, log-probabilities with top 10 |
| `embed` | Predicate and row embeddings (local) | `sentence-transformers/all-MiniLM-L6-v2` | none |

The helper was chosen among three local GGUF candidates by `serving/helper_bench.py`, which ranks
them by AUROC against oracle labels and by throughput. Helper requests go straight to the local
llama.cpp server (`direct_routes` in `configs/models.yaml`) instead of through the proxy.

### Prices and caps (`configs/budgets.yaml`)

Prices are in USD per million tokens. The `usd` column of every result file uses them.

| Alias | Input | Output |
|---|---:|---:|
| `oracle-llama` | 0.7526 | 1.484 |
| `oracle-gptoss` | 0.159 | 0.636 |
| `judge` | 0.159 | 0.636 |
| `helper` | 0 | 0 |

- The gpt-oss-120b prices come from a hosting service's public price list. The Llama 3.3 70B
  prices are the higher of the figures found, taken from a third-party calculator. Both were
  recorded on 2026-10-01.
- The file sets two caps on live spend: `usd_cap_total: 150` and
  `max_live_calls_total: 700000`. The tape recorder (`build_tape`) checks the spend priced from
  the recorded oracle tokens before every chunk, and stops once the USD cap is reached.
- Replaying the shipped tapes makes no live calls and costs nothing.
