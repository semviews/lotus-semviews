# Use of generative AI

The ACM Policy on Authorship asks authors to disclose the use of generative AI tools. We used
Claude Code (Claude Opus 5.5) as a coding and writing assistant, under the authors' direction.

**What it was used for.**

- Writing code: the implementation, its tests, the experiment and figure scripts, and the
  tooling for recording and replaying model outputs.
- Writing documentation for this repository.
- Writing the analyst-style predicates for the Reviews workload; the GoEmotions and DBpedia
  predicates were derived from the datasets' label taxonomies.
- Drafting and revising the text of the paper, including the appendix proofs, and giving
  feedback on drafts.

**Role of the authors.** The authors set the research question, directed the work, made the
decisions about the method, the experiments, and the paper, and are responsible for all of its
content.

**Safeguards.**

- Every number in the paper is computed by `experiments/figures.py` from the result files and
  inserted through generated macros; none was typed by hand. `make reproduce-figures` checks
  each one against `reference/claims.yaml`.
- A reference entered the bibliography only after its source had been opened.
- The confidence bounds have simulation tests of their coverage (`tests/test_certify.py`), and
  `tests/test_boundaries.py` checks that only the oracle module reads the recorded labels.

The models that the method itself calls (the oracle, the helper, the judge of predicate
relations, and the model that rewords questions) are part of the experiments, not authoring
tools; they are described in `docs/DATA.md` and in the paper.
