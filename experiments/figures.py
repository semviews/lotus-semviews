# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""make figures: regenerate every figure and table from experiments/results/*.parquet.

Writes <out>/figures/*.pdf, <out>/tables/*.tex, and <out>/claims.yaml (every number the
paper quotes, with its source file and this script). <out> defaults to paper/; pass --out DIR
to write elsewhere.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(ROOT, "paper")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
SCRIPT = "experiments/figures.py"

# fixed categorical order (validated palette); color follows the method, never its rank
COLORS = {"semviews": "#2a78d6", "lotus_cascade": "#eb6834", "lotus_cascade_full": "#1baf7a", "bargain_pr": "#1baf7a",
          "proxy_lr": "#eda100", "semviews_noviews": "#e87ba4", "oracle_all": "#52514e",
          "inferred_reuse": "#4a3aa7", "sim_cache@0.8": "#e34948"}
MARKERS = {"semviews": "o", "lotus_cascade": "s", "lotus_cascade_full": "^", "bargain_pr": "^", "proxy_lr": "D",
           "semviews_noviews": "v", "oracle_all": "", "inferred_reuse": "P", "sim_cache@0.8": "X"}
LABEL = {"semviews": "SemViews", "lotus_cascade": "LOTUS cascade", "lotus_cascade_full": "LOTUS cascade (full-range)",
         "proxy_lr": "Embedding proxy", "semviews_noviews": "SemViews, no views", "oracle_all": "Oracle on every row",
         "inferred_reuse": "Inferred-containment reuse", "bargain_pr": "BARGAIN", "bargain_views": "BARGAIN + our proxy", "semweave": "SemWeave (reimpl.)", "sim_cache@0.8": "Similarity cache (0.8)",
         "sim_cache@0.7": "Similarity cache (0.7)", "sim_cache@0.9": "Similarity cache (0.9)", "sim_proxy": "Similarity proxy",
         "exact_cache": "Exact cache", "semviews_greedy": "greedy decision", "semviews_nonadaptive": "no adaptive sampling",
         "semviews_k1": "k = 1", "semviews_k2": "k = 2", "semviews_k4": "k = 4", "semviews_topembed": "top-k by embedding only",
         "semviews_nohelper": "no helper scores", "semviews_ideal": "ideal view choice (G1)",
         "semviews_noviews_noemb": "no views, no embeddings", "semviews_strata": "per-signature strata",
         "semviews_noemb": "no row embeddings", "semviews_allviews": "all views as candidates",
         "semviews_shape100": "shaping sample 100", "semviews_shape400": "shaping sample 400",
         "semviews_v1": "no refit rounds, CP bounds (first draft)"}
DSNAME = {"pubmed": "PubMed", "goemotions": "GoEmotions", "dbpedia": "DBpedia", "reviews": "Reviews"}
DSORDER = ["pubmed", "goemotions", "dbpedia", "reviews"]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})

CLAIMS: dict[str, dict] = {}


def claim(key: str, value, fmt: str, source: str) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    CLAIMS[key] = {"value": value, "fmt": fmt, "source": source, "script": SCRIPT}


def _v1_label(m: str) -> str:
    """Rows of the first-draft operator keep their files but are renamed in memory: the paper's
    operator is now the one in configs/experiments/sprint.yaml."""
    if m == "semviews":
        return "semviews_v1"
    if m.startswith("semviews") and m != "semviews_strata":
        return "v1/" + m
    return m


def load(name: str) -> pd.DataFrame | None:
    """Concatenate <name>.parquet and per-dataset parts <name>__<datasets>.parquet. Operator rows in
    results/ are from the first-draft operator and are relabeled; results/op2/ holds the current
    operator's runs."""
    import glob

    files = sorted(glob.glob(os.path.join(RES, f"{name}.parquet")) + glob.glob(os.path.join(RES, f"{name}__*.parquet")))
    new = sorted(glob.glob(os.path.join(RES, "op2", f"{name}.parquet")) + glob.glob(os.path.join(RES, "op2", f"{name}__*.parquet")))
    if not files and not new:
        return None
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        if "method" in d and name != "HEADROOM":
            d["method"] = d.method.map(_v1_label)
        parts.append(d)
    parts += [pd.read_parquet(f) for f in new]
    return pd.concat(parts, ignore_index=True)


def boot_ci(x: np.ndarray, stat=np.median, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if len(x) < 2:
        return float(stat(x)), float(stat(x))
    bs = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "method", "seed"]).agg(calls=("oracle_calls", "sum"), met=("met", "mean"),
                                                         n_q=("query_idx", "size"), rows=("n_rows", "first")).reset_index()


def fig_e2(e2: pd.DataFrame, e3: pd.DataFrame | None = None) -> None:
    """Cumulative oracle calls as a share of labeling every row, as each workload unfolds."""
    methods = ["oracle_all", "bargain_pr", "lotus_cascade", "semweave", "semviews_noviews", "semviews"]
    ds = [d for d in DSORDER if d in e2.dataset.unique()]
    fig, axes = plt.subplots(1, len(ds), figsize=(7.0, 1.9), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], ds):
        sub = e2[e2.dataset == d]
        for m in methods:
            s = sub[sub.method == m]
            if s.empty:
                continue
            s = s.assign(rel=s.cum_oracle_calls / (s.n_rows * (s.query_idx + 1)))
            piv = s.pivot_table(index="query_idx", columns="seed", values="rel")
            med = piv.median(axis=1)
            lo, hi = piv.quantile(0.1, axis=1), piv.quantile(0.9, axis=1)
            invalid = False
            if e3 is not None and m in set(e3.method):
                miss = 1 - e3[e3.method == m].groupby("dataset").met.mean()
                invalid = bool(miss.max() > e3.delta.iloc[0])
            label = LABEL[m] + (" (misses targets)" if invalid else "")
            ax.plot(med.index + 1, med.values, color=COLORS.get(m, "#4a3aa7"), lw=1.6 if m == "semviews" else 1.2,
                    ls="--" if m == "oracle_all" else (":" if invalid else "-"), marker=MARKERS.get(m, "P") or None,
                    markevery=8, ms=3.5, label=label, alpha=0.8 if invalid else 1.0)
            if m in ("semviews", "semviews_noviews"):
                ax.fill_between(med.index + 1, lo.values, hi.values, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_title(DSNAME[d], fontsize=8)
        ax.set_xlabel("queries issued")
        ax.set_ylim(0.4, 1.05)
    axes[0][0].set_ylabel("cumulative calls /\nlabeling every row")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_cumulative.pdf"), bbox_inches="tight")
    plt.close(fig)


def breakdown(e2: pd.DataFrame, method: str = "semviews") -> pd.DataFrame:
    """Per-query oracle calls of CertifiedFilter split by purpose (exact: the parts sum to the total)."""
    sv = e2[e2.method == method].copy()
    st = sv.stats.map(json.loads)
    sv["shaping"] = st.map(lambda x: x.get("n_shape", 0))
    sv["oracle region"] = st.map(lambda x: x.get("plan_o", 0))
    sv["certification"] = sv.oracle_calls_cert
    sv["refit rounds"] = st.map(lambda x: x.get("n_refit", 0))
    sv["gap closing"] = sv.oracle_calls - sv.shaping - sv["refit rounds"] - sv["oracle region"] - sv.certification
    sv["steps"] = st.map(lambda x: x.get("close_steps", 0))
    assert (sv["gap closing"] >= 0).all()
    return sv


def fig_breakdown(e2: pd.DataFrame) -> None:
    parts = ["shaping", "refit rounds", "oracle region", "certification", "gap closing"]
    cols = ["#52514e", "#1baf7a", "#2a78d6", "#eda100", "#e34948"]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ds = [d for d in DSORDER if d in e2.dataset.unique()]
    x = np.arange(len(ds))
    w = 0.38
    for k, (m, off, hatch) in enumerate([("semviews_noviews", -w / 2 - 0.01, "//"), ("semviews", w / 2 + 0.01, None)]):
        b = breakdown(e2, m)
        mean = b.groupby("dataset")[parts].mean().reindex(ds).div(b.groupby("dataset").n_rows.first().reindex(ds), axis=0)
        bottom = np.zeros(len(ds))
        for p, c in zip(parts, cols):
            ax.bar(x + off, mean[p].values, w, bottom=bottom, color=c, hatch=hatch, edgecolor="white", lw=0.3,
                   label=p if k == 1 else None)
            bottom += mean[p].values
    ax.set_xticks(x)
    ax.set_xticklabels([DSNAME[d].split(" (")[0] for d in ds], fontsize=6.5)
    ax.set_ylabel("oracle calls per row\nper query (mean)")
    ax.set_ylim(0, 1.0)
    from matplotlib.patches import Patch

    h, lab = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="white", edgecolor="#555555", hatch="//"))
    lab.append("hatched: no views")
    ax.legend(h, lab, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.4))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_breakdown.pdf"), bbox_inches="tight")
    plt.close(fig)
    b = breakdown(e2, "semviews")
    share = b.groupby("dataset")[parts].sum().div(b.groupby("dataset").oracle_calls.sum(), axis=0)
    for p in parts:
        key = p.replace(" ", "_")
        claim(f"e2 breakdown {key} min", round(100 * share[p].min()), "{:.0f}\\%", "experiments/results/E2*.parquet")
        claim(f"e2 breakdown {key} max", round(100 * share[p].max()), "{:.0f}\\%", "experiments/results/E2*.parquet")
    rounds = np.ceil(b["refit rounds"] / np.maximum(1, (0.05 * b.n_rows).astype(int)))
    claim("e2 refit rounds median", int(rounds.median()), "{:,}", "experiments/results/E2*.parquet")
    claim("e2 refit rounds p90", int(rounds.quantile(0.9)), "{:,}", "experiments/results/E2*.parquet")
    st = b.groupby("dataset").steps
    claim("e2 close steps median max", int(st.median().max()), "{:,}", "experiments/results/E2*.parquet")
    claim("e2 close steps p90 max", int(round(st.quantile(0.9).max())), "{:,}", "experiments/results/E2*.parquet")
    claim("e2 close steps zero share", round(100 * float((b.steps == 0).mean())), "{:.0f}\\%", "experiments/results/E2*.parquet")


def usd_claims(e2: pd.DataFrame) -> None:
    """Dollar cost tracks oracle calls: compare each method's share of oracle-all calls and of oracle-all USD."""
    r = e2.groupby(["dataset", "method"]).agg(c=("oracle_calls", "sum"), u=("usd", "sum")).reset_index()
    oa = r[r.method == "oracle_all"].set_index("dataset")
    gap = 0.0
    for m in ["semviews", "semviews_noviews", "bargain_pr", "lotus_cascade"]:
        x = r[r.method == m].set_index("dataset")
        gap = max(gap, float((x.c / oa.c - x.u / oa.u).abs().max()))
    claim("e2 usd call gap", round(100 * gap, 1), "{:.1f}", "experiments/results/E2*.parquet")


def fig_tradeoff(e3: pd.DataFrame) -> None:
    """Cost against validity: each point is one (method, dataset)."""
    g = e3.groupby(["method", "dataset"]).agg(miss=("met", lambda x: 1 - x.mean()), calls=("oracle_calls", "sum"),
                                                rows=("n_rows", "sum")).reset_index()
    g["share"] = g.calls / g.rows
    order = ["semviews", "semviews_noviews", "bargain_pr", "lotus_cascade", "lotus_cascade_full", "proxy_lr", "sim_proxy",
             "semweave", "sim_cache@0.8"]
    extra = {"lotus_cascade_full": ("#c4501f", "<"), "sim_proxy": ("#8c6d31", ">"), "semweave": ("#4a3aa7", "P"),
             "bargain_pr": ("#1baf7a", "^")}
    worst = 1.0  # lowest miss rate among methods cheaper than SemViews on the same dataset
    for _d, sd in g.groupby("dataset"):
        sv = sd[sd.method == "semviews"].share.iloc[0]
        cheaper = sd[(sd.share < sv) & (sd.method != "semviews")]
        if len(cheaper):
            worst = min(worst, float(cheaper.miss.min()))
    claim("e3 cheaper than semviews min miss", round(100 * worst, 1), "{:.1f}\\%", "experiments/results/E3*.parquet")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m in order:
        s = g[g.method == m]
        if s.empty:
            continue
        c, mk = extra.get(m, (COLORS.get(m, "#888888"), MARKERS.get(m, "o")))
        ax.scatter(s.share, np.maximum(s.miss, 1e-3), color=c, marker=mk, s=18 if m == "semviews" else 12,
                   label=LABEL[m], zorder=3 if m == "semviews" else 2, edgecolor="none")
    ax.axhline(e3.delta.iloc[0], color="#888888", lw=0.8, ls="--")
    ax.text(0.97, e3.delta.iloc[0] * 1.15, "$\\delta$", fontsize=7, color="#555555", ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("oracle calls / labeling every row")
    ax.set_ylabel("share of queries\nmissing a target")
    ax.legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.32), handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e3_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def tab_datasets() -> None:
    """Dataset and workload summary; selectivities are read from the oracle tape."""
    import glob

    meta = {"pubmed": ("Biomedical abstracts", "Human-written, external"),
            "goemotions": ("Social-media comments", "From emotion taxonomy"),
            "dbpedia": ("Encyclopedia entries", "From class taxonomy"),
            "reviews": ("Product reviews", "Analyst-style, ours")}
    lines = ["\\begin{tabular}{@{}lllrr@{}}", "\\toprule",
             "Dataset & Text & Predicate pool & Preds & Selectivity \\\\", "\\midrule"]
    for d in DSORDER:
        f = glob.glob(os.path.join(ROOT, "data", "tapes", f"{d}__oracle-llama.parquet"))
        if not f:
            continue
        t = pd.read_parquet(f[0], columns=["predicate_id", "row_id", "label"])
        s = t.groupby("predicate_id").label.mean()
        lines.append(f"{DSNAME[d]} & {meta[d][0]} & {meta[d][1]} & {len(s)} & "
                     f"{100 * s.median():.0f}\\% ({100 * s.min():.1f}--{100 * s.max():.0f}\\%) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "datasets.tex"), "w").write("\n".join(lines) + "\n")


def tab_e2(e2: pd.DataFrame) -> pd.DataFrame:
    r = per_run(e2)
    best_pq = ["bargain_pr", "lotus_cascade", "lotus_cascade_full", "proxy_lr", "semviews_noviews"]
    rows = []
    for d in [x for x in DSORDER if x in r.dataset.unique()]:
        sd = r[r.dataset == d]
        oa = sd[sd.method == "oracle_all"].calls.median()
        sv = sd[sd.method == "semviews"].calls.median()
        valid_pq = [m for m in best_pq if (sd[sd.method == m].met.mean() >= 0.95)]
        pq_all = {m: sd[sd.method == m].calls.median() for m in best_pq if m in sd.method.unique()}
        best_valid = min((pq_all[m], m) for m in valid_pq) if valid_pq else (np.nan, "none")
        best_any = min((v, m) for m, v in pq_all.items())
        rows.append(dict(dataset=d, oracle_all=oa, semviews=sv, saving_vs_all=oa / sv,
                         best_valid_pq=best_valid[1], best_valid_pq_calls=best_valid[0], ratio_valid=best_valid[0] / sv,
                         best_any_pq=best_any[1], best_any_pq_calls=best_any[0], ratio_any=best_any[0] / sv))
    t = pd.DataFrame(rows)
    claim("e2 ratio valid min", round(t.ratio_valid.min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    claim("e2 ratio valid max", round(t.ratio_valid.max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    claim("e2 saving all min", round(t.saving_vs_all.min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    claim("e2 saving all max", round(t.saving_vs_all.max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    claim("e2 share all min", round(100 / t.saving_vs_all.max()), "{:.0f}\\%", "experiments/results/E2*.parquet")
    claim("e2 share all max", round(100 / t.saving_vs_all.min()), "{:.0f}\\%", "experiments/results/E2*.parquet")
    # views versus the same operator without views
    nv = r[r.method == "semviews_noviews"].groupby("dataset").calls.median()
    sv = r[r.method == "semviews"].groupby("dataset").calls.median()
    vr = (nv / sv).dropna()
    claim("e2 view gain min", round(100 * (1 - 1 / vr.min()) if len(vr) else 0), "{:.0f}\\%", "experiments/results/E2*.parquet")
    claim("e2 view gain max", round(100 * (1 - 1 / vr.max()) if len(vr) else 0), "{:.0f}\\%", "experiments/results/E2*.parquet")
    # short sessions: the first 10 queries of each order
    short = e2[e2.query_idx < 10].groupby(["dataset", "method", "seed"]).oracle_calls.sum().reset_index()
    ss = short.pivot_table(index=["dataset", "seed"], columns="method", values="oracle_calls")
    if {"semviews", "semviews_noviews", "bargain_pr"} <= set(ss.columns):
        g = ss.groupby("dataset").median()
        claim("e2 short vs bargain min", round((g.bargain_pr / g.semviews).min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 short vs bargain max", round((g.bargain_pr / g.semviews).max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 short view gain max", round(100 * (1 - (g.semviews / g.semviews_noviews).min())), "{:.0f}\\%", "experiments/results/E2*.parquet")
    worst_q = (e2[e2.method == "semviews"].oracle_calls / e2[e2.method == "semviews"].n_rows).max()
    claim("e2 semviews worst query share", round(100 * worst_q, 1), "{:.1f}\\%", "experiments/results/E2*.parquet")
    swr = r[r.method == "semweave"]
    if len(swr):
        sh = (swr.calls / (swr.rows * swr.n_q)).groupby(swr.dataset).median()
        claim("e2 semweave share min", round(100 * sh.min()), "{:.0f}\\%", "experiments/results/E2*.parquet")
        claim("e2 semweave share max", round(100 * sh.max()), "{:.0f}\\%", "experiments/results/E2*.parquet")
    bv = r[r.method == "bargain_views"].groupby("dataset").calls.median()
    rb = (bv / sv).dropna()
    if len(rb):
        claim("e2 vs bargain views min", round(rb.min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 vs bargain views max", round(rb.max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    v1 = r[r.method == "semviews_v1"].groupby("dataset").calls.median()
    rv = (bv / v1).dropna()
    if len(rv):
        claim("e2 vs bargain views v1 min", round(rv.min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 vs bargain views v1 max", round(rv.max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        cut = 1 - sv / v1
        claim("e2 refine cut inv min", round((v1 / sv).min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 refine cut inv max", round((v1 / sv).max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 refine cut min", round(100 * cut.min()), "{:.0f}\\%", "experiments/results/E2*.parquet")
        claim("e2 refine cut max", round(100 * cut.max()), "{:.0f}\\%", "experiments/results/E2*.parquet")
    bp = r[r.method == "bargain_pr"].groupby("dataset").calls.median()
    rp = (bp / sv).dropna()
    if len(rp):
        claim("e2 vs bargain min", round(rp.min(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
        claim("e2 vs bargain max", round(rp.max(), 2), "{:.2f}$\\times$", "experiments/results/E2*.parquet")
    for _, x in t.iterrows():
        claim(f"e2 {x.dataset} saving vs all", round(x.saving_vs_all, 1), "{:.1f}$\\times$", "experiments/results/E2.parquet")
        claim(f"e2 {x.dataset} ratio valid pq", round(x.ratio_valid, 2), "{:.2f}$\\times$", "experiments/results/E2.parquet")
        claim(f"e2 {x.dataset} ratio any pq", round(x.ratio_any, 2), "{:.2f}$\\times$", "experiments/results/E2.parquet")
    return t


def tab_e3(e3: pd.DataFrame) -> pd.DataFrame:
    """Missed-target rate per (dataset, method): share of queries that miss either target."""
    g = e3.groupby(["method", "dataset"]).agg(miss=("met", lambda x: 1 - x.mean()), seeds=("seed", "nunique")).reset_index()
    piv = g.pivot(index="method", columns="dataset", values="miss")
    piv = piv[[d for d in DSORDER if d in piv.columns]]
    order = ["semviews", "semviews_noviews", "bargain_pr", "lotus_cascade", "lotus_cascade_full", "proxy_lr", "sim_proxy",
             "semweave", "sim_cache@0.7", "sim_cache@0.8", "sim_cache@0.9"]
    piv = piv.loc[[m for m in order if m in piv.index]]
    lines = ["\\begin{tabular}{l" + "r" * len(piv.columns) + "}", "\\toprule",
             "Method & " + " & ".join(DSNAME[d].split(" (")[0] for d in piv.columns) + " \\\\", "\\midrule"]
    for m, row in piv.iterrows():
        cells = []
        for _d, v in row.items():
            bad = v > e3.delta.iloc[0]
            s = f"{100 * v:.1f}\\%"
            cells.append(f"\\textbf{{{s}}}" if bad else s)
        lines.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e3_validity.tex"), "w").write("\n".join(lines) + "\n")
    sv = g[g.method == "semviews"]
    claim("e3 semviews max miss", round(100 * sv.miss.max(), 1), "{:.1f}\\%", "experiments/results/E3.parquet")
    claim("e3 semviews seeds", int(sv.seeds.min()), "{:,}", "experiments/results/E3.parquet")
    lc = g[g.method == "lotus_cascade"]
    if len(lc):
        claim("e3 lotus max miss", round(100 * lc.miss.max(), 1), "{:.1f}\\%", "experiments/results/E3.parquet")
        claim("e3 lotus min miss", round(100 * lc.miss.min(), 1), "{:.1f}\\%", "experiments/results/E3.parquet")
    for m in ["sim_cache@0.8", "inferred_reuse", "semweave", "lotus_cascade_full", "proxy_lr"]:
        s = g[g.method == m]
        if len(s):
            claim(f"e3 {m} max miss", round(100 * s.miss.max(), 1), "{:.1f}\\%", "experiments/results/E3.parquet")
    claim("e3 delta", float(e3.delta.iloc[0]), "{}", "experiments/results/E3.parquet")
    claim("e3 semviews queries", int((e3.method == "semviews").sum()), "{:,}", "experiments/results/E3*.parquet")
    claim("e3 semviews ms per query", round(float(e3[e3.method == "semviews"].wall_ms.median())), "{:,}",
          "experiments/results/E3*.parquet")
    # per-predicate validity: each predicate is answered once per seed
    pp = e3[e3.method == "semviews"].groupby(["dataset", "predicate_id"]).met.agg(lambda x: 1 - x.mean())
    if len(pp):
        claim("e3 semviews worst predicate miss", round(100 * pp.max(), 1), "{:.1f}\\%", "experiments/results/E3*.parquet")
        claim("e3 semviews predicates", int(len(pp)), "{:,}", "experiments/results/E3*.parquet")
        claim("e3 semviews predicates above delta", int((pp > e3.delta.iloc[0]).sum()), "{:,}", "experiments/results/E3*.parquet")
        from semviews.certify import cp_upper

        worst = e3[e3.method == "semviews"].groupby(["dataset", "predicate_id"]).met.agg(["size", lambda x: int((~x).sum())])
        worst.columns = ["n", "miss"]
        w = worst.loc[(worst.miss / worst.n).idxmax()]
        claim("e3 semviews worst predicate upper", round(100 * float(cp_upper(w.miss, w.n, 0.025)), 1), "{:.1f}\\%",
              "experiments/results/E3*.parquet")
    lp = e3[e3.method == "lotus_cascade"].groupby(["dataset", "predicate_id"]).met.agg(lambda x: 1 - x.mean())
    if len(lp):
        claim("e3 lotus predicates above delta", int((lp > e3.delta.iloc[0]).sum()), "{:,}", "experiments/results/E3*.parquet")
        claim("e3 lotus predicates", int(len(lp)), "{:,}", "experiments/results/E3*.parquet")
    sc = g[g.method.str.startswith("sim_cache")]
    if len(sc):
        claim("e3 simcache min miss", round(100 * sc.miss.min(), 0), "{:.0f}\\%", "experiments/results/E3*.parquet")
        claim("e3 simcache max miss", round(100 * sc.miss.max(), 0), "{:.0f}\\%", "experiments/results/E3*.parquet")
    bg = g[g.method == "bargain_pr"]
    if len(bg):
        claim("e3 bargain max miss", round(100 * bg.miss.max(), 1), "{:.1f}\\%", "experiments/results/E3*.parquet")
    return piv


def tab_e1(e1: pd.DataFrame) -> None:
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Dataset & Preds & $\\sqsubseteq_{0.05}$ & $\\sqsubseteq_{0.1}$ & Excl.$_{0.05}$ & Judged $\\Rightarrow$ (fail) \\\\",
             "\\midrule"]
    for _, r in e1.set_index("dataset").loc[[d for d in DSORDER if d in set(e1.dataset)]].reset_index().iterrows():
        share = r["judge_implies_share_eps_gt_0.1"]
        lines.append(f"{DSNAME[r.dataset].split(' (')[0]} & {r.n_pred} & {r['contain@0.05']} & {r['contain@0.1']} & "
                     f"{r['excl@0.05']} & {r['judge_implies']} ({100 * share:.0f}\\%) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e1_relations.tex"), "w").write("\n".join(lines) + "\n")
    tot_imp = e1.judge_implies.sum()
    bad = (e1.judge_implies * e1["judge_implies_share_eps_gt_0.1"].fillna(0)).sum()
    claim("e1 judged implies", int(tot_imp), "{:,}", "experiments/results/E1_summary.parquet")
    claim("e1 judged implies bad share", round(100 * bad / max(tot_imp, 1)), "{:.0f}\\%", "experiments/results/E1_summary.parquet")
    claim("e1 contain 0.1 pairs", int(e1["contain@0.1"].sum()), "{:,}", "experiments/results/E1_summary.parquet")
    claim("e1 pairs", int(e1.n_ordered_pairs.sum()), "{:,}", "experiments/results/E1_summary.parquet")


def fig_e4(e4: pd.DataFrame, e2: pd.DataFrame | None) -> None:
    r = e4.groupby(["dataset", "method", "variant", "seed"]).agg(calls=("oracle_calls", "sum"), rows=("n_rows", "first"),
                                                                  nq=("query_idx", "size")).reset_index()
    r["calls_per_row_query"] = r.calls / (r.rows * r.nq)
    panels = [("targets=", "targets $\\gamma_P=\\gamma_R$"), ("delta=", "$\\delta$"), ("rows=", "table rows")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.7))
    for ax, (pref, xl) in zip(axes[:3], panels):
        sub = r[r.variant.str.startswith(pref)]
        if pref == "targets=":  # symmetric targets only on one axis
            sub = sub[sub.variant.map(lambda v: len(set(v.split("=")[1].split(","))) == 1)]
        vals = sorted(sub.variant.unique(), key=lambda v: [float(x) for x in v.split("=")[1].split(",")])
        for m in ["lotus_cascade", "semviews_noviews", "semviews"]:
            s = sub[sub.method == m].groupby("variant").calls_per_row_query.median().reindex(vals)
            lab = LABEL[m] + (" (misses targets)" if m == "lotus_cascade" else "")
            ax.plot(range(len(vals)), s.values, color=COLORS[m], marker=MARKERS[m], ms=3.5, lw=1.2, label=lab,
                    ls=":" if m == "lotus_cascade" else "-")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([v.split("=")[1].split(",")[0] for v in vals], fontsize=6)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("oracle calls per row\nper query")
    if e2 is not None:
        ks = ["semviews_shape100", "semviews", "semviews_shape400"]
        rr = per_run(e2[e2.method.isin(ks)])
        rr["cpr"] = rr.calls / (rr.rows * rr.n_q)
        s = rr.groupby("method").cpr.median().reindex(ks)
        axes[3].plot([100, 200, 400], s.values, color=COLORS["semviews"], marker="o", ms=3.5, lw=1.2)
        axes[3].set_xticks([100, 200, 400])
        axes[3].set_xlabel("shaping sample size")
        mid = float(np.nanmean(s.values))  # same y-span as a typical panel, so tiny differences look tiny
        axes[3].set_ylim(mid - 0.1, mid + 0.1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e4_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_e5(e5: pd.DataFrame, e2: pd.DataFrame | None) -> None:
    e5 = e5[e5.method == "semviews"]
    r = e5.groupby(["dataset", "variant", "seed"]).agg(calls=("oracle_calls", "sum"), rows=("n_rows", "first"),
                                                       nq=("query_idx", "size")).reset_index()
    r["cov"] = r.variant.str.split("=").str[1].astype(float)
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    for i, d in enumerate([x for x in DSORDER if x in r.dataset.unique()]):
        s = r[r.dataset == d].groupby("cov").calls.median()
        base = None
        if e2 is not None:
            b = per_run(e2[(e2.dataset == d) & (e2.method == "semviews_noviews")])
            base = b.calls.median()
        y = s / base if base else s
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        mk = ["o", "s", "^", "D"]
        ax.plot(s.index, y.values, color=cols[i], marker=mk[i], ms=3.5, lw=1.2, label=DSNAME[d])
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
    if e2 is not None:
        nv = per_run(e2[e2.method == "semviews_noviews"]).groupby("dataset").calls.median()
        rel = r.groupby(["dataset", "cov"]).calls.median() / nv.reindex(r.groupby(["dataset", "cov"]).calls.median().index.get_level_values(0)).values
        part = rel[rel.index.get_level_values("cov") < 1.0]
        other = part[part.index.get_level_values("dataset") != "goemotions"]
        claim("e5 partial other max dev", round(100 * float((other - 1).abs().max()), 1), "{:.1f}\\%", "experiments/results/E5*.parquet")
        if "goemotions" in part.index.get_level_values("dataset"):
            ge = part.loc["goemotions"]
            claim("e5 goemotions half saving", round(100 * float(1 - ge.loc[0.5])), "{:.0f}\\%", "experiments/results/E5*.parquet")
    ax.set_xlabel("share of each view's labels kept")
    ax.set_ylabel("calls relative to\nno views")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e5_partial.pdf"), bbox_inches="tight")
    plt.close(fig)


def tab_e9(e9: pd.DataFrame) -> None:
    r = per_run(e9)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Dataset & Preds & No views & SemViews & Overhead \\\\", "\\midrule"]
    worst = 0.0
    for d in [x for x in DSORDER if x in r.dataset.unique()]:
        sd = r[r.dataset == d]
        nv = sd[sd.method == "semviews_noviews"].calls.median()
        sv = sd[sd.method == "semviews"].calls.median()
        ov = sv / nv - 1
        worst = max(worst, ov)
        lines.append(f"{DSNAME[d].split(' (')[0]} & {int(sd.n_q.iloc[0])} & {nv:,.0f} & {sv:,.0f} & {100 * ov:+.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e9_overhead.tex"), "w").write("\n".join(lines) + "\n")
    claim("e9 worst overhead", round(100 * worst, 1), "{:.1f}\\%", "experiments/results/E9.parquet")


def tab_ablation(e2: pd.DataFrame) -> None:
    """Calls of each variant relative to SemViews on the same workload order: median [10th, 90th percentile]."""
    r = per_run(e2)
    piv = r.pivot_table(index=["dataset", "seed"], columns="method", values="calls")
    ab = ["semviews_v1", "semviews_noviews", "semviews_noviews_noemb", "semviews_strata", "semviews_noemb", "semviews_nohelper",
          "semviews_allviews", "semviews_shape100", "semviews_shape400"]
    ds = [d for d in DSORDER if d in r.dataset.unique()]
    short = {"semviews_v1": "first draft (no refit, CP)", "semviews_noviews": "no views",
             "semviews_allviews": "all views as candidates"}  # short labels keep the table legible
    hdr = {"goemotions": "GoEmo."}
    lines = ["\\begin{tabular}{@{}l" + "r" * len(ds) + "@{}}", "\\toprule",
             "Variant & " + " & ".join(hdr.get(d, DSNAME[d]) for d in ds) + " \\\\", "\\midrule"]
    for m in ab:
        if m not in piv.columns:
            continue
        cells = []
        for d in ds:
            x = (piv.loc[d, m] / piv.loc[d, "semviews"]).dropna()
            cells.append(f"{x.median():.2f}\\,{{\\scriptsize[{x.quantile(0.1):.2f},{x.quantile(0.9):.2f}]}}")
        lines.append(f"{short.get(m, LABEL[m])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def tab_e7(e7: pd.DataFrame) -> None:
    r = e7.groupby(["method", "variant", "seed"]).agg(calls=("oracle_calls", "sum"), met=("met", "mean")).reset_index()
    g = r.groupby(["method", "variant"]).agg(calls=("calls", "median"), met=("met", "mean")).reset_index()
    gi = g.set_index(["method", "variant"])
    try:
        nv = gi.loc[("semviews_noviews", "same model"), "calls"]
        cm = gi.loc[("semviews", "cross-model views"), "calls"]
        claim("e7 cross model saving", round(100 * (1 - cm / nv)), "{:.0f}\\%", "experiments/results/E7*.parquet")
        claim("e7 cross model met", round(100 * gi.loc[("semviews", "cross-model views"), "met"]), "{:.0f}\\%", "experiments/results/E7*.parquet")
        claim("e7 simcache cross met", round(100 * gi.loc[("sim_cache@0.9", "cross-model views"), "met"]), "{:.0f}\\%", "experiments/results/E7*.parquet")
    except KeyError:
        pass
    g = g[g.method != "inferred_reuse"]  # relation judgments exist only for same-model predicate ids
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{Same-oracle views} & \\multicolumn{2}{c}{Views from another oracle} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Method & Calls & Met & Calls & Met \\\\", "\\midrule"]
    for m in ["semviews", "semviews_noviews", "lotus_cascade", "sim_cache@0.9"]:
        if m not in set(g.method):
            continue
        cells = []
        for v in ["same model", "cross-model views"]:
            x = gi.loc[(m, v)]
            cells += [f"{x.calls:,.0f}", f"{100 * x.met:.0f}\\%"]
        lines.append(f"{LABEL.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e7_robustness.tex"), "w").write("\n".join(lines) + "\n")


def tape_claims() -> None:
    import glob

    n = 0
    for f in glob.glob(os.path.join(ROOT, "data", "tapes", "*__oracle-llama.parquet")):
        if "_100k__" not in os.path.basename(f):  # the appendix scale check (E13) is counted separately
            n += len(pd.read_parquet(f, columns=["label"]))
    claim("tape oracle calls", int(n), "{:,}", "data/tapes/*__oracle-llama.parquet (without *_100k__*)")
    g = glob.glob(os.path.join(ROOT, "data", "tapes", "*__oracle-gptoss.parquet"))
    if g:
        claim("tape gptoss calls", int(sum(len(pd.read_parquet(f, columns=["label"])) for f in g)), "{:,}",
              "data/tapes/*__oracle-gptoss.parquet")
    nc = glob.glob(os.path.join(RES, "noise_check_*.parquet"))
    if nc:
        d = pd.concat([pd.read_parquet(f) for f in nc])
        agree = ((d.rep0 == d.tape) & (d.rep1 == d.tape)).mean()
        claim("noise agreement", round(100 * agree, 2), "{:.2f}\\%", "experiments/results/noise_check_*.parquet")


def predictability_claims() -> None:
    d = load("E1_predictability")
    if d is None:
        return
    med = d.groupby("dataset").auroc.median()
    for ds, v in med.items():
        claim(f"e1 auroc {ds}", round(v, 2), "{:.2f}", "experiments/results/E1_predictability.parquet")


def e10_claims(e10: pd.DataFrame) -> None:
    r = e10.groupby(["dataset", "variant", "method", "seed"]).agg(c=("oracle_calls", "sum"), m=("met", "mean")).reset_index()
    g = r.groupby(["dataset", "variant", "method"]).agg(calls=("c", "median"), met=("m", "mean")).reset_index()
    adv = g[(g.variant == "adversarial") & (g.method == "semviews")]
    nv = g[(g.variant == "adversarial") & (g.method == "semviews_noviews")].set_index("dataset").calls
    claim("e10 adversarial met min", round(100 * adv.met.min(), 1), "{:.1f}\\%", "experiments/results/E10*.parquet")
    claim("e10 adversarial cost ratio max", round((adv.set_index("dataset").calls / nv).max(), 2), "{:.2f}",
          "experiments/results/E10*.parquet")
    inv = g[(g.variant == "inverted") & (g.method == "semviews")].set_index("dataset").calls
    claim("e10 inverted saving min", round(100 * (1 - (inv / nv).max())), "{:.0f}\\%", "experiments/results/E10*.parquet")
    claim("e10 inverted saving max", round(100 * (1 - (inv / nv).min())), "{:.0f}\\%", "experiments/results/E10*.parquet")
    sc = g[(g.method == "sim_cache@0.9")]
    claim("e10 simcache met", round(100 * sc.met.max()), "{:.0f}\\%", "experiments/results/E10*.parquet")


def tape_truth(d: str) -> pd.DataFrame:
    """Oracle labels as a row x predicate matrix (evaluation only: used to describe workloads)."""
    t = pd.read_parquet(os.path.join(ROOT, "data", "tapes", f"{d}__oracle-llama.parquet"),
                        columns=["predicate_id", "row_id", "label"])
    return t.pivot_table(index="row_id", columns="predicate_id", values="label", aggfunc="first").astype(float)


def framing_claims(e2: pd.DataFrame, e4: pd.DataFrame | None) -> None:
    """Savings relative to exact evaluation, warm-catalog costs, view gains conditional on a related
    earlier predicate, cost by selectivity, and the certification floor. Writes tables/views.tex."""
    src = "experiments/results/E2*.parquet"
    r = per_run(e2)
    med = r.groupby(["dataset", "method"]).calls.median().unstack(0)
    oa = med.loc["oracle_all"]
    sv_save = 1 - med.loc["semviews"] / oa
    bg_save = 1 - med.loc["bargain_pr"] / oa
    claim("e2 semviews saving min", round(100 * sv_save.min()), "{:.0f}\\%", src)
    claim("e2 semviews saving max", round(100 * sv_save.max()), "{:.0f}\\%", src)
    claim("e2 bargain saving min", round(100 * bg_save.min()), "{:.0f}\\%", src)
    claim("e2 bargain saving max", round(100 * bg_save.max()), "{:.0f}\\%", src)
    ratio = sv_save / bg_save
    claim("e2 saving ratio min", round(ratio.min()), "{:.0f}$\\times$", src)
    claim("e2 saving ratio max", round(ratio.max()), "{:.0f}$\\times$", src)

    # warm catalog: the last ten queries of every workload order
    nq = e2.groupby("dataset").query_idx.transform("max") + 1
    late = e2[e2.query_idx >= nq - 10]
    lr = late.groupby(["dataset", "method", "seed"]).oracle_calls.sum().unstack(1).groupby("dataset").median()
    late_gain = 1 - lr.semviews / lr.semviews_noviews
    late_bg = lr.bargain_pr / lr.semviews
    claim("e2 late view gain max", round(100 * late_gain.max()), "{:.0f}\\%", src)
    claim("e2 late vs bargain min", round(late_bg.min(), 2), "{:.2f}$\\times$", src)
    claim("e2 late vs bargain max", round(late_bg.max(), 2), "{:.2f}$\\times$", src)
    claim("e2 late share min", round(100 * (lr.semviews / (10 * 5000)).min()), "{:.0f}\\%", src)

    # views conditional on whether an earlier query of the order asked a strongly related predicate
    pv = e2[e2.method.isin(["semviews", "semviews_noviews"])].pivot_table(
        index=["dataset", "seed", "query_idx", "predicate_id"], columns="method", values="oracle_calls").reset_index()
    rows, sel_rows = [], []
    for d in [x for x in DSORDER if x in pv.dataset.unique()]:
        Y = tape_truth(d)
        C = Y.corr().abs().fillna(0.0)
        for p in C.columns:
            C.loc[p, p] = 0.0
        x = pv[pv.dataset == d].sort_values(["seed", "query_idx"])
        rel = []
        for _, g in x.groupby("seed", sort=False):
            seen: list[str] = []
            for p in g.predicate_id:
                rel.append(bool(seen) and float(C.loc[p, seen].max()) > 0.6)
                seen.append(p)
        x = x.assign(rel=rel)
        gain = lambda g: 1 - g.semviews.sum() / g.semviews_noviews.sum()  # noqa: E731
        rows.append(dict(dataset=d, partners=int((C.max(axis=1) > 0.6).sum()), preds=len(C), rel_share=x.rel.mean(),
                         gain_rel=gain(x[x.rel]) if x.rel.any() else np.nan, gain_unrel=gain(x[~x.rel]),
                         gain_all=1 - med.loc["semviews", d] / med.loc["semviews_noviews", d], late=late_gain.get(d, np.nan)))
        s = Y.mean()
        sel_rows.append(x.assign(sel=x.predicate_id.map(s)))
    t = pd.DataFrame(rows)
    # the first-draft operator on the same orders: does refinement change what views are worth?
    pv1 = e2[e2.method.isin(["semviews_v1", "v1/semviews_noviews"])].pivot_table(
        index=["dataset", "seed", "query_idx", "predicate_id"], columns="method", values="oracle_calls").reset_index()
    g1 = []
    for d in [x for x in DSORDER if x in pv1.dataset.unique()]:
        C = tape_truth(d).corr().abs().fillna(0.0)
        for p in C.columns:
            C.loc[p, p] = 0.0
        x = pv1[pv1.dataset == d].sort_values(["seed", "query_idx"])
        rel = []
        for _, g in x.groupby("seed", sort=False):
            seen: list[str] = []
            for p in g.predicate_id:
                rel.append(bool(seen) and float(C.loc[p, seen].max()) > 0.6)
                seen.append(p)
        x = x.assign(rel=rel)[rel]
        g1.append(1 - x.semviews_v1.sum() / x["v1/semviews_noviews"].sum())
    if g1:
        claim("views v1 related gain min", round(100 * min(g1)), "{:.0f}\\%", src)
        claim("views v1 related gain max", round(100 * max(g1)), "{:.0f}\\%", src)
    t = pd.DataFrame(rows)
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "& Related & \\multicolumn{4}{c}{Calls saved by views} \\\\", "\\cmidrule(l){3-6}",
             "Dataset & preds & related & unrelated & all & last 10 \\\\", "\\midrule"]
    for _, x in t.iterrows():
        lines.append(f"{DSNAME[x.dataset]} & {x.partners}/{x.preds} & {100 * x.gain_rel:.0f}\\% & "
                     f"{100 * x.gain_unrel:.0f}\\% & {100 * x.gain_all:.0f}\\% & {100 * x.late:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "views.tex"), "w").write("\n".join(lines) + "\n")
    strong = t[t.partners >= 0.2 * t.preds]
    claim("views related gain min", round(100 * t.gain_rel.min()), "{:.0f}\\%", src)
    claim("views related gain max", round(100 * t.gain_rel.max()), "{:.0f}\\%", src)
    claim("views related gain strong min", round(100 * strong.gain_rel.min()), "{:.0f}\\%", src)
    claim("views unrelated gain max", round(100 * t.gain_unrel.max()), "{:.0f}\\%", src)
    claim("views unrelated gain other max", round(100 * t[t.dataset != "goemotions"].gain_unrel.max()), "{:.0f}\\%", src)
    claim("views partners goemotions", int(t.set_index("dataset").partners.get("goemotions", 0)), "{}", src)
    claim("views partners other max", int(t[t.dataset != "goemotions"].partners.max()), "{}", src)

    # cost by selectivity: low-selectivity predicates sit at the certification floor
    sx = pd.concat(sel_rows)
    lo = sx[sx.sel <= 0.02]
    hi = sx[sx.sel > 0.10]
    claim("sel low share", round(100 * lo.semviews.sum() / (5000 * len(lo))), "{:.0f}\\%", src)
    claim("sel high share", round(100 * hi.semviews.sum() / (5000 * len(hi))), "{:.0f}\\%", src)
    npred = sx.drop_duplicates(["dataset", "predicate_id"])
    claim("sel low preds", int((npred.sel <= 0.05).sum()), "{}", src)
    claim("sel preds", int(len(npred)), "{}", src)
    # Certification floor: with zero positives drawn, n draws bound a region's positive rate by about
    # ln(1/alpha)/n; meeting recall gamma requires n >= gamma ln(1/alpha) / ((1 - gamma) sel).
    g, a = 0.9, 0.05 / 2
    coef = g * np.log(1 / a) / (1 - g)
    claim("floor coef", round(coef), "{:,}", "experiments/figures.py (closed form)")
    claim("floor draws at one pct", int(round(coef / 0.01, -2)), "{:,}", "experiments/figures.py (closed form)")
    if e4 is not None:
        rr = e4[(e4.method == "semviews") & e4.variant.str.startswith("rows=")]
        rr = rr.groupby(["variant", "dataset", "seed"]).agg(c=("oracle_calls", "sum"), n=("n_rows", "first"),
                                                             q=("query_idx", "size")).reset_index()
        rr["f"] = rr.c / (rr.n * rr.q)
        f = rr.groupby("variant").f.median()
        claim("e4 rows small share", round(100 * f["rows=1000"]), "{:.0f}\\%", "experiments/results/E4*.parquet")
        claim("e4 rows large share", round(100 * f["rows=5000"]), "{:.0f}\\%", "experiments/results/E4*.parquet")


def headroom_claims() -> None:
    """Perfect-score floor, fully warm catalog, and the tiled table-size simulation
    (experiments/headroom.py); the paper's operator only."""
    import glob

    f = os.path.join(RES, "op2", "HEADROOM.parquet")
    if not glob.glob(f):
        return
    src = "experiments/results/op2/HEADROOM.parquet"
    h = pd.read_parquet(f)
    h = h[h.variant == "paper"]
    share = lambda x: x.oracle_calls.sum() / x.n_rows.sum()  # noqa: E731

    def mean_share(df):  # mean over datasets of each dataset's pooled share
        return float(np.mean([share(g) for _, g in df.groupby("dataset")]))

    s1 = h[(h["mode"] == "scale") & (h.k == 1)]
    perf = h[(h["mode"] == "perfect") & (h.k == 1)]
    claim("headroom operator share", round(100 * mean_share(s1)), "{:.0f}\\%", src)
    claim("headroom perfect share", round(100 * mean_share(perf)), "{:.0f}\\%", src)
    claim("headroom perfect rare share", round(100 * share(perf[perf.selectivity <= 0.02])), "{:.0f}\\%", src)
    claim("headroom perfect common share", round(100 * share(perf[perf.selectivity > 0.10])), "{:.0f}\\%", src)
    warm = h[h["mode"] == "warm"]
    extra = [1 - share(warm[warm.dataset == d]) / share(s1[s1.dataset == d]) for d in s1.dataset.unique()]
    claim("headroom warm extra min", max(0, round(100 * min(extra))), "{:.0f}\\%", src)
    claim("headroom warm extra max", round(100 * max(extra)), "{:.0f}\\%", src)
    for k, name in ((4, "mid"), (20, "large")):
        claim(f"headroom scale {name} share", round(100 * mean_share(h[(h["mode"] == "scale") & (h.k == k)])), "{:.0f}\\%", src)
    big = h[(h["mode"] == "scale_shape") & (h.k == 20)]
    claim("headroom scale large shaped share", round(100 * mean_share(big)), "{:.0f}\\%", src)
    claim("headroom scale large perfect share",
          round(100 * mean_share(h[(h["mode"] == "perfect") & (h.k == 20)])), "{:.0f}\\%", src)
    claim("headroom scale max miss", round(100 * (1 - h[h["mode"].str.startswith("scale")].groupby(
        ["dataset", "k", "mode"]).met.mean()).max(), 1), "{:.1f}\\%", src)


def scale_real_claims() -> None:
    """E13 (experiments/scale_real.py): a real 100,000-tuple DBpedia relation, with the human
    ontology labels (all predicates) and the LLM oracle (8 predicates) as oracles; appendix table
    and claims."""
    import glob

    runs = []
    for key, name, label in (("human", "E13_scale", "DBpedia, human labels"),
                             ("llm dbpedia", "E13_llm__dbpedia_100k", "DBpedia, LLM oracle"),
                             ("llm reviews", "E13_llm__reviews_100k", "Reviews, LLM oracle")):
        f = os.path.join(RES, "op2", f"{name}.parquet")
        if glob.glob(f):
            runs.append((key, f"experiments/results/op2/{name}.parquet", label, pd.read_parquet(f)))
    if not runs:
        return
    share = lambda x: x.oracle_calls.sum() / x.n_rows.sum()  # noqa: E731
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "& & \\multicolumn{2}{c}{Share of exact} & Rare & Both \\\\", "\\cmidrule(lr){3-4}",
             "Oracle & Tuples & \\sys & no views & ($\\le$2\\%) & met \\\\"]
    for key, src, label, e in runs:
        sizes = sorted(e.n_rows.unique())
        sv, nv = e[e.method == "semviews"], e[e.method == "semviews_noviews"]
        lines.append("\\midrule")
        for i, n in enumerate(sizes):
            a, b = sv[sv.n_rows == n], nv[nv.n_rows == n]
            head = (label if i == 0 else f"({e.predicate_id.nunique()} predicates)" if i == 1 else "")
            lines.append(f"{head} & {n:,} & {100 * share(a):.0f}\\% & {100 * share(b):.0f}\\% & "
                         f"{100 * share(a[a.selectivity <= 0.02]):.0f}\\% & {100 * a.met.mean():.1f}\\% \\\\")
        big, small = max(sizes), min(sizes)
        claim(f"e13 {key} preds", int(e.predicate_id.nunique()), "{}", src)
        claim(f"e13 {key} orders", int(e.seed.nunique()), "{}", src)
        claim(f"e13 {key} share small", round(100 * share(sv[sv.n_rows == small])), "{:.0f}\\%", src)
        claim(f"e13 {key} share large", round(100 * share(sv[sv.n_rows == big])), "{:.0f}\\%", src)
        claim(f"e13 {key} noviews share large", round(100 * share(nv[nv.n_rows == big])), "{:.0f}\\%", src)
        claim(f"e13 {key} rare share small", round(100 * share(sv[(sv.n_rows == small) & (sv.selectivity <= 0.02)])), "{:.0f}\\%", src)
        claim(f"e13 {key} rare share large", round(100 * share(sv[(sv.n_rows == big) & (sv.selectivity <= 0.02)])), "{:.0f}\\%", src)
        claim(f"e13 {key} max miss", round(100 * (1 - e.groupby(["method", "n_rows"]).met.mean()).max(), 1), "{:.1f}\\%", src)
        claim(f"e13 {key} queries large", int((sv.n_rows == big).sum()), "{}", src)
        claim(f"e13 {key} sel min", round(100 * e[e.n_rows == big].selectivity.min(), 1), "{:.1f}\\%", src)
        claim(f"e13 {key} sel max", round(100 * e[e.n_rows == big].selectivity.max(), 1), "{:.1f}\\%", src)
        claim("e13 rows", int(big), "{:,}", src)
        claim("e13 small rows", int(small), "{:,}", src)
    llm = [e for k, _, _, e in runs if k.startswith("llm")]
    if llm:
        claim("e13 llm labels", int(sum(e.predicate_id.nunique() * e.n_rows.max() for e in llm)), "{:,}",
              "experiments/results/op2/E13_llm__*.parquet")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "e13_scale.tex"), "w").write("\n".join(lines) + "\n")
    # How the human-label oracle relates to the LLM oracle, on the 5,000-tuple DBpedia table.
    pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", "dbpedia.yaml")))["predicates"]
    rows = pd.read_parquet(os.path.join(ROOT, "data", "datasets", "dbpedia.parquet"))
    Y = tape_truth("dbpedia").reindex(rows.row_id)
    agree, f1 = [], []
    for p in pool:
        h = np.zeros(len(rows), dtype=bool)
        for col, vals in p["human"].items():
            h |= rows[col].isin(vals).to_numpy()
        y = Y[f"dbpedia/{p['id']}"].to_numpy().astype(bool)
        agree.append(float((h == y).mean()))
        f1.append(2 * (h & y).sum() / max(1, h.sum() + y.sum()))
    src_h = "data/tapes/dbpedia__oracle-llama.parquet, data/datasets/dbpedia.parquet"
    claim("e13 human llm agree median", round(100 * float(np.median(agree)), 1), "{:.1f}\\%", src_h)
    claim("e13 human llm f1 median", round(float(np.median(f1)), 2), "{:.2f}", src_h)


def rephrased_claims() -> None:
    """Workload W_R (experiments/rephrased.py): repeated questions in different words."""
    import glob

    f = os.path.join(RES, "op2", "ER.parquet")
    if not glob.glob(f):
        return
    src = "experiments/results/op2/ER*.parquet"
    er = pd.read_parquet(f)
    fv = os.path.join(RES, "op2", "ER_valid.parquet")
    ev = pd.read_parquet(fv) if glob.glob(fv) else er
    ds = ["reviews", "goemotions"]
    twins, agree, jac = {}, [], []
    for d in ds:
        pool = yaml.safe_load(open(os.path.join(ROOT, "configs", "predicates", f"{d}_rephrased.yaml")))["predicates"]
        Y = tape_truth(d)
        rw = pd.read_parquet(os.path.join(ROOT, "data", "tapes", f"{d}__oracle-llama__rw.parquet"),
                             columns=["predicate_id", "row_id", "label"]).pivot_table(
            index="row_id", columns="predicate_id", values="label", aggfunc="first").reindex(Y.index)
        for p in pool:
            a, b = f"{d}/{p['id']}", f"{d}/{p['rephrase_of']}"
            twins[a], twins[b] = b, a
            x, y = rw[a].astype(bool), Y[b].astype(bool)
            agree.append(float((x == y).mean()))
            jac.append(float((x & y).sum() / max(1, (x | y).sum())))
    claim("wr twins", len(agree) // len(ds), "{}", src)
    claim("wr agree min", round(100 * min(agree)), "{:.0f}\\%", src)
    claim("wr agree max", round(100 * max(agree), 1), "{:.1f}\\%", src)
    claim("wr jaccard min", round(min(jac), 2), "{:.2f}", src)
    claim("wr jaccard max", round(max(jac), 2), "{:.2f}", src)

    def flag(df):
        pos = {(r.dataset, r.seed, r.predicate_id): r.query_idx for r in df[df.method == df.method.iloc[0]].itertuples()}
        return [(p in twins) and pos.get((d, s, twins[p]), 10**9) < q
                for d, s, p, q in zip(df.dataset, df.seed, df.predicate_id, df.query_idx)]

    er = er.assign(repeat=flag(er))
    ev = ev.assign(repeat=flag(ev))
    rep, evr = er[er.repeat], ev[ev.repeat]
    share = rep.groupby(["method", "dataset"]).apply(lambda x: x.oracle_calls.sum() / x.n_rows.sum(), include_groups=False).unstack(1)
    miss = evr.groupby(["method", "dataset"]).met.agg(lambda m: 1 - m.mean()).unstack(1)
    order = ["semviews", "semviews_noviews", "bargain_pr", "lotus_cascade", "semweave", "sim_cache@0.8", "sim_cache@0.9"]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "& \\multicolumn{2}{c}{Reviews} & \\multicolumn{2}{c}{GoEmotions} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}", "Method & Calls & Miss & Calls & Miss \\\\", "\\midrule"]
    for m in [m for m in order if m in share.index and m in miss.index]:
        cells = []
        for d in ds:
            mi = miss.loc[m, d]
            ms = f"{100 * mi:.1f}\\%"
            cells += [f"{share.loc[m, d]:.2f}", f"\\textbf{{{ms}}}" if mi > 0.05 else ms]
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, "rephrased.tex"), "w").write("\n".join(lines) + "\n")
    gain = 1 - share.loc["semviews"] / share.loc["semviews_noviews"]
    claim("wr view gain min", round(100 * gain.min()), "{:.0f}\\%", src)
    claim("wr view gain max", round(100 * gain.max()), "{:.0f}\\%", src)
    claim("wr semviews share min", round(100 * share.loc["semviews"].min()), "{:.0f}\\%", src)
    claim("wr semviews share max", round(100 * share.loc["semviews"].max()), "{:.0f}\\%", src)
    claim("wr semviews miss max", round(100 * miss.loc["semviews"].max(), 1), "{:.1f}\\%", src)
    unsafe = miss.loc[[m for m in ("semweave", "sim_cache@0.8") if m in miss.index]]
    claim("wr unsafe miss min", round(100 * unsafe.values.min()), "{:.0f}\\%", src)
    claim("wr unsafe miss max", round(100 * unsafe.values.max()), "{:.0f}\\%", src)
    claim("wr unsafe met min", round(100 * (1 - unsafe.values.max())), "{:.0f}\\%", src)
    claim("wr unsafe met max", round(100 * (1 - unsafe.values.min())), "{:.0f}\\%", src)
    claim("wr semviews met min", round(100 * (1 - miss.loc["semviews"].max()), 1), "{:.1f}\\%", src)
    claim("wr semweave share max", round(100 * share.loc["semweave"].max()), "{:.0f}\\%", src)
    claim("wr orders valid", int(ev.seed.nunique()), "{:,}", src)
    claim("wr orders cost", int(er.seed.nunique()), "{:,}", src)
    claim("wr repeats", int(rep[rep.method == "semviews"].groupby("dataset").size().min() // er.seed.nunique()), "{}", src)


def extra_claims() -> None:
    import json as _json

    e11 = load("E11")
    if e11 is not None:
        r = e11.groupby(["dataset", "method", "seed"]).agg(c=("oracle_calls", "sum"), m=("met", "mean"),
                                                          n=("n_rows", "first"), q=("query_idx", "size")).reset_index()
        r["f"] = r.c / (r.n * r.q)
        g = r.groupby(["dataset", "method"]).agg(f=("f", "median"), m=("m", "mean")).reset_index()
        sv = g[g.method == "semviews"]
        claim("e11 append share min", round(100 * sv.f.min()), "{:.0f}\\%", "experiments/results/E11*.parquet")
        claim("e11 append share max", round(100 * sv.f.max()), "{:.0f}\\%", "experiments/results/E11*.parquet")
        claim("e11 append met", round(100 * sv.m.min(), 1), "{:.1f}\\%", "experiments/results/E11*.parquet")
        claim("e11 simcache met", round(100 * g[g.method == "sim_cache@0.9"].m.max()), "{:.0f}\\%", "experiments/results/E11*.parquet")
    e12 = load("E12")
    if e12 is not None:
        miss = e12[e12.method == "lotus_strict2"].groupby("dataset").met.agg(lambda x: 1 - x.mean())
        claim("e12 strict lotus miss min", round(100 * miss.min(), 1), "{:.1f}\\%", "experiments/results/E12*.parquet")
        claim("e12 strict lotus miss max", round(100 * miss.max(), 1), "{:.1f}\\%", "experiments/results/E12*.parquet")
        claim("e12 strict lotus valid datasets", int((miss <= 0.05).sum()), "{}", "experiments/results/E12*.parquet")
    e3 = load("E3")
    if e3 is not None:
        sc = e3[e3.method.str.startswith("sim_cache")]
        hit = sc.stats.map(lambda x: "hit" in _json.loads(x))
        claim("e3 simcache miss given hit", round(100 * (1 - sc[hit.values].met.mean())), "{:.0f}\\%", "experiments/results/E3*.parquet")
        ir = e3[e3.method == "inferred_reuse"]
        used = ir.stats.map(lambda x: "reused" in _json.loads(x))
        claim("e3 inferred miss given reuse", round(100 * (1 - ir[used.values].met.mean())), "{:.0f}\\%", "experiments/results/E3*.parquet")
        sw = e3[e3.method == "semweave"]
        if len(sw):
            ru = sw.stats.map(lambda x: _json.loads(x).get("reused_rows", 0) > 0)
            claim("e3 semweave miss given reuse", round(100 * (1 - sw[ru.values].met.mean())), "{:.0f}\\%", "experiments/results/E3*.parquet")
            claim("e3 semweave reuse share", round(100 * ru.mean()), "{:.0f}\\%", "experiments/results/E3*.parquet")
            claim("e3 semweave recall p5", round(sw[ru.values].recall.quantile(0.05), 2), "{:.2f}", "experiments/results/E3*.parquet")
            mm = sw.groupby("dataset").met.agg(lambda x: 1 - x.mean())
            claim("e3 semweave min miss", round(100 * mm.min(), 1), "{:.1f}\\%", "experiments/results/E3*.parquet")
        sv = e3[e3.method == "semviews"]
        claim("e3 semviews pooled miss", round(100 * (1 - sv.met.mean()), 2), "{:.2f}\\%", "experiments/results/E3*.parquet")
        claim("e3 semviews pooled met", round(100 * sv.met.mean(), 2), "{:.2f}\\%", "experiments/results/E3*.parquet")


def main() -> None:
    import argparse

    global OUT, FIG, TAB
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, help="output directory (default: paper/)")
    OUT = os.path.abspath(ap.parse_args().out)
    FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    extra_claims()
    headroom_claims()
    scale_real_claims()
    rephrased_claims()
    e10 = load("E10")
    if e10 is not None:
        e10_claims(e10)
    tape_claims()
    predictability_claims()
    e1, e2, e3, e4, e5, e7, e9 = (load(n) for n in ["E1_summary", "E2", "E3", "E4", "E5", "E7", "E9"])
    out = {}
    tab_datasets()
    if e1 is not None:
        tab_e1(e1)
    if e2 is not None:
        fig_e2(e2, e3)
        fig_breakdown(e2)
        usd_claims(e2)
        out["e2"] = tab_e2(e2)
        tab_ablation(e2)
    if e3 is not None:
        out["e3"] = tab_e3(e3)
        fig_tradeoff(e3)
    if e2 is not None:
        framing_claims(e2, e4)
    if e4 is not None:
        fig_e4(e4, e2)
    if e5 is not None:
        fig_e5(e5, e2)
    if e7 is not None:
        tab_e7(e7)
    if e9 is not None:
        tab_e9(e9)
    with open(os.path.join(OUT, "claims.yaml"), "w") as f:
        f.write("# Generated by experiments/figures.py. Every number in the paper. Do not edit by hand.\n")
        yaml.safe_dump({"claims": CLAIMS}, f, sort_keys=True, width=200)
    pd.set_option("display.width", 250)
    for k, v in out.items():
        print(k)
        print(v)
    print(json.dumps({k: v["value"] for k, v in CLAIMS.items()}, indent=0)[:3000])


if __name__ == "__main__":
    sys.exit(main())
