# Copyright 2026 The semviews authors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""E1b: an upper bound on what views can offer. For each predicate, 3-fold cross-validated AUROC
of a logistic model whose features are the exact oracle labels of every *other* predicate.
Writes experiments/results/E1_predictability.parquet."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semviews.workloads.runner import ROOT, Bench  # noqa: E402


def main(datasets):
    rows = []
    for ds in datasets:
        b = Bench(ds, use_helper_tape=False)
        Y = np.stack([b.truth[p] for p in b.preds], 1).astype(float)
        for j, p in enumerate(b.preds):
            X, y = np.delete(Y, j, axis=1), Y[:, j]
            pr = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=3, method="predict_proba")[:, 1]
            rows.append(dict(dataset=ds, predicate_id=p, selectivity=y.mean(), auroc=roc_auc_score(y, pr)))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "experiments", "results", "E1_predictability.parquet"), index=False)
    print(df.groupby("dataset").auroc.median())


if __name__ == "__main__":
    main(sys.argv[1:] or ["goemotions", "dbpedia", "reviews", "pubmed"])
