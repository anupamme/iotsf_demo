#!/usr/bin/env python3
"""
The clustering level for every cross-cell interval in the paper, defined once.

WHY THIS MODULE EXISTS. Cells are not independent observations. A cell is a (backbone, dataset,
size, horizon) combination, so the 31 cells reuse 3 backbones and 7 series between them: the Moirai
Small/Base/Large runs on ETTh2 at h=96 and h=192 share the series, the split, the normalisation
constants and -- for a given size -- the pretrained checkpoint. Resampling CELLS with replacement
treats those as independent draws and produces an interval that is too narrow, which is the standard
objection to a cell-level bootstrap and the reason a p-value computed that way is not conventional
inferential evidence.

The level at which the dependence lives is the (backbone, dataset) pair: shared pretraining corpus,
shared series, shared checkpoint family. Resampling THAT with replacement propagates it. The
intervals get wider; that is not a defect of the method, it is the dependence becoming visible.

WHY IT IS FACTORED OUT. cka_fixed_effects.py already implemented exactly this cluster definition for
its backbone fixed-effects fit, and cell_matrix.py needs the same definition for its rank
correlations. Two copies of a cluster key would drift -- one gaining a dataset alias the other
lacks -- and the two analyses would then silently cluster at different levels while both claiming to
cluster at "(backbone, dataset)". This module is imported by both, and by nothing that either of them
imports, so there is no cycle.

WHAT A WIDE CLUSTER INTERVAL DOES NOT MEAN. It does not establish absence of a relationship. With 7
series and 3 backbones the effective sample size for a cross-cell statistic is closer to 10 than to
31, and an interval that includes zero at that size means the data cannot pin the sign down. That is
weaker than "no relationship" and must be stated as the weaker thing.
"""
import re

import numpy as np

# Series-name aliases. Electricity7 must be matched BEFORE Electricity, or the shorter alternative
# wins and the two are silently merged into one cluster. Electricity7.csv is a symlink to
# Electricity.csv, so they are in fact the same series -- but the merge should be a decision recorded
# here, not a side effect of alternation order.
DATASET_RE = re.compile(r"(ETTh1|ETTh2|ETTm2|Weather|Electricity7|Electricity|ILI|M4)",
                        re.IGNORECASE)

BACKBONES = ("Moirai", "Chronos", "TimesFM")


def backbone_of(cell):
    """Backbone family from a cell label. Moirai is the default because its labels carry a size."""
    for bb in BACKBONES[1:]:
        if cell.startswith(bb):
            return bb
    return BACKBONES[0]


def dataset_of(cell):
    """The series a cell is fitted on, however the cell label is spelled."""
    m = DATASET_RE.search(cell)
    return m.group(1).lower() if m else "other"


def cluster_of(cell):
    """The unit of resampling: one backbone's runs on one series."""
    return f"{backbone_of(cell)}|{dataset_of(cell)}"


def clusters_for(cells):
    return np.array([cluster_of(c) for c in cells])


def cluster_bootstrap_spearman(x, y, clusters, n_boot=10000, rng=None, min_distinct=3):
    """Spearman rho with a 95% interval from resampling clusters with replacement.

    Returns (rho, lo, hi, kept, n_clusters). `kept` is the number of draws that were usable: a draw
    that happens to select clusters spanning fewer than `min_distinct` distinct predictor values
    cannot yield a rank correlation, and is discarded rather than counted as a rho of nan. Reporting
    `kept` is what makes the discard auditable -- if most draws are being thrown away, the interval
    is describing a small subset of the resampling distribution and the reader needs to know.
    """
    from scipy import stats

    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}

    rho = stats.spearmanr(x, y).statistic
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in draw])
        if len(set(x[idx])) < min_distinct:
            continue
        boots.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not boots:
        return rho, float("nan"), float("nan"), 0, len(uniq)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi), len(boots), len(uniq)


def cell_bootstrap_spearman(x, y, n_boot=10000, rng=None, min_distinct=3):
    """The cell-level bootstrap, kept only so it can be shown BESIDE the clustered one.

    This is the interval the paper reported in earlier rounds. It is retained not because it is
    defensible on its own but because the comparison is the evidence: printing both makes the cost of
    ignoring the dependence a number rather than a caveat.
    """
    from scipy import stats

    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) < min_distinct:
            continue
        boots.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not boots:
        return rho, float("nan"), float("nan"), 0
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi), len(boots)
