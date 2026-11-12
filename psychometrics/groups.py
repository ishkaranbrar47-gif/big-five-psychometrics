"""Group differences with effect sizes, not just p-values."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cohens_d(a, b):
    """Standardised mean difference using the pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def compare_groups(scores, groups, label_a, label_b):
    """Welch t-test per trait, Cohen's d, and Holm-corrected p-values."""
    rows = []
    for trait in scores.columns:
        a = scores.loc[groups == label_a, trait].dropna()
        b = scores.loc[groups == label_b, trait].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"trait": trait, "mean_a": a.mean(), "mean_b": b.mean(), "d": cohens_d(a, b), "t": t, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


def age_correlations(scores, age):
    rows = []
    for trait in scores.columns:
        ok = scores[trait].notna() & age.notna()
        r, p = stats.pearsonr(scores.loc[ok, trait], age[ok])
        rows.append({"trait": trait, "r": r, "p": p, "n": int(ok.sum())})
    return pd.DataFrame(rows)
