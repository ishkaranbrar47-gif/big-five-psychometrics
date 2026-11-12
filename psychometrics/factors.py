"""Exploratory factor analysis: does the data show the five intended traits?

KMO and Bartlett's test are implemented directly from their formulas, and the
factor model uses scikit-learn's maximum-likelihood FactorAnalysis with a
varimax rotation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

from .scoring import ITEMS, TRAITS, reverse_score

ITEM_TO_TRAIT = {i: t for t, items in TRAITS.items() for i in items}


def kmo(data):
    """Kaiser-Meyer-Olkin measure: squared correlations vs squared partial correlations."""
    r = np.corrcoef(np.asarray(data, float), rowvar=False)
    inv = np.linalg.inv(r)
    partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    off = ~np.eye(r.shape[0], dtype=bool)
    r2, p2 = (r[off] ** 2).sum(), (partial[off] ** 2).sum()
    return float(r2 / (r2 + p2))


def bartlett_sphericity(data):
    """Tests whether the correlation matrix differs from an identity matrix."""
    x = np.asarray(data, float)
    n, p = x.shape
    r = np.corrcoef(x, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(r))
    dof = p * (p - 1) / 2
    return float(chi2), float(stats.chi2.sf(chi2, dof))


def suitability(df):
    """KMO (> 0.6 acceptable, > 0.8 good) and Bartlett's test (p < .05 means factorable)."""
    data = reverse_score(df)[ITEMS].dropna()
    chi2, p = bartlett_sphericity(data)
    return {"n": len(data), "kmo": round(kmo(data), 3), "bartlett_chi2": round(chi2, 1), "bartlett_p": p}


def eigenvalues(df):
    data = reverse_score(df)[ITEMS].dropna()
    return np.sort(np.linalg.eigvalsh(np.corrcoef(data, rowvar=False)))[::-1]


def five_factor_efa(df, n_factors=5):
    """Maximum-likelihood factor analysis on standardised items, varimax-rotated."""
    data = reverse_score(df)[ITEMS].dropna()
    z = (data - data.mean()) / data.std(ddof=0)
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=0).fit(z)
    loadings = pd.DataFrame(fa.components_.T, index=ITEMS, columns=[f"F{i + 1}" for i in range(n_factors)])
    # Flip each factor so its strongest loadings are positive (sign is arbitrary in FA).
    for f in loadings.columns:
        if loadings[f].sum() < 0:
            loadings[f] = -loadings[f]
    return loadings, eigenvalues(df)


def label_factors(loadings):
    """Name each factor after the trait whose items load on it most strongly."""
    names = {}
    for f in loadings.columns:
        strength = loadings[f].abs().groupby(ITEM_TO_TRAIT).mean()
        names[f] = strength.idxmax()
    return names


def structure_check(loadings):
    """For each item, does its largest loading fall on the factor for its own trait?"""
    names = label_factors(loadings)
    rows = []
    for item in loadings.index:
        top = loadings.loc[item].abs().idxmax()
        rows.append(
            {
                "item": item,
                "intended": ITEM_TO_TRAIT[item],
                "loads_on": names[top],
                "loading": round(float(loadings.loc[item, top]), 2),
                "matches": names[top] == ITEM_TO_TRAIT[item],
            }
        )
    return pd.DataFrame(rows)
