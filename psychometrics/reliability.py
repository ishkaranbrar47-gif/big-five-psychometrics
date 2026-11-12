"""Internal consistency: Cronbach's alpha and item diagnostics."""

import numpy as np
import pandas as pd


def cronbach_alpha(items):
    """alpha = k/(k-1) * (1 - sum(item variances) / variance(total)), listwise complete cases."""
    data = pd.DataFrame(items).dropna()
    k = data.shape[1]
    if k < 2:
        raise ValueError("Alpha needs at least two items.")
    item_var = data.var(axis=0, ddof=1).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def item_analysis(items):
    """Per item: mean, SD, corrected item-total correlation, alpha if the item were dropped."""
    data = pd.DataFrame(items).dropna()
    rows = []
    for col in data.columns:
        rest = data.drop(columns=col)
        rows.append(
            {
                "item": col,
                "mean": data[col].mean(),
                "sd": data[col].std(ddof=1),
                "item_rest_r": np.corrcoef(data[col], rest.sum(axis=1))[0, 1],
                "alpha_if_dropped": cronbach_alpha(rest) if rest.shape[1] >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).round(3)
