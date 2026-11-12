"""Loading, cleaning and scoring the Big Five items.

Items are answered on a 6-point scale (1 = very inaccurate, 6 = very accurate).
Reverse-keyed items are flipped with (min + max - x) before scoring, so that a
higher score always means more of the trait.
"""

import pandas as pd

TRAITS = {
    "Agreeableness": ["A1", "A2", "A3", "A4", "A5"],
    "Conscientiousness": ["C1", "C2", "C3", "C4", "C5"],
    "Extraversion": ["E1", "E2", "E3", "E4", "E5"],
    "Neuroticism": ["N1", "N2", "N3", "N4", "N5"],
    "Openness": ["O1", "O2", "O3", "O4", "O5"],
}
# Keys from the psych package for scoring Neuroticism (not Emotional Stability).
REVERSED = {"A1", "C4", "C5", "E1", "E2", "O2", "O5"}
SCALE_MIN, SCALE_MAX = 1, 6
ITEMS = [i for items in TRAITS.values() for i in items]
MIN_PLAUSIBLE_AGE = 11  # younger "ages" are almost certainly entry errors


def load(path):
    df = pd.read_csv(path)
    if "rownames" in df.columns:
        df = df.rename(columns={"rownames": "id"})
    missing = set(ITEMS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing item columns: {', '.join(sorted(missing))}")
    return df


def clean(df):
    """Drop rows with impossible values and report what was removed."""
    out = df.copy()
    log = {"start": len(out)}
    bad_items = ~out[ITEMS].isin(range(SCALE_MIN, SCALE_MAX + 1)) & out[ITEMS].notna()
    out = out[~bad_items.any(axis=1)]
    log["out_of_range_items"] = log["start"] - len(out)
    young = out["age"] < MIN_PLAUSIBLE_AGE
    log["implausible_age"] = int(young.sum())
    out = out[~young]
    log["kept"] = len(out)
    return out.reset_index(drop=True), log


def reverse_score(df, items=REVERSED, lo=SCALE_MIN, hi=SCALE_MAX):
    out = df.copy()
    for item in items:
        out[item] = (lo + hi) - out[item]
    return out


def score_traits(df, min_items=3):
    """Mean of each trait's items (after reversing). Needs at least `min_items` answered."""
    keyed = reverse_score(df)
    scores = pd.DataFrame(index=df.index)
    for trait, items in TRAITS.items():
        answered = keyed[items].notna().sum(axis=1)
        scores[trait] = keyed[items].mean(axis=1).where(answered >= min_items)
    return scores
