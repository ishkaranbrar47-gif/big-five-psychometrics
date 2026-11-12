import numpy as np
import pandas as pd
import pytest

import analyze
from psychometrics.factors import bartlett_sphericity, kmo
from psychometrics.groups import cohens_d
from psychometrics.reliability import cronbach_alpha, item_analysis
from psychometrics.scoring import ITEMS, TRAITS, clean, load, reverse_score, score_traits

DATA = "data/bfi.csv"


# --- scoring -------------------------------------------------------------------


def test_reverse_score_flips_six_point_scale():
    df = pd.DataFrame({"A1": [1, 6, 3], "A2": [1, 6, 3]})
    out = reverse_score(df, items={"A1"})
    assert out["A1"].tolist() == [6, 1, 4]
    assert out["A2"].tolist() == [1, 6, 3]  # untouched


def test_score_needs_minimum_answered_items():
    row = {i: np.nan for i in ITEMS}
    row.update({"A2": 5, "A3": 5})  # only 2 agreeableness items answered
    scores = score_traits(pd.DataFrame([row]), min_items=3)
    assert np.isnan(scores.loc[0, "Agreeableness"])


def test_clean_drops_implausible_ages_and_out_of_range_answers():
    base = {i: 3 for i in ITEMS}
    df = pd.DataFrame([{**base, "age": 25}, {**base, "age": 5}, {**base, "A1": 9, "age": 30}])
    out, log = clean(df)
    assert len(out) == 1
    assert log["implausible_age"] == 1
    assert log["out_of_range_items"] == 1


# --- reliability -----------------------------------------------------------------


def test_alpha_matches_hand_calculation():
    items = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 2, 3, 5, 5], "c": [1, 3, 3, 4, 4]})
    k = 3
    expected = k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))
    assert cronbach_alpha(items) == pytest.approx(expected)


def test_alpha_is_one_for_identical_items():
    items = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    assert cronbach_alpha(items) == pytest.approx(1.0)


def test_alpha_rejects_single_item():
    with pytest.raises(ValueError):
        cronbach_alpha(pd.DataFrame({"a": [1, 2, 3]}))


def test_item_analysis_columns():
    items = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 2, 3, 5, 5], "c": [5, 3, 3, 2, 1]})
    out = item_analysis(items)
    assert list(out.columns) == ["item", "mean", "sd", "item_rest_r", "alpha_if_dropped"]
    # c runs against the other two, so it should have the weakest item-rest correlation
    assert out.set_index("item")["item_rest_r"].idxmin() == "c"


# --- factor suitability -----------------------------------------------------------


def test_kmo_and_bartlett_on_independent_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(2000, 6))
    _, p = bartlett_sphericity(noise)
    assert p > 0.01  # uncorrelated data should not look factorable
    assert kmo(noise) < 0.6


# --- effect size --------------------------------------------------------------------


def test_cohens_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 5.0])
    assert cohens_d(a, b) == pytest.approx(0.5)


# --- real data: results should match the values published for psych::bfi ----------


@pytest.fixture(scope="module")
def bfi():
    df, _ = clean(load(DATA))
    return df


@pytest.mark.parametrize(
    "trait, published",
    [("Agreeableness", 0.70), ("Conscientiousness", 0.72), ("Extraversion", 0.76), ("Neuroticism", 0.81), ("Openness", 0.60)],
)
def test_alphas_match_published_values(bfi, trait, published):
    keyed = reverse_score(bfi)
    assert cronbach_alpha(keyed[TRAITS[trait]]) == pytest.approx(published, abs=0.015)


def test_end_to_end_report(tmp_path, capsys):
    analyze.main(["--data", DATA, "--out", str(tmp_path)])
    for name in ("report.md", "reliability.png", "loadings.png", "gender_differences.png", "item_analysis.csv"):
        assert (tmp_path / name).exists(), name
    out = capsys.readouterr().out
    assert "25/25 items load on their intended factor" in out
