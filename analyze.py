"""Run the full psychometric analysis and write a report with charts.

Usage:
    python analyze.py --data data/bfi.csv --out reports
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from psychometrics.factors import five_factor_efa, label_factors, structure_check, suitability  # noqa: E402
from psychometrics.groups import age_correlations, compare_groups  # noqa: E402
from psychometrics.reliability import cronbach_alpha, item_analysis  # noqa: E402
from psychometrics.scoring import MIN_PLAUSIBLE_AGE, TRAITS, clean, load, reverse_score, score_traits  # noqa: E402

INK, ACCENT, MUTED = "#16202B", "#0E7C66", "#B8C2CC"


def fmt_p(p):
    return "< .001" if p < 0.001 else f"{p:.3f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def chart_alpha(alphas, path):
    s = pd.Series(alphas).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(s.index, s.values, color=[ACCENT if v >= 0.7 else MUTED for v in s.values])
    ax.axvline(0.7, color=INK, lw=1, ls="--")
    ax.text(0.705, -0.6, "0.70 rule of thumb", fontsize=8, color=INK)
    for i, v in enumerate(s.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cronbach's alpha")
    ax.set_title("Internal consistency of each Big Five scale")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_loadings(loadings, names, path):
    lab = loadings.rename(columns=names)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(lab.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(lab.shape[1]), lab.columns, rotation=30, ha="right")
    ax.set_yticks(range(lab.shape[0]), lab.index)
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            v = lab.values[i, j]
            if abs(v) >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Factor loadings (varimax rotation), |loading| >= 0.30 shown")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_gender(diff, path):
    d = diff.set_index("trait")["d"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(d.index, d.values, color=[ACCENT if v > 0 else MUTED for v in d.values])
    ax.axvline(0, color=INK, lw=1)
    for i, v in enumerate(d.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("Cohen's d (positive = higher in women)")
    ax.set_title("Gender differences in Big Five scores")
    ax.set_xlim(-0.6, 0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(data_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load(data_path)
    df, log = clean(raw)
    keyed = reverse_score(df)
    scores = score_traits(df)

    alphas = {t: round(cronbach_alpha(keyed[items]), 3) for t, items in TRAITS.items()}
    items = pd.concat([item_analysis(keyed[i]).assign(trait=t) for t, i in TRAITS.items()])
    weak = items[items["alpha_if_dropped"] > items["trait"].map(alphas)]

    suit = suitability(df)
    loadings, eigen = five_factor_efa(df)
    names = label_factors(loadings)
    check = structure_check(loadings)

    # gender: 1 = male, 2 = female in this dataset
    diff = compare_groups(scores, df["gender"], 2, 1)
    ages = age_correlations(scores, df["age"])

    chart_alpha(alphas, out / "reliability.png")
    chart_loadings(loadings, names, out / "loadings.png")
    chart_gender(diff, out / "gender_differences.png")
    scores.describe().round(2).to_csv(out / "trait_descriptives.csv")
    items.to_csv(out / "item_analysis.csv", index=False)
    loadings.round(3).rename(columns=names).to_csv(out / "loadings.csv")

    miss = check[~check["matches"]]
    diff_t = diff.assign(p_holm=diff["p_holm"].map(fmt_p)).round(3)
    ages_t = ages.assign(p=ages["p"].map(fmt_p)).round(3)
    report = f"""# Big Five Psychometric Report

Data: {log['start']:,} participants from the SAPA project (`psych::bfi`). Removed
{log['implausible_age']} rows with an implausible age (under {MIN_PLAUSIBLE_AGE}); kept {log['kept']:,}.

![Reliability](reliability.png)

## 1. Reliability (Cronbach's alpha)

{md_table(pd.DataFrame({'trait': list(alphas), 'alpha': list(alphas.values())}))}

Items whose removal would raise their scale's alpha:

{md_table(weak[['trait', 'item', 'item_rest_r', 'alpha_if_dropped']]) if len(weak) else 'None.'}

## 2. Factor structure

KMO = {suit['kmo']} and Bartlett's test chi2 = {suit['bartlett_chi2']:,} (p < .001) on
{suit['n']:,} complete cases, so the items are suitable for factor analysis. The first six
eigenvalues are {', '.join(f'{e:.2f}' for e in eigen[:6])}: five are above 1.

{int(check['matches'].sum())} of 25 items load most strongly on their intended trait.

{md_table(miss) if len(miss) else 'Every item loads on its intended trait.'}

![Loadings](loadings.png)

## 3. Gender differences

Welch t-tests, Holm-corrected. d > 0 means women score higher.

{md_table(diff_t[['trait', 'mean_a', 'mean_b', 'd', 'p_holm']].rename(columns={'mean_a': 'women', 'mean_b': 'men'}))}

![Gender](gender_differences.png)

## 4. Age

{md_table(ages_t)}
"""
    (out / "report.md").write_text(report)
    return {"log": log, "alphas": alphas, "matches": int(check["matches"].sum()), "diff": diff, "ages": ages, "suit": suit}


def main(argv=None):
    p = argparse.ArgumentParser(description="Psychometric analysis of the Big Five Inventory.")
    p.add_argument("--data", default="data/bfi.csv")
    p.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    res = run(a.data, a.out)
    print(f"Kept {res['log']['kept']:,} of {res['log']['start']:,} participants")
    for t, v in res["alphas"].items():
        print(f"  alpha {t:<18} {v:.3f}")
    print(f"  {res['matches']}/25 items load on their intended factor")
    print(f"Report written to {Path(a.out) / 'report.md'}")


if __name__ == "__main__":
    main()
