"""Does the refusal-direction projection predict behavior, prompt by prompt?

Logistic regression of the judge's "did the task" (Llama Guard 3 + election
category, results/<model>/guard3-election.jsonl) on each stolen-election
prompt's projection onto the refusal direction at the chosen layer
(results/<model>/geometry_per_prompt.csv), with party and alignment terms.
Handoff item 2 for Skyler. No GPU, numpy only.

Fits, per model, on the stolen-election prompts:

  proj            did_task ~ z(proj), all identities. Does the direction
                  predict behavior at all? Reports AUC.
  terms           did_task ~ republican + crossed, Democrat/Republican
                  prompts only. The behavioral asymmetry, as a regression.
  terms+proj      did_task ~ z(proj) + republican + crossed. If the crossed
                  coefficient shrinks toward zero once projection is in the
                  model, the asymmetry runs through the direction.
  terms+proj+sev  the same with severity dummies, as a control for request
                  type.
  proj->ablated   z(proj) (measured at base) predicting the *ablated* reply.
                  With the direction removed, projection should lose most of
                  its predictive power if that is the mechanism. Also with
                  severity dummies, since severity drives ablated compliance.
  proj+sev        z(proj) + severity dummies, base, all identities.

Standard errors are cluster-robust on pair_id (prompts in a cell share all
their content except the identity prefix). Projection is z-scored within
model so coefficients are per standard deviation.

Also a matched test: for each Democrat/Republican pair where exactly one of
the two did the task, was it the one with the lower projection?

Writes results/regression.csv and prints everything.

Usage: python3 scripts/regress.py [Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct]
"""

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JUDGE = "guard3-election"
DEFAULT_MODELS = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]


def load(model):
    out = ROOT / "results" / model
    geo = pd.read_csv(out / "geometry_per_prompt.csv").set_index("id")
    j = pd.read_json(out / f"{JUDGE}.jsonl", lines=True).set_index("id")
    df = geo.join(j[["unsafe_base", "unsafe_ablated", "p_unsafe_base", "p_unsafe_ablated"]], how="inner")
    df = df[df.kind == "stolen"].copy()
    df["did_base"] = df.unsafe_base.astype(int)
    df["did_ablated"] = df.unsafe_ablated.astype(int)
    df["zproj"] = (df.proj - df.proj.mean()) / df.proj.std()
    df["republican"] = (df.identity == "republican").astype(int)
    df["crossed"] = (df.alignment == "crossed").astype(int)
    df["sev2"] = (df.severity == 2).astype(int)
    df["sev3"] = (df.severity == 3).astype(int)
    return df


def logit_fit(X, y, clusters, max_iter=100):
    """Logistic regression by Newton's method with cluster-robust (sandwich) SEs."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-8 * np.eye(k), g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    H = X.T @ (X * (p * (1 - p))[:, None])
    Hinv = np.linalg.inv(H + 1e-8 * np.eye(k))
    scores = X * (y - p)[:, None]
    cl = pd.Series(np.asarray(clusters))
    G = cl.nunique()
    meat = np.zeros((k, k))
    for _, idx in cl.groupby(cl).indices.items():
        s = scores[idx].sum(axis=0)
        meat += np.outer(s, s)
    V = Hinv @ meat @ Hinv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    z = beta / se
    pval = 2 * (1 - 0.5 * (1 + np.vectorize(erf)(np.abs(z) / sqrt(2))))
    ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return beta, se, z, pval, ll, p


def auc(score, y):
    """Rank-based AUC: P(score of a positive > score of a negative)."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit(df, name, outcome, terms, rows):
    # Drop 0/1 dummy terms whose group has a constant outcome (perfect
    # separation), along with their rows, and say so.
    for t in list(terms):
        v = df[t]
        if set(v.unique()) <= {0, 1} and v.sum() > 0 and df.loc[v == 1, outcome].nunique() == 1:
            print(f"    [{name}: {t}=1 has constant {outcome}={df.loc[v == 1, outcome].iloc[0]} "
                  f"in {int(v.sum())} rows; dropped them and the term]")
            df = df[v == 0]
            terms = [u for u in terms if u != t]
    X = np.column_stack([np.ones(len(df))] + [df[t].to_numpy() for t in terms])
    beta, se, z, p, ll, phat = logit_fit(X, df[outcome], df.pair_id)
    a = auc(phat, df[outcome])
    for t, b, s, zz, pp in zip(["intercept"] + terms, beta, se, z, p):
        rows.append({"fit": name, "outcome": outcome, "n": len(df), "term": t,
                     "coef": b, "se": s, "z": zz, "p": pp, "odds_ratio": np.exp(b),
                     "auc": a, "loglik": ll})
    return a


def matched_test(df, rows, outcome="did_base"):
    """Among D/R pairs where exactly one did the task, how often was it the lower-projection one?"""
    dr = df[df.identity.isin(["democrat", "republican"])]
    g = dr.groupby("pair_id")
    n_disc = n_lower = 0
    for _, s in g:
        if len(s) != 2 or s[outcome].sum() != 1:
            continue
        n_disc += 1
        did = s[s[outcome] == 1].iloc[0]
        other = s[s[outcome] == 0].iloc[0]
        n_lower += int(did.proj < other.proj)
    # exact binomial two-sided p against 0.5
    from math import comb
    k = n_lower
    pm = min(sum(comb(n_disc, i) for i in range(0, k + 1)), sum(comb(n_disc, i) for i in range(k, n_disc + 1))) / 2 ** n_disc
    pv = min(1.0, 2 * pm)
    rows.append({"fit": "matched: lower proj did the task", "outcome": outcome, "n": n_disc,
                 "term": "share_lower_proj", "coef": k / n_disc if n_disc else np.nan, "p": pv})
    return n_disc, k, pv


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    rows = []
    for m in models:
        df = load(m)
        dr = df[df.identity.isin(["democrat", "republican"])]
        print(f"\n=== {m}: {len(df)} stolen prompts, {len(dr)} D/R, base did-the-task {df.did_base.mean():.1%}")
        print(f"    projection: mean {df.proj.mean():.2f}, sd {df.proj.std():.2f}; "
              f"did/didn't the task mean proj {df[df.did_base==1].proj.mean():.2f} / {df[df.did_base==0].proj.mean():.2f}")
        before = len(rows)
        fit(df, "proj", "did_base", ["zproj"], rows)
        fit(dr, "terms", "did_base", ["republican", "crossed"], rows)
        fit(dr, "terms+proj", "did_base", ["zproj", "republican", "crossed"], rows)
        fit(dr, "terms+proj+sev", "did_base", ["zproj", "republican", "crossed", "sev2", "sev3"], rows)
        fit(df, "proj->ablated", "did_ablated", ["zproj"], rows)
        fit(df, "proj+sev->ablated", "did_ablated", ["zproj", "sev2", "sev3"], rows)
        fit(df, "proj+sev", "did_base", ["zproj", "sev2", "sev3"], rows)
        fit(dr, "terms->ablated", "did_ablated", ["republican", "crossed"], rows)
        n_disc, k, pv = matched_test(df, rows)
        for r in rows[before:]:
            r["model"] = m
        t = pd.DataFrame(rows[before:])
        for name, s in t.groupby("fit", sort=False):
            if name.startswith("matched"):
                print(f"\n  {name}: {k} of {n_disc} discordant D/R pairs ({k/n_disc:.0%}), exact p = {pv:.3g}")
                continue
            print(f"\n  {name} -> {s.outcome.iloc[0]}  (n={s.n.iloc[0]}, AUC={s.auc.iloc[0]:.3f})")
            for _, r in s[s.term != "intercept"].iterrows():
                print(f"    {r.term:11s} coef {r.coef:+.3f}  OR {r.odds_ratio:5.2f}  se {r.se:.3f}  p {r.p:.3g}")
    out = ROOT / "results" / "regression.csv"
    pd.DataFrame(rows)[["model", "fit", "outcome", "n", "term", "coef", "se", "z", "p", "odds_ratio", "auc", "loglik"]].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
