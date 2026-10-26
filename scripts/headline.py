"""Headline left/right results for one or more models, from saved results only.

Behavior uses the judge: Llama Guard 3 with the election-delegitimization
category (results/<model>/guard3-election.jsonl), where "unsafe" means the
reply did the task. Projection and geometry come from the geometry stage.

  behavior   Democrat vs Republican, and aligned vs crossed (asking about your
             own side's claim vs the other side's), base and ablated, with
             exact McNemar tests on matched pairs; rates by severity
  projection the same two comparisons on the refusal-direction projection,
             with sign-flip permutation tests
  geometry   lowest cos(r_D, r_R) over layers, and how many layers have a
             D-minus-R gap aligned with the refusal direction beyond chance

Writes results/headline_behavior.csv and prints everything.

Usage: uv run python scripts/headline.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from grade import mcnemar  # noqa: E402
from summarize import sign_flip_p  # noqa: E402

JUDGE = "guard3-election"


def paired(df, value, a, b):
    """Values for groups a and b (masks), matched on pair_id."""
    x = df[a].set_index("pair_id")[value]
    return x, df[b].set_index("pair_id")[value].reindex(x.index)


def comparisons(st):
    return [("Republican vs Democrat", st.identity == "republican", st.identity == "democrat"),
            ("crossed vs aligned", st.alignment == "crossed", st.alignment == "aligned")]


def behavior(model):
    g = pd.read_json(ROOT / "results" / model / f"{JUDGE}.jsonl", lines=True)
    st = g[g.kind == "stolen"]
    rows = []
    for cond in ["base", "ablated"]:
        v = f"unsafe_{cond}"
        rows.append({"model": model, "condition": cond, "comparison": "overall", "subset": "all",
                     "A": st[v].mean(), "B": g[g.kind == "benign"][v].mean(), "p": np.nan})
        for side in ["all", "R", "D"]:
            sub = st if side == "all" else st[st.claim_side == side]
            for name, a, b in comparisons(sub):
                x, y = paired(sub, v, a, b)
                rows.append({"model": model, "condition": cond, "comparison": name, "subset": f"{side}-coded",
                             "A": x.mean(), "B": y.mean(),
                             "p": mcnemar(int((x & ~y).sum()), int((y & ~x).sum()))})
        for s, rate in st.groupby("severity")[v].mean().items():
            rows.append({"model": model, "condition": cond, "comparison": "severity", "subset": f"level {s}",
                         "A": rate, "B": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def projection(model):
    geo = pd.read_csv(ROOT / "results" / model / "geometry_per_prompt.csv")
    st = geo[geo.kind == "stolen"]
    gap = st[st.identity != "none"].groupby("pair_id").proj.mean().mean() - \
        geo[geo.kind == "benign"].proj.mean()
    print(f"  stolen minus benign projection: {gap:.2f} (scale for the gaps below)")
    for name, a, b in comparisons(st):
        x, y = paired(st, "proj", a, b)
        d = (x - y).values
        print(f"  {name:24s} mean gap {d.mean():+.3f}  p={sign_flip_p(d):.4f}")


def geometry(model):
    geo = pd.read_csv(ROOT / "results" / model / "party_directions_by_layer.csv")
    above = (geo.cos_delta_anchor.abs() > geo.cos_delta_anchor_null95).sum()
    print(f"  min cos(r_D, r_R) over layers: {geo.cos_D_R.min():.4f}")
    print(f"  layers where the D-minus-R gap lines up with the refusal direction beyond chance: "
          f"{above} of {len(geo)}")


def main():
    models = sys.argv[1:] or ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
    out = []
    pd.set_option("display.width", 200)
    for m in models:
        print(f"\n===== {m}")
        b = behavior(m)
        out.append(b)
        print("behavior (A vs B = 'did the task' rate; overall: A = stolen, B = benign)")
        print(b.round(3).to_string(index=False))
        print("projection onto the refusal direction")
        projection(m)
        print("geometry")
        geometry(m)
    pd.concat(out).to_csv(ROOT / "results" / "headline_behavior.csv", index=False)


if __name__ == "__main__":
    main()
