"""Cross-model summary: party-label effect vs. aligned/crossed effect.

Two different ways identity could matter:
  party label   Democrat vs Republican, whatever the claim
  alignment     asking about your own side's claim (aligned: Republican on
                2020, Democrat on 2016) vs the other side's claim (crossed)

Every (election, template) cell has one Democrat and one Republican prompt,
and exactly one of them is aligned, so both comparisons are paired.

Usage: uv run python scripts/summarize.py Qwen2.5-3B-Instruct Llama-3.2-3B-Instruct
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from grade import mcnemar  # noqa: E402

SEED = 0


def sign_flip_p(diff, n=10000):
    rng = np.random.default_rng(SEED)
    null = np.array([(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(n)])
    return (np.abs(null) >= abs(diff.mean())).mean()


def paired(df, value, a_mask, b_mask):
    a = df[a_mask].set_index("pair_id")[value]
    b = df[b_mask].set_index("pair_id")[value].reindex(a.index)
    return a, b


def summarize(model):
    d = ROOT / "results" / model
    geo = pd.read_csv(d / "geometry_per_prompt.csv")
    beh = pd.read_json(d / "behavior_graded.jsonl", lines=True)
    rows = []
    for kind in ["stolen", "benign"]:
        g = geo[geo.kind == kind]
        b = beh[beh.kind == kind]
        for name, a_mask, b_mask in [
            ("Republican minus Democrat", lambda x: x.identity == "republican", lambda x: x.identity == "democrat"),
            ("crossed minus aligned", lambda x: x.alignment == "crossed", lambda x: x.alignment == "aligned"),
        ]:
            pa, pb = paired(g, "proj", a_mask(g), b_mask(g))
            row = {"model": model, "kind": kind, "comparison": name, "n_pairs": len(pa),
                   "proj_diff": (pa - pb).mean(), "proj_p": sign_flip_p((pa - pb).values)}
            for cond in ["base", "ablated"]:
                ra, rb = paired(b, f"no_task_{cond}", a_mask(b), b_mask(b))
                row[f"refuse_{cond}_A"] = ra.mean()
                row[f"refuse_{cond}_B"] = rb.mean()
                row[f"refuse_{cond}_p"] = mcnemar(int((ra & ~rb).sum()), int((rb & ~ra).sum()))
            rows.append(row)
    return rows


def main():
    models = sys.argv[1:] or ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]
    out = pd.DataFrame([r for m in models for r in summarize(m)])
    out.to_csv(ROOT / "results" / "summary_identity_effects.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
