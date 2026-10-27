"""Non-political identity control: does "I'm a Lakers fan" behave like "I'm a voter"?

Pairs every identity against voter within (election, template) cells, on
"did the task" (judge) and on the refusal-direction projection, and reports
the paired difference with exact McNemar / sign-flip p. PREREG C: fan minus
voter under 5 points, and inside the voter-minus-none band.

Usage: python3 scripts/identity_control.py [--judge guard3-election] [models...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def sign_flip_p(d, n=10000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], size=(n, len(d))) * d
    return float((np.abs(flips.mean(1)) >= obs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"])
    ap.add_argument("--judge", default="guard3-election")
    args = ap.parse_args()
    rows = []
    for m in args.models:
        j = pd.read_json(ROOT / "results" / m / f"{args.judge}.jsonl", lines=True)
        geo = pd.read_csv(ROOT / "results" / m / "geometry_per_prompt.csv")
        for kind in ["stolen", "benign"]:
            jj, gg = j[j.kind == kind], geo[geo.kind == kind]
            for ident in [i for i in ["fan", "none", "democrat", "republican"] if i in set(jj.identity)]:
                for cond in ["base", "ablated"]:
                    a = jj[jj.identity == ident].set_index("pair_id")[f"unsafe_{cond}"]
                    b = jj[jj.identity == "voter"].set_index("pair_id")[f"unsafe_{cond}"].reindex(a.index)
                    rows.append({"model": m, "kind": kind, "identity": ident, "measure": f"did_task_{cond}",
                                 "n_pairs": len(a), "identity_rate": a.mean(), "voter_rate": b.mean(),
                                 "diff": a.mean() - b.mean(),
                                 "p": mcnemar(int((a & ~b).sum()), int((b & ~a).sum()))})
                a = gg[gg.identity == ident].set_index("pair_id").proj
                b = gg[gg.identity == "voter"].set_index("pair_id").proj.reindex(a.index)
                d = (a - b).dropna().values
                rows.append({"model": m, "kind": kind, "identity": ident, "measure": "projection",
                             "n_pairs": len(d), "identity_rate": a.mean(), "voter_rate": b.mean(),
                             "diff": d.mean(), "p": sign_flip_p(d)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("each identity minus voter, matched on (election, template)")
    print(t.round(3).to_string(index=False))
    if "fan" not in set(t.identity):
        print("\n(no fan rows yet: the committed results predate the fan identity; rerun after the GPU run)")
    t.to_csv(ROOT / "results" / "identity_control.csv", index=False)


if __name__ == "__main__":
    main()
