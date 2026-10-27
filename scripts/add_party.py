"""Addition by party: does adding the refusal direction induce refusal equally
for Democrat and Republican askers? (Handoff item 1 for Skyler.)

Ablation removes refusal for everyone, so it is a weak test of symmetry. This
goes the other way: take the benign election questions (which nobody refuses),
add the refusal direction at the chosen layer (same hook and scale as the
direction stage's add_harmless check: coeff * diff-of-means direction, every
position), and measure how much refusal that induces per identity.

Two measures per prompt, base and added:
  score   the logit refusal score (log P(refusal first token) - log P(other)),
          continuous, so paired tests have power even when few replies flip
  reply   greedy generation, so the Llama Guard / Claude judges can score it
          (guard.py --input add_party.jsonl; judge_claude.py --input add_party.jsonl)

Writes results/<model>/add_party.jsonl and add_party_test.csv (paired
Democrat-vs-Republican and crossed-vs-aligned differences in induced score,
with sign-flip permutation p, plus keyword refusal rates). --coeffs sweeps
the addition strength; the first value is the one written to the replies.

Usage:
  uv run python scripts/add_party.py --model Qwen/Qwen2.5-7B-Instruct
  uv run python scripts/add_party.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 2   # smoke test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from model import DEFAULT_MODEL, Model, classify  # noqa: E402
from run import SEED, load_political  # noqa: E402
from summarize import sign_flip_p  # noqa: E402


def paired_diff(df, col, a_mask, b_mask):
    x = df[a_mask].set_index("pair_id")[col]
    y = df[b_mask].set_index("pair_id")[col].reindex(x.index)
    d = (x - y).dropna().values
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--kind", default="benign", choices=["benign", "stolen", "both"])
    ap.add_argument("--coeffs", nargs="+", type=float, default=[1.0, 0.5, 2.0])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="prompts per (kind, identity), 0 = all")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    out = ROOT / "results" / args.model.split("/")[-1]
    torch.manual_seed(SEED)
    d = torch.load(out / "direction.pt")
    layer, r = int(d["layer"]), d["direction"]

    df = load_political()
    if args.kind != "both":
        df = df[df.kind == args.kind]
    if args.limit:
        df = df.groupby(["kind", "identity"]).head(args.limit)
    df = df.reset_index(drop=True)
    instr = df.instruction.tolist()
    m = Model(args.model)
    print(f"{len(df)} prompts, direction layer {layer}, norm {r.norm():.2f}, coeffs {args.coeffs}")

    df["score_base"] = m.refusal_score(instr, bs=args.bs).numpy()
    for c in args.coeffs:
        with m.add(r, layer, coeff=c):
            df[f"score_added_{c:g}"] = m.refusal_score(instr, bs=args.bs).numpy()
        print(f"coeff {c:g}: mean score base {df.score_base.mean():.2f} -> added {df[f'score_added_{c:g}'].mean():.2f}")
    c0 = args.coeffs[0]
    df["score_added"] = df[f"score_added_{c0:g}"]
    df["reply_base"] = m.generate(instr, max_new_tokens=args.max_new_tokens, bs=args.bs)
    with m.add(r, layer, coeff=c0):
        df["reply_added"] = m.generate(instr, max_new_tokens=args.max_new_tokens, bs=args.bs)
    df["label_base"] = df.reply_base.map(classify)
    df["label_added"] = df.reply_added.map(classify)
    df["coeff"] = c0
    df.to_json(out / "add_party.jsonl", orient="records", lines=True)

    rows = []
    for kind, sub in df.groupby("kind"):
        comps = [("Republican minus Democrat", sub.identity == "republican", sub.identity == "democrat")]
        if kind == "stolen":
            comps.append(("crossed minus aligned", sub.alignment == "crossed", sub.alignment == "aligned"))
        for name, a, b in comps:
            for c in args.coeffs:
                col = f"score_added_{c:g}"
                dd = paired_diff(sub, col, a, b)
                dd0 = paired_diff(sub, "score_base", a, b)
                rows.append({"kind": kind, "comparison": name, "coeff": c, "n_pairs": len(dd),
                             "mean_diff_added": dd.mean(), "p_added": sign_flip_p(dd),
                             "mean_diff_base": dd0.mean(), "p_base": sign_flip_p(dd0)})
        for ident, s in sub.groupby("identity"):
            rows.append({"kind": kind, "comparison": f"rate:{ident}", "coeff": c0, "n_pairs": len(s),
                         "refuse_base": (s.label_base == "refuse").mean(),
                         "refuse_added": (s.label_added == "refuse").mean(),
                         "score_base": s.score_base.mean(), "score_added": s.score_added.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(out / "add_party_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
