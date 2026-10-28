"""Decompose the alignment direction: which part carries the gap?

CAUSAL_TESTS.md section D: in Qwen, shifting prompts by the alignment
direction a swaps the aligned/crossed compliance rates, but shifting along
the refusal direction alone (projection matching) does nothing, and a has
cos 0.6 with refusal. So the inference is that a's component orthogonal to
the refusal direction does the work. This tests it directly.

For the Democrat/Republican stolen prompts, generate four conditions, each
with aligned prompts shifted by +v and crossed by -v at the chosen layer:
  shift_par    v = a's component along the unit refusal direction
  shift_perp   v = a's component orthogonal to it
  shift_full   v = a  (replicates alignment_dir.py's shift)
  shift_rand   v = random direction with |v| = |a|, orthogonal to both
Writes results/<model>/alignment_decompose.jsonl (+ reply_base). Judge with
guard.py / judge_claude.py --input alignment_decompose.jsonl, then compare
the aligned-vs-crossed gap per condition.

Needs direction.pt and political_acts.pt (both on the pod from the last run)
and the model. About 5 minutes per model.

Usage: uv run python scripts/alignment_decompose.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from model import DEFAULT_MODEL, Model  # noqa: E402
from run import SEED, load_political  # noqa: E402
from alignment_dir import cell_diffs, shift  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="prompts per (kind, alignment), 0 = all")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--kind", default="stolen", choices=["stolen", "both"])
    args = ap.parse_args()
    out = ROOT / "results" / args.model.split("/")[-1]
    torch.manual_seed(SEED)
    d = torch.load(out / "direction.pt")
    layer, pos = int(d["layer"]), int(d["pos"])
    pa = torch.load(out / "political_acts.pt")
    acts = pa["acts"].float()
    df = load_political().reset_index(drop=True)
    assert df.id.tolist() == pa["ids"]
    df["row"] = np.arange(len(df))
    ds, _ = cell_diffs(df, acts, "stolen")
    db, _ = cell_diffs(df, acts, "benign")
    a = ds.mean(0)[layer] - db.mean(0)[layer]
    u = d["direction"] / d["direction"].norm()
    a_par = (a @ u) * u
    a_perp = a - a_par
    g = torch.Generator().manual_seed(SEED + 1)
    r = torch.randn(a.shape, generator=g)
    r = r - (r @ u) * u
    r = r - (r @ a_perp) * a_perp / (a_perp.norm() ** 2)
    a_rand = r / r.norm() * a.norm()
    vecs = {"shift_par": a_par, "shift_perp": a_perp, "shift_full": a, "shift_rand": a_rand}
    info = {"layer": layer, "pos": pos, "a_norm": float(a.norm()), "a_par_norm": float(a_par.norm()),
            "a_perp_norm": float(a_perp.norm()), "cos_a_refusal": float(a @ u / a.norm()),
            "refusal_norm": float(d["direction"].norm())}
    print(json.dumps(info, indent=1), flush=True)

    sub = df[df.alignment.isin(["aligned", "crossed"])]
    if args.kind == "stolen":
        sub = sub[sub.kind == "stolen"]
    if args.limit:
        sub = sub.groupby(["kind", "alignment"]).head(args.limit)
    sub = sub.reset_index(drop=True).copy()
    instr = sub.instruction.tolist()
    signs_all = torch.tensor([1.0 if x == "aligned" else -1.0 for x in sub.alignment])
    m = Model(args.model)
    base = pd.read_json(out / "behavior.jsonl", lines=True).set_index("id") if (out / "behavior.jsonl").exists() else None
    sub["reply_base"] = [base.reply_base.get(i) for i in sub.id] if base is not None else m.generate(instr, max_new_tokens=args.max_new_tokens, bs=args.bs)
    for name, v in vecs.items():
        cur = [None]
        outs = []
        with shift(m, v, layer, cur):
            for i in range(0, len(instr), args.bs):
                cur[0] = signs_all[i:i + args.bs]
                outs += m.generate(instr[i:i + args.bs], max_new_tokens=args.max_new_tokens, bs=args.bs)
        sub[f"reply_{name}"] = outs
        print(f"{name}: |v| = {v.norm():.3f}, done", flush=True)
    sub.to_json(out / "alignment_decompose.jsonl", orient="records", lines=True)
    json.dump(info, open(out / "alignment_decompose.json", "w"), indent=2)
    print(f"wrote {out / 'alignment_decompose.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
