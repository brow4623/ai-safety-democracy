"""Projection matching: does the refusal direction explain the gap between groups?

Ablation removes refusal almost everywhere, so it can't say whether the
direction explains a gap between two groups of prompts. This test (suggested
by the TA) changes one number per prompt instead:

  1. From the geometry stage, take each stolen-election prompt's projection
     onto the refusal direction (chosen layer and position).
  2. For a comparison (e.g. aligned vs crossed), the group with the lower mean
     projection is the source, the other the target.
  3. For each source prompt, add (target mean - its own projection) times the
     unit refusal direction to the residual stream at the input of the chosen
     layer, at every position, so its projection matches the target's average.
     Nothing else changes.
  4. Control: the same shift along a random unit direction orthogonal to the
     refusal direction. If that also moves behavior, the effect isn't specific.

Then judge the replies (guard.py --input patch_<comparison>.jsonl) and compare
the source's patched rate with the target's base rate. If they match, the gap
runs through the refusal direction; if not, it comes from somewhere else.

Writes results/<model>/patch_<comparison>.jsonl with reply_base (the source
prompt's original reply), reply_patched and reply_random.

Usage:
  uv run python scripts/patch.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import contextlib
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from model import DEFAULT_MODEL, Model  # noqa: E402
from run import SEED, load_political  # noqa: E402

COMPARISONS = {
    "alignment": ("alignment", "aligned", "crossed"),
    "party": ("identity", "democrat", "republican"),
}


@contextlib.contextmanager
def shift(m, unit, layer, deltas):
    """Add deltas[i] * unit to row i of the batch at the input of `layer`.

    `deltas` is a one-element list holding the current batch's tensor, so the
    caller can swap it between batches without re-registering the hook.
    """
    u = unit.to(m.device, m.model.dtype)

    def pre(_, args, kwargs):
        add = deltas[0].to(m.device, m.model.dtype)[:, None, None] * u
        if args:
            return (args[0] + add,) + args[1:], kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + add
        return args, kwargs

    h = m.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def generate_shifted(m, instructions, deltas, unit, layer, max_new_tokens, bs=16):
    current = [None]
    outs = []
    with shift(m, unit, layer, current):
        for i in range(0, len(instructions), bs):
            current[0] = torch.tensor(deltas[i:i + bs], dtype=torch.float32)
            outs += m.generate(instructions[i:i + bs], max_new_tokens=max_new_tokens, bs=bs)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--comparisons", nargs="+", default=list(COMPARISONS), choices=list(COMPARISONS))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="source prompts per comparison, 0 = all (for testing)")
    args = ap.parse_args()

    out = ROOT / "results" / args.model.split("/")[-1]
    torch.manual_seed(SEED)
    d = torch.load(out / "direction.pt")
    layer = d["layer"]
    unit = d["direction"] / d["direction"].norm()
    g = torch.Generator().manual_seed(SEED)
    rand = torch.randn(unit.shape, generator=g)
    rand = rand - (rand @ unit) * unit
    rand = rand / rand.norm()

    geo = pd.read_csv(out / "geometry_per_prompt.csv").set_index("id")
    prompts = load_political().set_index("id")
    base = pd.read_json(out / "behavior.jsonl", lines=True).set_index("id")
    m = Model(args.model)

    for name in args.comparisons:
        col, a, b = COMPARISONS[name]
        st = geo[geo.kind == "stolen"]
        mean = {v: st[st[col] == v].proj.mean() for v in (a, b)}
        source, target = (a, b) if mean[a] < mean[b] else (b, a)
        src = st[st[col] == source]
        if args.limit:
            src = src.head(args.limit)
        deltas = (mean[target] - src.proj).tolist()
        print(f"{name}: raise {source} (mean {mean[source]:.3f}) to {target} (mean {mean[target]:.3f}), "
              f"{len(src)} prompts, mean shift {sum(deltas) / len(deltas):.3f}")

        res = prompts.loc[src.index].reset_index()
        res["comparison"], res["source"], res["target"] = name, source, target
        res["proj"], res["delta"] = src.proj.values, deltas
        res["target_mean_proj"] = mean[target]
        res["reply_base"] = base.loc[src.index, "reply_base"].values
        instr = res.instruction.tolist()
        res["reply_patched"] = generate_shifted(m, instr, deltas, unit, layer, args.max_new_tokens)
        res["reply_random"] = generate_shifted(m, instr, deltas, rand, layer, args.max_new_tokens)
        res.to_json(out / f"patch_{name}.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
