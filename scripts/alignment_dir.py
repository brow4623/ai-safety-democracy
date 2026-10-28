"""Is there an "alignment direction", and does it overlap the refusal direction?

The behavioral finding is that both models help more when the asker's party
matches the claim's side (aligned) than when it doesn't (crossed). The
refusal direction predicts refusal but carries little or none of this gap
(NO_GPU_FOLLOWUPS.md; projection matching tests it causally). This asks what
does carry it, with the language extension's censorship-direction method
(scripts/lang_censordir.py):

    for each (election, template) cell:  a_cell = act[crossed] - act[aligned]
    a = mean over stolen cells - mean over benign cells

i.e. what being crossed adds on stolen-election prompts beyond what it adds
on benign ones. Within a cell the two prompts differ only in the party word,
so a is the model's representation of "this party is asking about the other
side's claim".

Geometry (needs results/<model>/political_acts.pt, no model):
  - cos(a_l, refusal_l) at every layer, with a sign-flip null over cells
  - |a| relative to |refusal|
  - every prompt's projection on a at the chosen layer, cross-fitted
    (direction from the other half of the cells), so it can be joined to the
    judges and regressed on did-the-task without by-construction inflation
Writes alignment_dir.json and alignment_projection.csv.

Causal (needs the model; skip with --no-generate):
  for all Democrat/Republican prompts (stolen and benign):
    reply_abl_align   a projected out of the residual stream everywhere
    reply_abl_random  a random direction of the same norm projected out
    reply_shift       aligned prompts get +a, crossed prompts get -a, at the
                      chosen layer (coeff 1): each prompt moved to the other
                      cell's alignment state
Writes alignment_dir.jsonl (with reply_base from behavior.jsonl). Judge with
guard.py --input alignment_dir.jsonl and judge_claude.py --input alignment_dir.jsonl.
If ablating a closes the aligned-vs-crossed gap and the random control does
not, the gap runs through a.

Usage:
  uv run python scripts/alignment_dir.py --model Qwen/Qwen2.5-7B-Instruct
  uv run python scripts/alignment_dir.py --model Qwen/Qwen2.5-7B-Instruct --no-generate   # geometry only
"""

import argparse
import contextlib
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


def cosv(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def cell_diffs(df, acts, kind):
    """[n_cells, L, D]: crossed minus aligned activations, one per (election, template) cell."""
    sub = df[(df.kind == kind) & df.alignment.isin(["aligned", "crossed"])]
    piv = sub.pivot(index="pair_id", columns="alignment", values="row")
    piv = piv.dropna()
    return acts[piv.crossed.astype(int).values] - acts[piv.aligned.astype(int).values], piv.index.tolist()


def geometry(out, df, acts, d, n_perm):
    L = acts.shape[1]
    layer = int(d["layer"])
    pos = int(d["pos"])
    ref_dirs = d["all_dirs"][:, pos]                                 # [L, D] anchor refusal direction per layer
    ds, cells_s = cell_diffs(df, acts, "stolen")                     # [136, L, D]
    db, cells_b = cell_diffs(df, acts, "benign")                     # [48, L, D]
    a_raw = ds.mean(0)                                               # [L, D]
    a = a_raw - db.mean(0)
    res = {"n_stolen_cells": len(cells_s), "n_benign_cells": len(cells_b), "layer": layer, "pos": pos,
           "layers": list(range(1, L)),
           "cos_align_vs_refusal_by_layer": [cosv(a[l], ref_dirs[l]) for l in range(1, L)],
           "cos_align_raw_vs_refusal_by_layer": [cosv(a_raw[l], ref_dirs[l]) for l in range(1, L)],
           "cos_benign_gap_vs_refusal_by_layer": [cosv(db.mean(0)[l], ref_dirs[l]) for l in range(1, L)],
           "align_norm_over_refusal_norm_by_layer": [float(a[l].norm() / (ref_dirs[l].norm() + 1e-8)) for l in range(1, L)],
           "cos_align_stolen_vs_benign_gap_by_layer": [cosv(a_raw[l], db.mean(0)[l]) for l in range(1, L)]}
    # Sign-flip null at the chosen layer: swap crossed/aligned within each stolen cell.
    g = torch.Generator().manual_seed(SEED)
    cp = ds[:, layer]
    null = []
    for _ in range(n_perm):
        s = (torch.randint(0, 2, (len(cp),), generator=g) * 2 - 1).float()
        null.append(cosv((cp * s[:, None]).mean(0) - db.mean(0)[layer], ref_dirs[layer]))
    null = np.array(null)
    obs = cosv(a[layer], ref_dirs[layer])
    res["chosen_layer"] = {"cos_align_vs_refusal": obs, "perm_null_95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                           "perm_p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
                           "align_norm": float(a[layer].norm()), "refusal_norm": float(ref_dirs[layer].norm()),
                           "align_projection_on_refusal": float(a[layer] @ (ref_dirs[layer] / ref_dirs[layer].norm()))}
    # Cross-fitted projections at the chosen layer: direction from the other half of the cells.
    rng = np.random.default_rng(SEED)
    halves = rng.permutation(len(cells_s)) % 2
    u_full = a[layer] / a[layer].norm()
    proj_full = (acts[:, layer] @ u_full).numpy()
    proj_cv = np.full(len(df), np.nan)
    cell_half = dict(zip(cells_s, halves))
    for h in (0, 1):
        a_h = ds[torch.tensor(halves != h)][:, layer].mean(0) - db[:, layer].mean(0)
        u_h = a_h / a_h.norm()
        rows = df.index[df.pair_id.map(lambda p: cell_half.get(p, -1) == h)]
        proj_cv[rows] = (acts[rows, layer] @ u_h).numpy()
    res["chosen_layer"]["cos_half_directions"] = cosv(
        ds[torch.tensor(halves == 0)][:, layer].mean(0) - db[:, layer].mean(0),
        ds[torch.tensor(halves == 1)][:, layer].mean(0) - db[:, layer].mean(0))
    pj = df[["id", "pair_id", "kind", "election", "claim_side", "template", "severity", "identity", "alignment"]].copy()
    pj["proj_align"] = proj_full
    pj["proj_align_cv"] = proj_cv
    pj["proj_refusal"] = (acts[:, layer] @ (ref_dirs[layer] / ref_dirs[layer].norm())).numpy()
    pj.to_csv(out / "alignment_projection.csv", index=False)
    json.dump(res, open(out / "alignment_dir.json", "w"), indent=2)
    print(json.dumps(res["chosen_layer"], indent=1), flush=True)
    print("cos(align, refusal) by layer:", np.round(res["cos_align_vs_refusal_by_layer"], 2).tolist(), flush=True)
    return a[layer]


@contextlib.contextmanager
def shift(m, vec, layer, signs):
    """Add signs[i] * vec to row i at the input of `layer`, every position. signs is a 1-element list."""
    v = vec.to(m.device, m.model.dtype)

    def pre(_, args, kwargs):
        add = signs[0].to(m.device, m.model.dtype)[:, None, None] * v
        if args:
            return (args[0] + add,) + args[1:], kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + add
        return args, kwargs

    h = m.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def causal(out, df, a_vec, layer, model_name, max_new_tokens, limit, bs):
    m = Model(model_name)
    sub = df[df.alignment.isin(["aligned", "crossed"])].copy()
    if limit:
        sub = sub.groupby(["kind", "alignment"]).head(limit)
    sub = sub.reset_index(drop=True)
    instr = sub.instruction.tolist()
    if (out / "behavior.jsonl").exists():
        base = pd.read_json(out / "behavior.jsonl", lines=True).set_index("id")
        sub["reply_base"] = [base.reply_base.get(i, None) for i in sub.id]
    else:
        print("no behavior.jsonl; generating base replies too", flush=True)
        sub["reply_base"] = m.generate(instr, max_new_tokens=max_new_tokens, bs=bs)
    g = torch.Generator().manual_seed(SEED)
    rand = torch.randn(a_vec.shape, generator=g)
    rand = rand - (rand @ a_vec) * a_vec / (a_vec.norm() ** 2)
    rand = rand / rand.norm() * a_vec.norm()
    print(f"causal: {len(sub)} prompts, |a| = {a_vec.norm():.3f}", flush=True)
    with m.ablate(a_vec):
        sub["reply_abl_align"] = m.generate(instr, max_new_tokens=max_new_tokens, bs=bs)
    with m.ablate(rand):
        sub["reply_abl_random"] = m.generate(instr, max_new_tokens=max_new_tokens, bs=bs)
    signs_all = torch.tensor([1.0 if x == "aligned" else -1.0 for x in sub.alignment])
    cur = [None]
    outs = []
    with shift(m, a_vec, layer, cur):
        for i in range(0, len(instr), bs):
            cur[0] = signs_all[i:i + bs]
            outs += m.generate(instr[i:i + bs], max_new_tokens=max_new_tokens, bs=bs)
    sub["reply_shift"] = outs
    sub.to_json(out / "alignment_dir.jsonl", orient="records", lines=True)
    print(f"wrote {out / 'alignment_dir.jsonl'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="causal: prompts per (kind, alignment), 0 = all")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    out = ROOT / "results" / args.model.split("/")[-1]
    torch.manual_seed(SEED)
    d = torch.load(out / "direction.pt")
    pa = torch.load(out / "political_acts.pt")
    acts = pa["acts"].float()                                        # [N, L, D]
    df = load_political().reset_index(drop=True)
    assert df.id.tolist() == pa["ids"], "political_acts.pt ids do not match data/political_prompts.jsonl; rerun --stage geometry"
    df["row"] = np.arange(len(df))
    a_vec = geometry(out, df, acts, d, args.n_perm)
    if not args.no_generate:
        causal(out, df, a_vec, int(d["layer"]), args.model, args.max_new_tokens, args.limit, args.bs)


if __name__ == "__main__":
    main()
