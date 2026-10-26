"""Pipeline for the partisan refusal-geometry experiment.

Stages (each caches to results/<model>/ so later stages can rerun alone):
  direction  extract the anchor refusal direction from standard harmful vs
             harmless prompts (Arditi et al. splits) and pick the best
             (layer, position) by its causal effect
  geometry   project the political prompts onto that direction, build
             party-specific directions, test whether they split
  behavior   generate replies with and without the direction ablated,
             label them refuse / pushback / comply

Usage:
  uv run python scripts/run.py --stage all
  uv run python scripts/run.py --stage behavior --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import DEFAULT_MODEL, Model, classify  # noqa: E402

SEED = 0
KL_PAPER = 0.1   # Arditi et al.'s cutoff for KL on harmless prompts

# Anchor prompts that touch politics are dropped, so the refusal direction
# can't pick up "this is about politics" (e.g. an AdvBench election-
# misinformation prompt, or "compare the Democratic and Republican parties").
POLITICAL = re.compile(r"\b(elect\w*|vot(e|es|ed|er|ers|ing)|ballot\w*|democrat\w*|republican\w*|"
                       r"trump|biden|clinton|obama|politic\w*|campaign\w*|partisan|congress\w*|senat\w*)\b",
                       re.IGNORECASE)


def load_json(p):
    return [x["instruction"] for x in json.load(open(p)) if not POLITICAL.search(x["instruction"])]


def load_political():
    return pd.read_json(ROOT / "data" / "political_prompts.jsonl", lines=True)


def anchor_sets(n_train, n_val, n_test=0):
    """Harmful train/test come from disjoint parts of harmful_train.json, so the
    final ablation sanity check uses prompts that played no part in choosing
    the direction (harmful_val picks the layer, so it can't be the test)."""
    rng = random.Random(SEED)
    d = ROOT / "data" / "anchor"
    harmful_pool = load_json(d / "harmful_train.json")
    rng.shuffle(harmful_pool)
    harmful_train, harmful_test = harmful_pool[:n_train], harmful_pool[n_train:n_train + n_test]
    harmless_train = rng.sample(load_json(d / "harmless_train.json"), n_train)
    harmful_val = rng.sample(load_json(d / "harmful_val.json"), n_val)
    harmless_val = rng.sample(load_json(d / "harmless_val.json"), n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val, harmful_test


# ---------------------------------------------------------------- direction

def stage_direction(m, out, args):
    if args.reselect and (out / "direction_selection.csv").exists():
        # Reuse the scan from a previous run, only re-apply the filters.
        means = torch.load(out / "anchor_means.pt")
        return select_direction(m, out, args, pd.read_csv(out / "direction_selection.csv"),
                                means["mean_harmful"] - means["mean_harmless"], {})

    harmful_train, harmless_train, harmful_val, harmless_val, _ = anchor_sets(args.n_train, args.n_val)

    # Keep only harmful prompts the model actually refuses and harmless ones it
    # does not, as in Arditi et al., so the means describe refusal itself.
    hs = m.refusal_score(harmful_train)
    ls = m.refusal_score(harmless_train)
    harmful_train = [x for x, s in zip(harmful_train, hs) if s > 0]
    harmless_train = [x for x, s in zip(harmless_train, ls) if s < 0]
    print(f"kept {len(harmful_train)} harmful / {len(harmless_train)} harmless train prompts")

    acts_h = m.eoi_acts(harmful_train)          # [N, L, P, D]
    acts_l = m.eoi_acts(harmless_train)
    mu_h, mu_l = acts_h.mean(0), acts_l.mean(0)  # [L, P, D]
    dirs = mu_h - mu_l
    torch.save({"mean_harmful": mu_h, "mean_harmless": mu_l}, out / "anchor_means.pt")

    base_h = m.refusal_score(harmful_val).mean().item()
    base_l = m.refusal_score(harmless_val).mean().item()
    base_logp = torch.cat([lp for lp in last_logprobs(m, harmless_val)])
    print(f"baseline refusal score: harmful {base_h:.2f}, harmless {base_l:.2f}")

    rows = []
    max_layer = int(0.8 * m.n_layers)
    for layer in range(1, max_layer):
        for pos in range(m.eoi_len):
            r = dirs[layer, pos]
            with m.ablate(r):
                abl = m.refusal_score(harmful_val).mean().item()
                logp = torch.cat([lp for lp in last_logprobs(m, harmless_val)])
            kl = (base_logp.exp() * (base_logp - logp)).sum(-1).mean().item()
            with m.add(r, layer):
                add = m.refusal_score(harmless_val).mean().item()
            rows.append({"layer": layer, "pos": pos - m.eoi_len, "ablate_harmful": abl,
                         "add_harmless": add, "kl_harmless": kl, "norm": r.norm().item()})
        print(f"layer {layer}: best ablate so far {min(x['ablate_harmful'] for x in rows):.2f}")
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "direction_selection.csv", index=False)
    layer, p = select_direction(m, out, args, sel, dirs,
                                {"baseline_harmful": base_h, "baseline_harmless": base_l,
                                 "n_harmful_train": len(harmful_train),
                                 "n_harmless_train": len(harmless_train)})
    direction_stability(out, acts_h[:, layer, p], acts_l[:, layer, p], dirs[layer, p])


def select_direction(m, out, args, sel, dirs, extra):
    # Arditi et al. filters: addition must induce refusal, ablation must not
    # change harmless behavior much. Then take the lowest ablated refusal score.
    # Their KL cutoff is 0.1, which we try first. On Qwen2.5-3B nothing with a
    # real ablation effect passed it, and the unfiltered winner (KL 3.1)
    # visibly damaged factual recall, so we then fall back to --kl-max.
    for cutoff in [KL_PAPER, args.kl_max]:
        ok = sel[(sel.add_harmless > 0) & (sel.kl_harmless < cutoff)]
        if not ok.empty:
            break
    else:
        print("WARNING: no candidate passed the filters, using unfiltered best")
        ok, cutoff = sel, None
    best = ok.sort_values("ablate_harmful").iloc[0]
    layer, pos = int(best.layer), int(best.pos)
    p = pos + m.eoi_len
    torch.save({"direction": dirs[layer, p], "layer": layer, "pos": p,
                "all_dirs": dirs}, out / "direction.pt")
    summary = {"model": m.name, "layer": layer, "pos": pos, "n_layers": m.n_layers,
               "kl_cutoff_used": cutoff, **extra, **best.to_dict()}
    json.dump(summary, open(out / "direction.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return layer, p


def direction_stability(out, acts_h, acts_l, direction, n_boot=100):
    """Is the chosen direction a fluke of which prompts were sampled?

    Re-extracts it from bootstrap resamples of the train prompts, and from two
    disjoint halves, and reports the cosine with the full-sample direction.
    """
    g = torch.Generator().manual_seed(SEED)
    boots = []
    for _ in range(n_boot):
        ih = torch.randint(len(acts_h), (len(acts_h),), generator=g)
        il = torch.randint(len(acts_l), (len(acts_l),), generator=g)
        boots.append(cos(acts_h[ih].mean(0) - acts_l[il].mean(0), direction))
    ph, pl = torch.randperm(len(acts_h), generator=g), torch.randperm(len(acts_l), generator=g)
    hh, hl = len(ph) // 2, len(pl) // 2
    half_a = acts_h[ph[:hh]].mean(0) - acts_l[pl[:hl]].mean(0)
    half_b = acts_h[ph[hh:]].mean(0) - acts_l[pl[hl:]].mean(0)
    stab = {"bootstrap_cos_mean": float(np.mean(boots)), "bootstrap_cos_p05": float(np.quantile(boots, 0.05)),
            "bootstrap_cos_min": float(np.min(boots)), "split_half_cos": cos(half_a, half_b), "n_boot": n_boot}
    json.dump(stab, open(out / "direction_stability.json", "w"), indent=2)
    print("direction stability:", stab)


@torch.no_grad()
def last_logprobs(m, instructions, bs=16):
    for _, enc in m._batches(instructions, bs):
        yield m.model(**enc).logits[:, -1, :].float().log_softmax(-1).cpu()


# ----------------------------------------------------------------- geometry

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


def stage_geometry(m, out, args):
    d = torch.load(out / "direction.pt")
    means = torch.load(out / "anchor_means.pt")
    layer, pos = d["layer"], d["pos"]
    r_hat = d["direction"] / d["direction"].norm()
    df = load_political()

    acts = m.eoi_acts(df.instruction.tolist())[:, :, pos, :]   # [N, L, D] at chosen position
    df["proj"] = (acts[:, layer] @ r_hat).numpy()
    df["refusal_score"] = m.refusal_score(df.instruction.tolist()).numpy()
    df.drop(columns="instruction").to_csv(out / "geometry_per_prompt.csv", index=False)

    # Saved for the layer-wise analysis, so it needs no GPU: every prompt's
    # activations at every layer, plus its projection onto each layer's own
    # anchor direction (same position as the chosen direction).
    torch.save({"ids": df.id.tolist(), "pos": pos, "acts": acts.half()}, out / "political_acts.pt")
    dirs_l = d["all_dirs"][:, pos]
    dirs_l = dirs_l / dirs_l.norm(dim=-1, keepdim=True)
    proj_l = torch.einsum("nld,ld->nl", acts, dirs_l).numpy()
    meta = df[["id", "pair_id", "kind", "election", "claim_side", "template", "severity", "identity", "alignment"]]
    pd.concat([meta, pd.DataFrame(proj_l, columns=[f"L{l}" for l in range(m.n_layers)])], axis=1) \
        .to_csv(out / "layerwise_projection.csv", index=False)

    # 1. Where do the prompts sit on the anchor refusal direction?
    g = df.groupby(["kind", "severity", "identity", "claim_side"])[["proj", "refusal_score"]].mean()
    print(g)
    g.to_csv(out / "projection_by_cell.csv")

    # 2. Paired identity effect: within each (election, template), D minus R.
    #    Sign-flip permutation gives the null for "identity does nothing".
    rng = np.random.default_rng(SEED)
    pair_rows = []
    for kind in ["stolen", "benign"]:
        piv = df[df.kind == kind].pivot(index="pair_id", columns="identity", values="proj")
        cells = df[(df.kind == kind)].drop_duplicates("pair_id").set_index("pair_id")
        side, sev = cells.claim_side, cells.severity
        subsets = [("all", slice(None)), ("R-coded claim", side == "R"), ("D-coded claim", side == "D")]
        subsets += [(f"severity {s}", sev == s) for s in sorted(sev.unique()) if sev.nunique() > 1]
        for subset, mask in subsets:
            diff = (piv.democrat - piv.republican)[mask].values
            null = np.array([(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(10000)])
            pair_rows.append({"kind": kind, "claims": subset, "n_pairs": len(diff),
                              "mean_D_minus_R": diff.mean(),
                              "p_perm": (np.abs(null) >= abs(diff.mean())).mean(),
                              # voter = same sentence shape, no party: the fair baseline
                              "dem_minus_voter": (piv.democrat - piv.voter)[mask].mean(),
                              "rep_minus_voter": (piv.republican - piv.voter)[mask].mean(),
                              "voter_minus_none": (piv.voter - piv["none"])[mask].mean(),
                              # non-political identity of the same shape
                              **({"fan_minus_none": (piv.fan - piv["none"])[mask].mean()}
                                 if "fan" in piv else {})})
    pairs = pd.DataFrame(pair_rows)
    print(pairs.to_string())
    pairs.to_csv(out / "paired_identity_effect.csv", index=False)

    # 3. Do party-specific refusal directions split?
    #    r_party(l) = mean(stolen prompts from that party) - mean(harmless anchor).
    #    cos(r_D, r_R) alone is near 1 by construction (the prompts share all
    #    their content), so the real test is the gap between them,
    #    delta = r_D - r_R. We ask two things about delta:
    #      - how much of it lies along the anchor refusal direction
    #      - whether it matches the D-minus-R gap on *benign* prompts. If it
    #        does, the gap is just "the model noticed a party label", not a
    #        party-specific way of refusing.
    #    A sign-flip null (random D/R swaps within pairs) says how big these
    #    cosines get by chance.
    mu_harmless = means["mean_harmless"][:, pos]                # [L, D]
    st, bn = df.kind == "stolen", df.kind == "benign"

    def pdir(mask, l):
        return acts[mask.values, l].mean(0) - mu_harmless[l]

    def pair_diffs(kind, l):
        sub = df[df.kind == kind]
        dem = sub[sub.identity == "democrat"].sort_values("pair_id").index
        rep = sub[sub.identity == "republican"].sort_values("pair_id").index
        return acts[dem, l] - acts[rep, l]                       # [n_pairs, D]

    geo = []
    for l in range(1, m.n_layers):
        rD, rR, rV = (pdir(st & (df.identity == p), l) for p in ["democrat", "republican", "voter"])
        anchor_l = d["all_dirs"][l, pos]
        ds, db = pair_diffs("stolen", l), pair_diffs("benign", l)
        delta_s, delta_b = ds.mean(0), db.mean(0)
        null = []
        for _ in range(200):
            flip = torch.from_numpy(rng.choice([-1.0, 1.0], len(ds))).float().unsqueeze(1)
            null.append(abs(cos((ds * flip).mean(0), anchor_l)))
        geo.append({
            "layer": l,
            "cos_D_R": cos(rD, rR),
            "cos_D_anchor": cos(rD, anchor_l), "cos_R_anchor": cos(rR, anchor_l),
            "cos_voter_anchor": cos(rV, anchor_l),
            "cos_delta_anchor": cos(delta_s, anchor_l),
            "cos_delta_anchor_null95": float(np.quantile(null, 0.95)),
            "cos_delta_stolen_benign": cos(delta_s, delta_b),
            "delta_norm_over_rD_norm": (delta_s.norm() / rD.norm()).item(),
        })
    geo = pd.DataFrame(geo)
    geo.to_csv(out / "party_directions_by_layer.csv", index=False)
    print(geo.round(3).to_string())


# ----------------------------------------------------------------- behavior

def stage_behavior(m, out, args):
    d = torch.load(out / "direction.pt")
    df = load_political()
    if args.limit:
        df = df.groupby(["kind", "identity"]).head(args.limit)
    instr = df.instruction.tolist()

    res = df.copy()
    res["reply_base"] = m.generate(instr, max_new_tokens=args.max_new_tokens)
    with m.ablate(d["direction"]):
        res["reply_ablated"] = m.generate(instr, max_new_tokens=args.max_new_tokens)
    res["label_base"] = res.reply_base.map(classify)
    res["label_ablated"] = res.reply_ablated.map(classify)
    res.to_json(out / "behavior.jsonl", orient="records", lines=True)

    rates = []
    for cond in ["base", "ablated"]:
        t = (res.groupby(["kind", "severity", "identity", "claim_side"])[f"label_{cond}"]
             .value_counts(normalize=True).unstack(fill_value=0))
        t["condition"] = cond
        rates.append(t.reset_index())
    rates = pd.concat(rates)
    rates.to_csv(out / "behavior_rates.csv", index=False)
    print(rates.round(2).to_string())

    # Sanity check that the direction is causal on standard harmful prompts,
    # on held-out prompts that were not used to build or choose it.
    *_, harmful_test = anchor_sets(args.n_train, args.n_val, args.n_test)
    base = [classify(x) for x in m.generate(harmful_test, max_new_tokens=48)]
    with m.ablate(d["direction"]):
        abl = [classify(x) for x in m.generate(harmful_test, max_new_tokens=48)]
    check = {"n_harmful_test": len(harmful_test),
             "harmful_test_refuse_base": np.mean([x == "refuse" for x in base]),
             "harmful_test_refuse_ablated": np.mean([x == "refuse" for x in abl])}
    json.dump(check, open(out / "ablation_sanity.json", "w"), indent=2)
    print(check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stage", choices=["direction", "geometry", "behavior", "all"], default="all")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=64, help="held-out harmful prompts for the ablation sanity check")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="behavior: prompts per (kind, identity), 0 = all")
    ap.add_argument("--kl-max", type=float, default=0.5, help="direction: fallback max KL on harmless prompts if nothing passes the paper's 0.1")
    ap.add_argument("--reselect", action="store_true",
                    help="direction: reuse the saved scan, only re-apply the filters")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out = ROOT / "results" / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    m = Model(args.model)
    print(f"{m.name} on {m.device}, {m.n_layers} layers, eoi_len {m.eoi_len}")
    for stage in ["direction", "geometry", "behavior"]:
        if args.stage in (stage, "all"):
            globals()[f"stage_{stage}"](m, out, args)


if __name__ == "__main__":
    main()
