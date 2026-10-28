"""Language & refusal experiment, GPU side. One model per invocation.

Phases (each writes its own file under results/<model>/, so they can rerun alone):

  direction  Verify Divya's English refusal direction reproduces at her (layer, pos),
             build a Chinese one at the SAME (layer, pos) from translated anchors,
             compare them layer by layer, and run the cross-language transfer test.
             A Chinese-only scan over layers is recorded as secondary evidence.
  behavior   Generate replies to every language prompt: base, with the English
             direction ablated, and (Qwen) with the Chinese direction ablated.
  patch      Projection matching, corrected for the language gap measured on the
             matched controls, in both directions (zh->en and en->zh), plus a
             random-direction control.

    uv run python scripts/lang_experiment.py --model Qwen/Qwen2.5-7B-Instruct --stage all
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
from model import DEFAULT_MODEL, Model, classify_any  # noqa: E402
from run import SEED, anchor_sets, cos, last_logprobs  # noqa: E402
from patch import shift  # noqa: E402

TOL_ABLATE = 1.0      # allowed drift vs Divya's ablate_harmful
TOL_NORM = 0.05       # fractional drift allowed on the direction norm


def load_prompts():
    return pd.read_json(ROOT / "data" / "lang_experiment_prompts.jsonl", lines=True)


def anchor_zh():
    """Chinese anchors keyed by their English source, so the splits match run.py."""
    return {r["en"]: r for r in json.load(open(ROOT / "data" / "lang_anchor_zh.json"))}


def zh_of(texts, table):
    return [table[t]["zh"] for t in texts if table.get(t, {}).get("zh")]


def cjk_share(s):
    chars = [c for c in s if not c.isspace()]
    return sum("一" <= c <= "鿿" for c in chars) / max(len(chars), 1)


def layer_means(m, texts, pos, bs=16):
    """Mean residual-stream activation at `pos`, for every layer: [n_layers, d_model]."""
    return m.eoi_acts(texts, bs=bs)[:, :, pos].mean(0)


# --------------------------------------------------------------- direction

def stage_direction(m, out, args):
    ref = json.load(open(out / "direction.json"))
    layer, pos_rel = int(ref["layer"]), int(ref["pos"])
    pos = pos_rel + m.eoi_len
    res = {"model": m.name, "layer": layer, "pos": pos_rel, "eoi_len": m.eoi_len,
           "reference": {k: ref[k] for k in ["ablate_harmful", "add_harmless", "kl_harmless", "norm"]}}

    hf_tr, hl_tr, hf_val, hl_val, hf_test = anchor_sets(args.n_train, args.n_val, args.n_test)
    tab = anchor_zh()

    # --- English direction at Divya's (layer, pos), her filters and splits.
    keep_h = [x for x, s in zip(hf_tr, m.refusal_score(hf_tr, lang="en")) if s > 0]
    keep_l = [x for x, s in zip(hl_tr, m.refusal_score(hl_tr, lang="en")) if s < 0]
    all_h, all_l = layer_means(m, keep_h, pos), layer_means(m, keep_l, pos)   # [L, D]
    dirs_en = all_h - all_l
    d_en = dirs_en[layer]
    base_logp = torch.cat(list(last_logprobs(m, hl_val)))
    with m.ablate(d_en):
        abl = m.refusal_score(hf_val, lang="en").mean().item()
        logp = torch.cat(list(last_logprobs(m, hl_val)))
    with m.add(d_en, layer):
        add = m.refusal_score(hl_val, lang="en").mean().item()
    kl = (base_logp.exp() * (base_logp - logp)).sum(-1).mean().item()
    res["english"] = {"n_harmful_train": len(keep_h), "n_harmless_train": len(keep_l),
                      "ablate_harmful": abl, "add_harmless": add, "kl_harmless": kl,
                      "norm": d_en.norm().item()}
    res["verified"] = bool(abs(abl - ref["ablate_harmful"]) <= TOL_ABLATE and add > 0
                           and abs(d_en.norm().item() - ref["norm"]) / ref["norm"] <= TOL_NORM)
    print(f"english direction verified={res['verified']} {res['english']}", flush=True)

    # --- Chinese direction at the SAME (layer, pos), same procedure, Chinese filters.
    hf_tr_zh, hl_tr_zh = zh_of(hf_tr, tab), zh_of(hl_tr, tab)
    hf_val_zh, hl_val_zh = zh_of(hf_val, tab), zh_of(hl_val, tab)
    hf_test_zh = zh_of(hf_test, tab)
    keep_hz = [x for x, s in zip(hf_tr_zh, m.refusal_score(hf_tr_zh, lang="zh")) if s > 0]
    keep_lz = [x for x, s in zip(hl_tr_zh, m.refusal_score(hl_tr_zh, lang="zh")) if s < 0]
    all_hz, all_lz = layer_means(m, keep_hz, pos), layer_means(m, keep_lz, pos)
    dirs_zh = all_hz - all_lz
    d_zh = dirs_zh[layer]
    with m.ablate(d_zh):
        abl_z = m.refusal_score(hf_val_zh, lang="zh").mean().item()
    with m.add(d_zh, layer):
        add_z = m.refusal_score(hl_val_zh, lang="zh").mean().item()
    res["chinese"] = {"n_harmful_train": len(keep_hz), "n_harmless_train": len(keep_lz),
                      "ablate_harmful_zh": abl_z, "add_harmless_zh": add_z, "norm": d_zh.norm().item(),
                      "baseline_harmful_zh": m.refusal_score(hf_val_zh, lang="zh").mean().item()}
    res["cos_en_zh_chosen_layer"] = cos(d_en, d_zh)
    print(f"chinese direction {res['chinese']}, cos={res['cos_en_zh_chosen_layer']:.3f}", flush=True)

    # --- Cosine at every layer (same position), with a random-direction baseline.
    g = torch.Generator().manual_seed(SEED)
    res["cos_by_layer"] = [cos(dirs_en[l], dirs_zh[l]) for l in range(m.n_layers)]
    rnd = torch.randn(dirs_en.shape[-1], generator=g)
    res["cos_random_baseline"] = [cos(dirs_en[l], rnd) for l in range(m.n_layers)]

    # --- Secondary: where a Chinese-only scan would put the direction (same position).
    scan = []
    for l in range(1, int(0.8 * m.n_layers)):
        with m.ablate(dirs_zh[l]):
            scan.append({"layer": l, "ablate_harmful_zh": m.refusal_score(hf_val_zh, lang="zh").mean().item()})
    res["chinese_scan_secondary"] = {"best_layer": int(min(scan, key=lambda r: r["ablate_harmful_zh"])["layer"]),
                                     "scan": scan}

    # --- Transfer test: does each direction remove the other language's refusals?
    tr = {}
    for name, d in [("english", d_en), ("chinese", d_zh)]:
        for lg, test in [("en", hf_test), ("zh", hf_test_zh)]:
            base = [classify_any(x) for x in m.generate(test, max_new_tokens=48)]
            with m.ablate(d):
                ab = [classify_any(x) for x in m.generate(test, max_new_tokens=48)]
            tr[f"{name}_dir_on_{lg}_prompts"] = {
                "n": len(test),
                "refuse_base": float(np.mean([x == "refuse" for x in base])),
                "refuse_ablated": float(np.mean([x == "refuse" for x in ab]))}
    res["transfer"] = tr
    print("transfer:", json.dumps(tr, indent=1), flush=True)

    # --- Fluency control: KL on Chinese control prompts under the real vs 5 random ablations.
    df = load_prompts()
    zh_ctrl = df[(df.kind == "control") & (df.lang == "zh_s")].prompt.tolist()
    base_zh = torch.cat(list(last_logprobs(m, zh_ctrl)))
    def kl_under(d):
        with m.ablate(d):
            lp = torch.cat(list(last_logprobs(m, zh_ctrl)))
        return (base_zh.exp() * (base_zh - lp)).sum(-1).mean().item()
    unit = d_en / d_en.norm()
    rands = []
    for _ in range(5):
        r = torch.randn(unit.shape, generator=g)
        r = r - (r @ unit) * unit
        rands.append(r / r.norm() * d_en.norm())
    res["kl_zh_controls"] = {"english_direction": kl_under(d_en),
                             "chinese_direction": kl_under(d_zh),
                             "random_directions": [kl_under(r) for r in rands]}

    torch.save({"d_en": d_en, "d_zh": d_zh, "layer": layer, "pos": pos,
                "dirs_en": dirs_en, "dirs_zh": dirs_zh, "randoms": torch.stack(rands)},
               out / "lang_directions.pt")
    json.dump(res, open(out / "lang_direction.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote {out / 'lang_direction.json'}", flush=True)


# ---------------------------------------------------------------- behavior

def gen_rows(m, df, max_new_tokens, llama_zh_tokens=None):
    """Generate for a frame, grouping by language so Llama gets a longer cap in Chinese."""
    replies = pd.Series(index=df.index, dtype=object)
    for lg, sub in df.groupby("lang"):
        cap = llama_zh_tokens if (llama_zh_tokens and lg.startswith("zh")) else max_new_tokens
        replies.loc[sub.index] = m.generate(sub.prompt.tolist(), max_new_tokens=cap, bs=32)
    return replies


def stage_behavior(m, out, args):
    d = torch.load(out / "lang_directions.pt")
    df = load_prompts()
    llama_zh = 400 if "Llama" in m.name else None
    conds = {"base": None, "abl_en": d["d_en"]}
    if args.ablate_zh:
        conds["abl_zh"] = d["d_zh"]
    res = df.copy()
    for name, direction in conds.items():
        print(f"generating {name} ({len(df)} prompts)", flush=True)
        if direction is None:
            res[f"reply_{name}"] = gen_rows(m, df, args.max_new_tokens, llama_zh)
        else:
            with m.ablate(direction):
                res[f"reply_{name}"] = gen_rows(m, df, args.max_new_tokens, llama_zh)
        res[f"cjk_{name}"] = res[f"reply_{name}"].map(cjk_share)
        res[f"ntok_{name}"] = [len(m.tok.encode(x, add_special_tokens=False)) for x in res[f"reply_{name}"]]
        cap = res.lang.map(lambda lg: llama_zh if (llama_zh and lg.startswith("zh")) else args.max_new_tokens)
        res[f"truncated_{name}"] = res[f"ntok_{name}"] >= cap - 2
    res.to_json(out / "lang_behavior.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"wrote {out / 'lang_behavior.jsonl'}", flush=True)


# ------------------------------------------------------------------- patch

def stage_patch(m, out, args):
    """Projection matching corrected for the language gap seen on the controls.

    The censorship-specific elevation is
        (zh_sensitive - en_sensitive) - (zh_control - en_control).
    zh->en removes exactly that from each Chinese sensitive prompt; en->zh adds it
    to each English one. A random orthogonal direction of the same size is the control.
    """
    d = torch.load(out / "lang_directions.pt")
    layer, pos = d["layer"], d["pos"]
    unit = (d["d_en"] / d["d_en"].norm()).float()
    rand = d["randoms"][0].float()
    rand = rand / rand.norm()
    df = load_prompts()
    df = df[df.lang.isin(["en", "zh_s"]) & df.kind.isin(["sensitive", "control"])].reset_index(drop=True)

    acts = m.eoi_acts(df.prompt.tolist(), bs=16)[:, layer, pos]       # [N, D]
    df["proj"] = (acts.float() @ unit).tolist()
    stats, out_rows = {}, []
    for st, sub in df.groupby("set"):
        mean = {(k, lg): sub[(sub.kind == k) & (sub.lang == lg)].proj.mean()
                for k in ["sensitive", "control"] for lg in ["en", "zh_s"]}
        lang_gap = mean[("control", "zh_s")] - mean[("control", "en")]
        cens = (mean[("sensitive", "zh_s")] - mean[("sensitive", "en")]) - lang_gap
        stats[st] = {"mean_proj": {f"{k}_{lg}": float(v) for (k, lg), v in mean.items()},
                     "language_gap_controls": float(lang_gap), "censorship_specific": float(cens)}
        print(f"{st}: language gap {lang_gap:+.3f}, censorship-specific {cens:+.3f}", flush=True)
        for lg, direction, delta in [("zh_s", "zh_to_en", -cens), ("en", "en_to_zh", +cens)]:
            src = sub[(sub.kind == "sensitive") & (sub.lang == lg)]
            if args.limit:
                src = src.head(args.limit)
            r = src.copy()
            r["comparison"], r["delta"] = direction, delta
            deltas = [delta] * len(r)
            cap = 400 if ("Llama" in m.name and lg.startswith("zh")) else args.max_new_tokens
            r["reply_patched"] = generate_shifted(m, r.prompt.tolist(), deltas, unit, layer, cap)
            r["reply_random"] = generate_shifted(m, r.prompt.tolist(), deltas, rand, layer, cap)
            out_rows.append(r)
    pd.concat(out_rows).to_json(out / "lang_patch.jsonl", orient="records", lines=True, force_ascii=False)
    json.dump(stats, open(out / "lang_patch_stats.json", "w"), indent=2)
    print(f"wrote {out / 'lang_patch.jsonl'}", flush=True)


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
    ap.add_argument("--stage", choices=["direction", "behavior", "patch", "all"], default="all")
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--ablate-zh", action="store_true", help="behavior: also ablate the Chinese direction")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out = ROOT / "results" / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    m = Model(args.model)
    print(f"{m.name} on {m.device}, {m.n_layers} layers, eoi_len {m.eoi_len}", flush=True)
    for stage in ["direction", "behavior", "patch"]:
        if args.stage in (stage, "all"):
            globals()[f"stage_{stage}"](m, out, args)


if __name__ == "__main__":
    main()
