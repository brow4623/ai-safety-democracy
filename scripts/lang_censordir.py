"""Plan step B5: is there a "Chinese censorship" direction, and does it overlap the refusal direction?

On the China set, for each matched pair (same topic and style):
    c = (zh_sensitive - en_sensitive) - (zh_control - en_control)
i.e. what asking in Chinese adds on sensitive prompts beyond what it adds on harmless ones.
The mean of c over pairs is the "censorship direction". We report its cosine with the
English and Chinese refusal directions at every layer, and a permutation null that
randomly swaps sensitive/control within each pair (2,000 draws).

    uv run python scripts/lang_censordir.py --model Qwen/Qwen2.5-7B-Instruct

Writes results/<model>/lang_censordir.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from model import Model  # noqa: E402
from run import SEED, anchor_sets  # noqa: E402
from lang_experiment import anchor_zh, layer_means, load_prompts, zh_of  # noqa: E402


def cosv(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    m = Model(args.model)
    out = ROOT / "results" / args.model.split("/")[-1]
    ref = json.load(open(out / "direction.json"))
    layer, pos = int(ref["layer"]), int(ref["pos"]) + m.eoi_len

    # Refusal directions at every layer, same procedure as lang_experiment.py stage_direction.
    hf_tr, hl_tr, *_ = anchor_sets(128, 32, 32)
    tab = anchor_zh()
    keep_h = [x for x, s in zip(hf_tr, m.refusal_score(hf_tr, lang="en")) if s > 0]
    keep_l = [x for x, s in zip(hl_tr, m.refusal_score(hl_tr, lang="en")) if s < 0]
    dirs_en = layer_means(m, keep_h, pos) - layer_means(m, keep_l, pos)
    hz, lz = zh_of(hf_tr, tab), zh_of(hl_tr, tab)
    keep_hz = [x for x, s in zip(hz, m.refusal_score(hz, lang="zh")) if s > 0]
    keep_lz = [x for x, s in zip(lz, m.refusal_score(lz, lang="zh")) if s < 0]
    dirs_zh = layer_means(m, keep_hz, pos) - layer_means(m, keep_lz, pos)

    # China-set activations at every layer for the four cells of each pair.
    df = load_prompts()
    df = df[(df["set"] == "china") & df.kind.isin(["sensitive", "control"]) & df.lang.isin(["en", "zh_s"])]
    acts = m.eoi_acts(df.prompt.tolist(), bs=16)[:, :, pos]          # [N, L, D]
    idx = {(r.pair, r.kind, r.lang): i for i, r in enumerate(df.itertuples())}
    pairs = sorted({p for p, k, l in idx if all((p, kk, ll) in idx for kk in ["sensitive", "control"] for ll in ["en", "zh_s"])})

    def a(p, k, l):
        return acts[idx[(p, k, l)]]
    lang_sens = torch.stack([a(p, "sensitive", "zh_s") - a(p, "sensitive", "en") for p in pairs])  # [P, L, D]
    lang_ctrl = torch.stack([a(p, "control", "zh_s") - a(p, "control", "en") for p in pairs])
    c_pairs = lang_sens - lang_ctrl
    c = c_pairs.mean(0)                                                # [L, D]

    L = m.n_layers
    res = {"model": args.model, "layer": layer, "pos": int(ref["pos"]), "n_pairs": len(pairs),
           "cos_censor_vs_en_refusal_by_layer": [cosv(c[l], dirs_en[l]) for l in range(1, L)],
           "cos_censor_vs_zh_refusal_by_layer": [cosv(c[l], dirs_zh[l]) for l in range(1, L)],
           "cos_language_only_vs_en_refusal_by_layer": [cosv(lang_ctrl.mean(0)[l], dirs_en[l]) for l in range(1, L)],
           "layers": list(range(1, L))}
    u = dirs_en[layer] / dirs_en[layer].norm()
    res["chosen_layer"] = {
        "cos_censor_vs_en_refusal": cosv(c[layer], dirs_en[layer]),
        "cos_censor_vs_zh_refusal": cosv(c[layer], dirs_zh[layer]),
        "censor_norm": float(c[layer].norm()),
        "censor_projection_on_en_refusal": float(c[layer] @ u),
        "refusal_norm": float(dirs_en[layer].norm()),
        "refusal_norm_divya": float(ref["norm"]),  # sanity check: same direction as Divya's
    }

    # Permutation null: swap sensitive/control within each pair (flips the sign of c_pairs[p]).
    g = torch.Generator().manual_seed(SEED)
    null = []
    cp = c_pairs[:, layer]
    for _ in range(args.n_perm):
        s = (torch.randint(0, 2, (len(pairs),), generator=g) * 2 - 1).float()
        null.append(cosv((cp * s[:, None]).mean(0), dirs_en[layer]))
    null = np.array(null)
    obs = res["chosen_layer"]["cos_censor_vs_en_refusal"]
    res["chosen_layer"]["perm_null_mean"] = float(null.mean())
    res["chosen_layer"]["perm_null_95"] = [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]
    res["chosen_layer"]["perm_p_two_sided"] = float((np.abs(null) >= abs(obs)).mean())
    json.dump(res, open(out / "lang_censordir.json", "w"), indent=2)
    print(json.dumps(res["chosen_layer"], indent=1), flush=True)


if __name__ == "__main__":
    main()
