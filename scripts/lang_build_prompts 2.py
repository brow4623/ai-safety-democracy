"""Build the combined prompt set for the language experiment.

  china  : src/lang_prompts.py (12 topics x question/writing x sensitive/control
           x en/zh_s/zh_t) + the 6 sanity anchors            -> 150 rows
  us     : the team's stolen-election prompts, identity "none", 6 stolen
           templates (2 per severity) + 2 benign, in English and the Simplified
           Chinese translations from data/lang_us_zh.json     -> 128 rows

Every row carries: id, set, pair, topic, style, kind, lang, prompt.
`pair` matches a sensitive prompt to its control and to its other languages.

    uv run python scripts/lang_build_prompts.py   ->  data/lang_experiment_prompts.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lang_prompts import build as build_china  # noqa: E402


def main():
    rows = []
    for r in build_china():
        rows.append({"id": f"cn-{r['id']}", "set": "china", "pair": r["pair"], "topic": r["topic"],
                     "style": r["style"], "kind": r["kind"], "lang": r["lang"], "prompt": r["prompt"]})

    us = json.load(open(ROOT / "data" / "lang_us_zh.json"))
    for r in us:
        pair = r["pair_id"]                       # e.g. stolen-2020_pres-t0
        kind = "sensitive" if r["kind"] == "stolen" else "control"
        meta = {"set": "us", "pair": pair, "topic": r["election"], "style": f"t{r['template']}",
                "kind": kind, "severity": r["severity"], "claim_side": r["claim_side"]}
        rows.append({"id": f"us-{r['id']}-en", "lang": "en", "prompt": r["instruction"], **meta})
        if r["zh"]:
            rows.append({"id": f"us-{r['id']}-zh_s", "lang": "zh_s", "prompt": r["zh"], **meta})

    out = ROOT / "data" / "lang_experiment_prompts.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = {}
    for r in rows:
        n[(r["set"], r["kind"], r["lang"])] = n.get((r["set"], r["kind"], r["lang"]), 0) + 1
    print(f"wrote {len(rows)} prompts to {out}")
    for k in sorted(n, key=str):
        print(f"  {k}: {n[k]}")


if __name__ == "__main__":
    main()
