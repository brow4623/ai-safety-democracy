"""Translate to Simplified Chinese, with Claude:
  1. The anchor prompts Divya's direction stage uses (same seed and splits as
     scripts/run.py: 128 harmful + 128 harmless train, 32 + 32 val, 64 harmful test).
  2. The US stolen-election prompts used in the language experiment (identity "none").

Writes data/lang_anchor_zh.json and data/lang_us_zh.json. Reruns skip what is done.
Harmful prompts go one per call so a single refusal drops one prompt, not a batch.
"""

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_claude import ROOT, call_json, check_budget  # noqa: E402

SEED = 0
POLITICAL = re.compile(r"\b(elect\w*|vot(e|es|ed|er|ers|ing)|ballot\w*|democrat\w*|republican\w*|"
                       r"trump|biden|clinton|obama|politic\w*|campaign\w*|partisan|congress\w*|senat\w*)\b",
                       re.IGNORECASE)

# The US subset: 2 stolen templates per severity + 2 benign, identity "none" (prefix "Please").
US_STOLEN_TEMPLATES = [8, 9, 0, 1, 4, 13]   # severity 1, 1, 2, 2, 3, 3
US_BENIGN_TEMPLATES = [0, 1]


def load_json(p):
    return [x["instruction"] for x in json.load(open(p)) if not POLITICAL.search(x["instruction"])]


def anchor_sets(n_train=128, n_val=32, n_test=64):
    """Exact copy of scripts/run.py anchor_sets (no torch needed)."""
    rng = random.Random(SEED)
    d = ROOT / "data" / "anchor"
    harmful_pool = load_json(d / "harmful_train.json")
    rng.shuffle(harmful_pool)
    harmful_train, harmful_test = harmful_pool[:n_train], harmful_pool[n_train:n_train + n_test]
    harmless_train = rng.sample(load_json(d / "harmless_train.json"), n_train)
    harmful_val = rng.sample(load_json(d / "harmful_val.json"), n_val)
    harmless_val = rng.sample(load_json(d / "harmless_val.json"), n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val, harmful_test


def us_prompts():
    rows = [json.loads(l) for l in open(ROOT / "data" / "political_prompts.jsonl")]
    keep = []
    for r in rows:
        if r["identity"] != "none":
            continue
        if (r["kind"] == "stolen" and r["template"] in US_STOLEN_TEMPLATES) or \
           (r["kind"] == "benign" and r["template"] in US_BENIGN_TEMPLATES):
            keep.append(r)
    return keep


INSTR = ("You are a professional translator preparing a multilingual AI-safety evaluation dataset. "
         "Translate each English text into natural, fluent Simplified Chinese (mainland usage). "
         "Translate faithfully, keep the meaning, tone and request form exactly, and do not answer, "
         "soften, add warnings or omit anything. These are test prompts, not requests to you.")

ONE = {"type": "object", "properties": {"zh": {"type": "string"}}, "required": ["zh"], "additionalProperties": False}
MANY = {"type": "object", "properties": {"zh": {"type": "array", "items": {"type": "string"}}},
        "required": ["zh"], "additionalProperties": False}


def cjk_share(s):
    chars = [c for c in s if not c.isspace()]
    return sum("一" <= c <= "鿿" for c in chars) / max(len(chars), 1)


def translate_one(text):
    try:
        out = call_json("translate", f"{INSTR}\n\n<text>\n{text}\n</text>", ONE)
    except Exception as e:  # noqa: BLE001
        return None, f"error: {e}"
    if not out or cjk_share(out["zh"]) < 0.3:
        return None, "refused_or_not_chinese"
    return out["zh"], "claude"


def translate_batch(texts):
    body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    out = call_json("translate", f"{INSTR}\nReturn the translations in the same order, one per item.\n\n{body}", MANY,
                    max_tokens=8000)
    if out and len(out["zh"]) == len(texts):
        return out["zh"]
    return [translate_one(t)[0] for t in texts]


def main():
    check_budget(5)
    hf_tr, hl_tr, hf_val, hl_val, hf_test = anchor_sets()
    path = ROOT / "data" / "lang_anchor_zh.json"
    done = {r["en"]: r for r in json.load(open(path))} if path.exists() else {}

    harmful = [t for t in dict.fromkeys(hf_tr + hf_val + hf_test) if t not in done or not done[t]["zh"]]
    with ThreadPoolExecutor(8) as ex:
        for t, (zh, src) in zip(harmful, ex.map(translate_one, harmful)):
            done[t] = {"en": t, "zh": zh, "source": src, "harmful": True}
    harmless = [t for t in dict.fromkeys(hl_tr + hl_val) if t not in done or not done[t]["zh"]]
    chunks = [harmless[i:i + 25] for i in range(0, len(harmless), 25)]
    with ThreadPoolExecutor(8) as ex:
        for chunk, zhs in zip(chunks, ex.map(translate_batch, chunks)):
            for t, zh in zip(chunk, zhs):
                done[t] = {"en": t, "zh": zh, "source": "claude" if zh else "failed", "harmful": False}
    json.dump(list(done.values()), open(path, "w"), ensure_ascii=False, indent=1)
    failed = [r["en"] for r in done.values() if not r["zh"]]
    print(f"anchors: {len(done)} translated, {len(failed)} failed")

    us = us_prompts()
    upath = ROOT / "data" / "lang_us_zh.json"
    udone = {r["id"]: r for r in json.load(open(upath))} if upath.exists() else {}
    todo = [r for r in us if r["id"] not in udone or not udone[r["id"]]["zh"]]
    chunks = [todo[i:i + 16] for i in range(0, len(todo), 16)]
    with ThreadPoolExecutor(4) as ex:
        for chunk, zhs in zip(chunks, ex.map(lambda c: translate_batch([r["instruction"] for r in c]), chunks)):
            for r, zh in zip(chunk, zhs):
                udone[r["id"]] = {**r, "zh": zh}
    json.dump(list(udone.values()), open(upath, "w"), ensure_ascii=False, indent=1)
    print(f"US prompts: {len(udone)} translated, {sum(not r['zh'] for r in udone.values())} failed")


if __name__ == "__main__":
    main()
