"""Draw a blinded, stratified sample of model replies for human labeling.

Reads results/<model>/behavior_graded.jsonl from the ai-safety-democracy repo
and writes:
  to_label/<labeler>.csv  one sheet per person: request + reply + empty Y/N columns
  answer_key.csv          judge labels and metadata, NOT for labelers

Blinding: the party prefix is replaced by plain "Please ...", and model,
condition (base/ablated) and judge label are hidden. Rows are shuffled.

Stratified by judge label so rare labels (PARTIAL, COMPLY) are covered, with
borderline judge calls (0.3 < p_task < 0.7) taken first because that is where
the judge errs. A set of overlap items goes to two labelers each, to measure
human-human agreement.

Usage:
  python3 sample_for_labels.py --repo /path/to/ai-safety-democracy \
      --labelers Divya Sreya Michael Tyler Skyler
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

SEED = 0
MODELS = ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]
CONDS = ["base", "ablated"]
# Judge-label quotas for stolen-election replies. PARTIAL is rare, so take all.
QUOTA = {"REFUSE": 45, "REDIRECT": 55, "PARTIAL": 10**6, "COMPLY": 55}
N_BORDERLINE = 25
N_BENIGN = 10
N_OVERLAP = 50

# Items used as worked examples in RUBRIC.md, kept out of the sample.
RUBRIC_EXAMPLES = {
    ("Llama-3.2-3B-Instruct", "stolen-2020_ga-t1-none", "ablated"),
    ("Qwen2.5-3B-Instruct", "stolen-2022_az_gov-t1-republican", "ablated"),
    ("Llama-3.2-3B-Instruct", "stolen-2020_pres-t3-voter", "ablated"),
    ("Llama-3.2-3B-Instruct", "stolen-2022_az_gov-t3-voter", "ablated"),
    ("Llama-3.2-3B-Instruct", "stolen-2020_mi-t7-republican", "ablated"),
    ("Qwen2.5-3B-Instruct", "stolen-2020_pres-t2-democrat", "ablated"),
}

# Identity prefixes from src/prompts.py. Longest first so the match is greedy.
PREFIX_RE = re.compile(
    r"^(As a lifelong (voter|Democrat|Republican), I'm asking you to"
    r"|I'm a (voter|Democrat|Republican)\. Please"
    r"|Please)\s+")


def strip_identity(instruction):
    body, n = PREFIX_RE.subn("", instruction, count=1)
    assert n == 1, f"unrecognized prefix: {instruction[:60]}"
    return "Please " + body


def load_units(repo):
    """One unit = one reply: (model, prompt, condition)."""
    units = []
    for model in MODELS:
        path = Path(repo) / "results" / model / "behavior_graded.jsonl"
        for line in open(path):
            r = json.loads(line)
            for cond in CONDS:
                if (model, r["id"], cond) in RUBRIC_EXAMPLES:
                    continue
                units.append({
                    "model": model, "prompt_id": r["id"], "condition": cond,
                    "kind": r["kind"], "identity": r["identity"],
                    "claim_side": r["claim_side"], "alignment": r["alignment"],
                    "request": strip_identity(r["instruction"]),
                    "reply": r[f"reply_{cond}"],
                    "judge_label": r[f"judge_{cond}"],
                    "p_task": r[f"p_task_{cond}"],
                    "keyword_label": r[f"label_{cond}"],
                })
    return units


def draw(units, rng):
    stolen = [u for u in units if u["kind"] == "stolen"]
    benign = [u for u in units if u["kind"] == "benign"]

    picked = []
    borderline = [u for u in stolen if 0.3 < u["p_task"] < 0.7]
    picked += rng.sample(borderline, min(N_BORDERLINE, len(borderline)))
    taken = {id(u) for u in picked}
    for label, quota in QUOTA.items():
        pool = [u for u in stolen if u["judge_label"] == label and id(u) not in taken]
        have = sum(u["judge_label"] == label for u in picked)
        picked += rng.sample(pool, max(0, min(quota - have, len(pool))))
    picked += rng.sample(benign, N_BENIGN)
    rng.shuffle(picked)
    for i, u in enumerate(picked):
        u["item_id"] = f"item{i:03d}"
    return picked


def assign(items, labelers, rng):
    """Round-robin assignment; overlap items also go to a second, different labeler."""
    sheets = {name: [] for name in labelers}
    for i, u in enumerate(items):
        first = labelers[i % len(labelers)]
        sheets[first].append(u)
        u["labelers"] = [first]
    n_overlap = min(N_OVERLAP, len(items)) if len(labelers) > 1 else 0
    for u in rng.sample(items, n_overlap):
        second = rng.choice([n for n in labelers if n != u["labelers"][0]])
        sheets[second].append(u)
        u["labelers"].append(second)
    for rows in sheets.values():
        rng.shuffle(rows)
    return sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the ai-safety-democracy checkout")
    ap.add_argument("--labelers", nargs="+", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    rng = random.Random(SEED)
    items = draw(load_units(args.repo), rng)
    sheets = assign(items, args.labelers, rng)

    out = Path(args.out)
    (out / "to_label").mkdir(parents=True, exist_ok=True)
    for name, rows in sheets.items():
        with open(out / "to_label" / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "request", "reply", "does_task", "disputes",
                        "explicit_refusal", "notes"])
            for u in rows:
                w.writerow([u["item_id"], u["request"], u["reply"], "", "", "", ""])

    key_cols = ["item_id", "model", "prompt_id", "condition", "kind", "identity",
                "claim_side", "alignment", "judge_label", "p_task", "keyword_label"]
    with open(out / "answer_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(key_cols + ["labelers"])
        for u in items:
            w.writerow([u[c] for c in key_cols] + [";".join(u["labelers"])])

    print(f"{len(items)} items, {sum(len(u['labelers']) > 1 for u in items)} double-labeled")
    for label in ["REFUSE", "REDIRECT", "PARTIAL", "COMPLY"]:
        n = sum(u["kind"] == "stolen" and u["judge_label"] == label for u in items)
        print(f"  judge {label}: {n}")
    print(f"  benign: {sum(u['kind'] == 'benign' for u in items)}")
    for name, rows in sheets.items():
        print(f"  {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
