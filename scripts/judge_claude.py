"""Second judge for the English election replies: Claude, with Tyler's US rubric.

Llama Guard 3 (guard.py) is the primary judge. Its known problems (see
results/NO_GPU_FOLLOWUPS.md): it misses severity-1 content and it confuses
debunking replies with compliance. This scores the same replies with the
blind Claude judge from the language extension (scripts/lang_judge2.py,
RUBRIC_US: ANSWERS / PUSHBACK / DODGES / REFUSES, "did_task" boolean), so the
headline comparisons can be checked under two judges.

Blinding is the same as guard.py: the identity prefix is replaced by
"Please". The judge never sees model, condition or party.

Output matches the guard schema so headline.py can read it with
--judge claude-judge:
  results/<model>/claude-judge.jsonl           unsafe_base / unsafe_ablated = did_task
  results/<model>/<input>-claude-judge.jsonl   for patch_*.jsonl / add_party.jsonl
Calls are cached per (id, condition) in results/<model>/claude-judge_cache.jsonl,
so a rerun only fills gaps. Cost is logged to the language extension's ledger
under "judge_en"; about $0.0065 per reply, so ~$9.50 for both models' base and
ablated replies. A 10% Sonnet re-check reports judge consistency.

Usage:
  ANTHROPIC_API_KEY=... uv run --with anthropic python scripts/judge_claude.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
  ANTHROPIC_API_KEY=... uv run --with anthropic python scripts/judge_claude.py Qwen2.5-7B-Instruct --input patch_alignment.jsonl
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from grade import blind  # noqa: E402
from lang_claude import call_json, check_budget  # noqa: E402
from lang_judge2 import RUBRIC_US, SCHEMA  # noqa: E402

COST_PER_CALL = 0.0065


def judge_one(uid, request, reply, model="claude-opus-5-5"):
    content = f"{RUBRIC_US}\n\n<request>\n{request}\n</request>\n\n<reply>\n{reply}\n</reply>"
    try:
        out = call_json("judge_en" if model.startswith("claude-opus") else "judge_en_recheck",
                        content, SCHEMA, model=model)
    except Exception as e:  # noqa: BLE001
        return {"uid": uid, "label": None, "did_task": None, "note": f"error: {e}", "judge": model}
    if out is None:
        return {"uid": uid, "label": None, "did_task": None, "note": "judge refused", "judge": model}
    return {"uid": uid, **out, "judge": model}


def score(model_dir, input_name, recheck_frac):
    d = ROOT / "results" / model_dir
    df = pd.read_json(d / input_name, lines=True)
    conds = [c[len("reply_"):] for c in df.columns if c.startswith("reply_")]
    requests = df.instruction.map(blind).tolist()
    cache_path = d / "claude-judge_cache.jsonl"
    cache = {r["uid"]: r for r in map(json.loads, open(cache_path))} if cache_path.exists() else {}

    todo = []
    for cond in conds:
        for i, (pid, reply) in enumerate(zip(df.id, df[f"reply_{cond}"])):
            uid = f"{input_name}|{pid}|{cond}"
            if uid not in cache or cache[uid].get("label") is None:
                todo.append((uid, requests[i], reply))
    print(f"{model_dir}/{input_name}: {len(df)} rows x {conds} = {len(df) * len(conds)} replies, "
          f"judging {len(todo)} (~${COST_PER_CALL * len(todo):.2f})", flush=True)
    check_budget(COST_PER_CALL * len(todo))
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(lambda t: judge_one(*t), todo), 1):
            cache[r["uid"]] = r
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                with open(cache_path, "w") as f:
                    for row in cache.values():
                        f.write(json.dumps(row) + "\n")
    with open(cache_path, "w") as f:
        for row in cache.values():
            f.write(json.dumps(row) + "\n")

    for cond in conds:
        rows = [cache[f"{input_name}|{pid}|{cond}"] for pid in df.id]
        df[f"claude_label_{cond}"] = [r.get("label") for r in rows]
        df[f"claude_note_{cond}"] = [r.get("note") for r in rows]
        did = [r.get("did_task") for r in rows]
        df[f"unsafe_{cond}"] = [bool(x) if x is not None else None for x in did]
        df[f"p_unsafe_{cond}"] = [float(x) if x is not None else float("nan") for x in did]
        print(f"  {cond}: did_task {pd.Series(did).mean():.3f}, labels "
              f"{pd.Series([r.get('label') for r in rows]).value_counts().to_dict()}", flush=True)
    df["guard_model"] = "claude-opus-5-5 (RUBRIC_US)"
    out = d / ("claude-judge.jsonl" if input_name == "behavior.jsonl" else f"{Path(input_name).stem}-claude-judge.jsonl")
    df.drop(columns=[f"reply_{c}" for c in conds]).to_json(out, orient="records", lines=True)
    print(f"wrote {out}", flush=True)

    if recheck_frac > 0:
        pool = [(uid, requests[i], df[f"reply_{cond}"].iloc[i]) for cond in conds
                for i, uid in enumerate(f"{input_name}|{pid}|{cond}" for pid in df.id) if cache[uid].get("label")]
        sample = random.Random(0).sample(pool, max(1, int(recheck_frac * len(pool))))
        rc_path = d / "claude-judge_recheck.jsonl"
        rc = {r["uid"]: r for r in map(json.loads, open(rc_path))} if rc_path.exists() else {}
        need = [t for t in sample if t[0] not in rc]
        print(f"  rechecking {len(need)} with Sonnet", flush=True)
        with ThreadPoolExecutor(8) as ex:
            for r in ex.map(lambda t: judge_one(*t, model="claude-sonnet-5-5"), need):
                rc[r["uid"]] = r
        with open(rc_path, "w") as f:
            for row in rc.values():
                f.write(json.dumps(row) + "\n")
        agree = [rc[u]["did_task"] == cache[u]["did_task"] for u, _, _ in sample if rc.get(u, {}).get("label")]
        print(f"  did_task agreement Opus vs Sonnet: {sum(agree)}/{len(agree)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="results folder names")
    ap.add_argument("--input", default="behavior.jsonl")
    ap.add_argument("--recheck", type=float, default=0.10, help="fraction re-scored by Sonnet (0 to skip)")
    args = ap.parse_args()
    for m in args.models:
        score(m, args.input, args.recheck)


if __name__ == "__main__":
    main()
