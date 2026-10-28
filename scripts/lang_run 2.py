"""Generate replies to the language-test prompts (data/lang_prompts.jsonl).

    uv run python scripts/lang_run.py --models Qwen/Qwen2.5-7B-Instruct 01-ai/Yi-1.5-9B-Chat unsloth/Llama-3.1-8B-Instruct

Writes results/<model>/lang_test.jsonl: one row per prompt with the reply and
the refusal-lean score (log-odds the reply opens with "I" / "As" / "Sorry";
that score is English-centric, so compare it within a language, not across).
Score the replies with scripts/lang_judge.py.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(ROOT / "data" / "lang_prompts.jsonl")]
    texts = [p["prompt"] for p in prompts]
    for name in args.models:
        print(f"== {name}", flush=True)
        m = Model(name)
        print(f"   eoi_len={m.eoi_len} layers={m.n_layers}", flush=True)
        scores = m.refusal_score(texts).tolist()
        replies = m.generate(texts, max_new_tokens=args.max_new_tokens, bs=32)
        out = ROOT / "results" / name.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "lang_test.jsonl", "w") as f:
            for p, s, r in zip(prompts, scores, replies):
                f.write(json.dumps({**p, "model": name, "refusal_score": round(s, 3), "reply": r},
                                   ensure_ascii=False) + "\n")
        print(f"   wrote {out / 'lang_test.jsonl'}", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
