"""Shared Claude API helper for the language experiment: one client, a JSON-schema
call, and a running cost ledger (results/lang_experiment/cost_ledger.json).

Run scripts that import this with:
    ANTHROPIC_API_KEY=$(cat ~/.config/anthropic_key) uv run --no-project --with anthropic python scripts/<x>.py
"""

import json
import threading
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "lang_experiment" / "cost_ledger.json"
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}  # $ per M tokens in/out
CAP = 60.0

client = anthropic.Anthropic(max_retries=6)
_lock = threading.Lock()


def load_ledger():
    if LEDGER.exists():
        return json.load(open(LEDGER))
    return {"cap_usd": CAP, "prior_spend_usd": 4.0,
            "prior_note": "pilot pods (~$1.65) + pilot judging and smoke tests (~$2.35), estimated",
            "runpod": [], "claude": {}}


def save_ledger(led):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    json.dump(led, open(LEDGER, "w"), indent=2)


def total_spend(led=None):
    led = led or load_ledger()
    claude = sum(v["usd"] for v in led["claude"].values())
    pods = sum(p.get("usd", 0) for p in led["runpod"])
    return led["prior_spend_usd"] + claude + pods


def _record(task, model, usage):
    pin, pout = PRICES.get(model, PRICES["claude-opus-5-5"])
    usd = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
    with _lock:
        led = load_ledger()
        t = led["claude"].setdefault(task, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        t["calls"] += 1
        t["input_tokens"] += usage.input_tokens
        t["output_tokens"] += usage.output_tokens
        t["usd"] = round(t["usd"] + usd, 5)
        save_ledger(led)


def call_json(task, content, schema, model="claude-opus-5-5", effort="low", max_tokens=4000):
    """One request returning parsed JSON, or None if the whole fallback chain refused."""
    kwargs = dict(model=model, max_tokens=max_tokens,
                  output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                  messages=[{"role": "user", "content": content}])
    if model == "claude-opus-5-5":
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    _record(task, resp.model if resp.model in PRICES else model, resp.usage)
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(text) if text else None


def check_budget(projected_extra=0.0):
    spent = total_spend()
    if spent + projected_extra > CAP:
        raise SystemExit(f"BUDGET STOP: spent ${spent:.2f} + projected ${projected_extra:.2f} > cap ${CAP}")
    return spent
