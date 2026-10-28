#!/usr/bin/env bash
# Language & refusal experiment, GPU side. Run from the repo root on the pod:
#   nohup bash scripts/lang_pod_run.sh > /workspace/logs/lang.log 2>&1 &
# Each step leaves a marker in /workspace/logs, so a rerun skips finished steps.
set -euo pipefail
LOGS=/workspace/logs
mkdir -p "$LOGS"
export HF_HOME=${HF_HOME:-/workspace/hf}
export PATH=$HOME/.local/bin:$PATH

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "$LOGS/$name.done" ]]; then echo "== skip $name (done)"; return; fi
  echo "== start $name $(date -u +%H:%M:%S)"
  "$@" > "$LOGS/$name.log" 2>&1
  touch "$LOGS/$name.done"
  echo "== done  $name $(date -u +%H:%M:%S)"
}

step setup uv sync

# Qwen gets the Chinese-direction ablation as well; Llama is the English-only comparison.
step qwen-direction uv run python scripts/lang_experiment.py --model Qwen/Qwen2.5-7B-Instruct --stage direction
step qwen-behavior  uv run python scripts/lang_experiment.py --model Qwen/Qwen2.5-7B-Instruct --stage behavior --ablate-zh
step qwen-patch     uv run python scripts/lang_experiment.py --model Qwen/Qwen2.5-7B-Instruct --stage patch

step llama-direction uv run python scripts/lang_experiment.py --model unsloth/Llama-3.1-8B-Instruct --stage direction
step llama-behavior  uv run python scripts/lang_experiment.py --model unsloth/Llama-3.1-8B-Instruct --stage behavior
step llama-patch     uv run python scripts/lang_experiment.py --model unsloth/Llama-3.1-8B-Instruct --stage patch

echo "== ALL DONE $(date -u +%H:%M:%S)"
