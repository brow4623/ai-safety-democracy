#!/usr/bin/env bash
# Full 7B/8B run on a single GPU pod. Run from the repo root on the pod:
#   nohup bash scripts/pod_run.sh > /workspace/logs/run.log 2>&1 &
# Each step writes a marker in /workspace/logs/ when it finishes, so a rerun
# skips finished steps. Needs HF_TOKEN in the environment (gated Meta models).
set -euo pipefail
LOGS=/workspace/logs
mkdir -p "$LOGS"
export HF_HOME=${HF_HOME:-/workspace/hf}

GUARDS=(meta-llama/Meta-Llama-Guard-2-8B meta-llama/Llama-Guard-3-8B)
# The behavior judge: Llama Guard 3 plus our election-delegitimization
# category (guard.py --election-policy). Guard 2 and plain Guard 3 are kept
# for comparison with the paper; neither catches most compliant replies.
JUDGE=(--guard meta-llama/Llama-Guard-3-8B --election-policy)
MODELS=(Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct)
PILOT=(Qwen2.5-3B-Instruct Llama-3.2-3B-Instruct)

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "$LOGS/$name.done" ]]; then echo "== skip $name (done)"; return; fi
  echo "== start $name $(date -u +%H:%M:%S)"
  "$@" > "$LOGS/$name.log" 2>&1
  touch "$LOGS/$name.done"
  echo "== done  $name $(date -u +%H:%M:%S)"
}

step setup uv sync

# 1. Smoke test: Llama Guard on 8 pilot replies, both versions.
for g in "${GUARDS[@]}"; do
  step "smoke-${g##*/}" uv run python scripts/guard.py --results Qwen2.5-3B-Instruct --guard "$g" --sample 8
done

# 2. Llama Guard on the pilot (3B) replies.
for r in "${PILOT[@]}"; do
  for g in "${GUARDS[@]}"; do
    step "pilot-$r-${g##*/}" uv run python scripts/guard.py --results "$r" --guard "$g"
  done
  step "pilot-$r-judge" uv run python scripts/guard.py --results "$r" "${JUDGE[@]}"
done

# 3. Full pipeline on the 7B/8B models, then 4. Llama Guard on their replies.
for m in "${MODELS[@]}"; do
  step "pipeline-${m##*/}" uv run python scripts/run.py --model "$m" --stage all
  for g in "${GUARDS[@]}"; do
    step "guard-${m##*/}-${g##*/}" uv run python scripts/guard.py --results "${m##*/}" --guard "$g"
  done
  step "judge-${m##*/}" uv run python scripts/guard.py --results "${m##*/}" "${JUDGE[@]}"
  # 5. Projection matching (TA's causal test), then judge the shifted replies.
  step "patch-${m##*/}" uv run python scripts/patch.py --model "$m"
  for c in alignment party; do
    step "patch-judge-${m##*/}-$c" uv run python scripts/guard.py --results "${m##*/}" "${JUDGE[@]}" \
      --input "patch_$c.jsonl"
  done
  # 6. Addition by party (handoff item 1): add the direction to the benign
  # election questions and judge the induced replies.
  step "add-party-${m##*/}" uv run python scripts/add_party.py --model "$m"
  step "add-party-judge-${m##*/}" uv run python scripts/guard.py --results "${m##*/}" "${JUDGE[@]}" \
    --input "add_party.jsonl"
  # 7. Alignment direction (what "crossed" adds beyond the party label) and
  # its causal test, then judge those replies.
  step "align-dir-${m##*/}" uv run python scripts/alignment_dir.py --model "$m"
  step "align-dir-judge-${m##*/}" uv run python scripts/guard.py --results "${m##*/}" "${JUDGE[@]}" \
    --input "alignment_dir.jsonl"
done

echo "== ALL DONE $(date -u +%H:%M:%S)"
