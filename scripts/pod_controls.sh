#!/usr/bin/env bash
# Projection-matching controls, after pod_rerun.sh:
#   own     each source prompt set to its own group's mean (gap unchanged)
#   shift   constant shift by 1x the group-mean gap (keeps the spread)
#   dose    constant shift by 3x and 10x the gap (10x also with a random direction)
#   nohup bash scripts/pod_controls.sh > /workspace/logs/controls.log 2>&1 &
set -euo pipefail
LOGS=/workspace/logs
export HF_HOME=${HF_HOME:-/workspace/hf}
JUDGE=(--guard meta-llama/Llama-Guard-3-8B --election-policy)
MODELS=(Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct)

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "$LOGS/$name.done" ]]; then echo "== skip $name (done)"; return; fi
  echo "== start $name $(date -u +%H:%M:%S)"
  "$@" > "$LOGS/$name.log" 2>&1
  touch "$LOGS/$name.done"
  echo "== done  $name $(date -u +%H:%M:%S)"
}

for m in "${MODELS[@]}"; do
  r=${m##*/}
  step "own-$r" uv run python scripts/patch.py --model "$m" --mode own
  step "shift1-$r" uv run python scripts/patch.py --model "$m" --mode shift --scale 1
  step "shift3-$r" uv run python scripts/patch.py --model "$m" --mode shift --scale 3 --comparisons alignment
  step "shift10-$r" uv run python scripts/patch.py --model "$m" --mode shift --scale 10 --comparisons alignment --random
  for f in alignment_own1 party_own1 alignment_shift1 party_shift1 alignment_shift3 alignment_shift10; do
    step "judge-$r-$f" uv run python scripts/guard.py --results "$r" "${JUDGE[@]}" --input "patch_$f.jsonl"
  done
done
echo "== ALL DONE $(date -u +%H:%M:%S)"
