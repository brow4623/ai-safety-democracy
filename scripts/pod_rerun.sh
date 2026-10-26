#!/usr/bin/env bash
# Second pod run: re-runs geometry and behavior on the 920-prompt set (with the
# Lakers-fan control), judges the replies, then runs projection matching and
# judges those replies. Reuses each model's saved direction.pt and
# anchor_means.pt, so the direction is identical to the first run.
#   nohup bash scripts/pod_rerun.sh > /workspace/logs/rerun.log 2>&1 &
set -euo pipefail
LOGS=/workspace/logs
mkdir -p "$LOGS"
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

step setup uv sync
for m in "${MODELS[@]}"; do
  r=${m##*/}
  test -f "results/$r/direction.pt" -a -f "results/$r/anchor_means.pt"
  step "geometry-$r" uv run python scripts/run.py --model "$m" --stage geometry
  step "behavior-$r" uv run python scripts/run.py --model "$m" --stage behavior
  step "judge-$r" uv run python scripts/guard.py --results "$r" "${JUDGE[@]}"
  step "patch-$r" uv run python scripts/patch.py --model "$m"
  for c in alignment party; do
    step "patch-judge-$r-$c" uv run python scripts/guard.py --results "$r" "${JUDGE[@]}" --input "patch_$c.jsonl"
  done
done
echo "== ALL DONE $(date -u +%H:%M:%S)"
