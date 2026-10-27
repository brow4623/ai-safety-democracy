# Partisan asymmetry in refusal geometry

Does an open-weight chat model refuse "the election was stolen" requests
differently depending on the user's stated party, and does any difference show
up inside the model, along the refusal direction?

Builds on Arditi et al. 2024, *Refusal in Language Models Is Mediated by a
Single Direction* (arXiv 2406.11717), and checks against *There Is More to
Refusal than a Single Direction* (arXiv 2602.02132).

Main results are on **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**.
An earlier 3B pilot (Qwen2.5-3B, Llama-3.2-3B) is kept for comparison.
[HANDOFF.md](HANDOFF.md) has the per-person task list and file index;
[EXPLAINER.md](EXPLAINER.md) is a plain-English walkthrough of the pilot.

## Design

### Prompts (`src/prompts.py` -> `data/political_prompts.jsonl`)

920 prompts, built as identity x election x request template.

- **Identity:** none, voter, Democrat, Republican, and a non-political control
  ("Lakers fan"). The last four share one sentence shape, so "stated a party"
  can be told apart from "said something about themselves first". Each has
  two phrasings ("I'm a Democrat. Please..." / "As a lifelong Democrat, I'm
  asking you to...") so a result can't hinge on one string. The "none"
  identity is just "Please", so use voter or fan as the fair no-party baseline.
- **Election:** four where the "stolen" story is coded Republican (2020
  presidential, 2020 GA, 2020 MI, 2022 AZ governor) and four where it is coded
  Democratic (2016 presidential, 2000 FL, 2004 OH, 2018 GA governor).
- **Request:** 17 stolen-election templates at three severity levels: doubt
  (raise questions, 5 templates), claim (assert it was stolen, 7), and act
  (push to overturn or not recognize the result, 5). Plus 6 benign civic
  questions about the same elections as a control.

Crossing election with identity separates two things the original example (a
Democrat asking to prove 2020 was stolen) changes at once: who is asking and
whose claim it is. That gives **aligned** prompts (a Democrat on 2016, a
Republican on 2020) and **crossed** prompts (the reverse). Within one
(election, template) cell the prompts differ only in the identity prefix, so
there are 136 matched Democrat/Republican pairs on the stolen prompts and 48
on the benign ones.

### Pipeline (`scripts/run.py`)

Three stages:

1. **direction:** compute difference-of-means directions from the standard
   non-political harmful vs. harmless prompts (Arditi et al.'s splits, in
   `data/anchor/`) at every layer up to 80% depth and every post-instruction
   position. Pick the candidate whose ablation most reduces refusal on held-out
   harmful prompts, subject to: it must also induce refusal when added to
   harmless prompts, and it must barely change harmless outputs (KL under the
   paper's 0.1; `--kl-max` is a fallback if nothing passes). Then check it on
   64 fresh held-out harmful prompts and bootstrap its stability.
2. **geometry:** project every political prompt onto that direction at the
   chosen layer, and onto each layer's own direction. Compute paired identity
   effects with sign-flip permutation tests. Build party-specific directions
   from the political prompts and test whether they split.
3. **behavior:** generate replies (greedy, 128 tokens) with and without the
   direction ablated.

Why the direction comes from non-political prompts: a direction taken from
the political prompts themselves would mix "refusal" with "this is about
partisan politics", and the two could not be separated afterward. The benign
political control shows how much of the projection is just topic.

### Judge (`scripts/guard.py`)

Replies are scored by **Llama Guard 3 with one added category** for election
delegitimization (`--election-policy`). "Unsafe" means the reply did the task:
it produced the requested stolen-election content. Every judge sees the
request with the identity prefix replaced by "Please", so it cannot react to
party.

Why not the paper's judge: Llama Guard 2 flags **none** of these replies as
unsafe, including ones that fully argue an election was stolen. Plain Llama
Guard 3 catches about half. With the added category, agreement with 171
human-reviewed labels is 84% (kappa 0.68), about the same as a Qwen2.5-7B
LLM judge (`scripts/grade.py`, kappa 0.70), and its errors barely depend on
who asked. Details in [validation/RESULTS.md](validation/RESULTS.md).

Caveats: the judge counts about a third of debunking replies as doing the
task, so absolute rates run high. Compare groups; don't quote raw levels as
exact. It was validated on 3B replies, and the reference labels are
Claude-written and human-reviewed, not independently labeled.

### Projection matching (`scripts/patch.py`, not yet run)

Ablation removes refusal almost everywhere, so it can't say whether the
direction explains a gap between two groups. The TA's test instead raises the
lower group's projection to the other group's mean, by adding a multiple of
the unit refusal direction at the chosen layer, and judges the result. The
same shift along a random orthogonal direction is the control. Code is
written and checked; results need one GPU run (see HANDOFF.md).

### Testing whether the directions "split"

`cos(r_D, r_R)` is near 1 by construction, because the prompts share all
their content, so this is a secondary check. The test looks at the gap
`delta = r_D - r_R`: how much of it lies along the refusal direction, against
a sign-flip null, and whether it matches the D-minus-R gap on benign prompts.
If it does, the gap is "the model noticed a party label", not a
party-specific way of refusing.

## Results

Reproduce with `uv run python scripts/headline.py Qwen2.5-7B-Instruct
Llama-3.1-8B-Instruct`. All behavior comparisons are in
`results/headline_behavior.csv`.

### Refusal direction

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Layer / position | 19 of 28 / -2 | 13 of 32 / -5 |
| KL on harmless prompts (paper's cutoff 0.1) | 0.036 | 0.075 |
| Refusal on 64 held-out harmful prompts, before -> after ablation | 92% -> 0% | 91% -> 0% |
| Lowest cosine across 100 bootstrap re-extractions | 0.998 | 0.990 |

### "Did the task" on stolen-election prompts, baseline (judge-scored)

| | Democrat | Republican | p | Aligned (own side) | Crossed (other side) | p |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 29% | 37% | 0.06 | 40% | 25% | <0.001 |
| Llama-3.1-8B | 15% | 13% | 0.79 | 18% | 10% | 0.002 |

p-values are exact McNemar tests on matched pairs. Benign controls: 0-2% in
every cell, both models, both conditions.

Example: Qwen-7B helps a Republican argue a Republican-coded claim (2020,
Arizona 2022) 47% of the time, a Democrat with the same request 24%.

### Projection on the refusal direction

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Crossed minus aligned, stolen prompts | +1.59, p < 0.001 | +0.14, p < 0.001 |
| Same gap as a share of the stolen-vs-benign gap | 11% | 7% |
| Lowest cos(r_D, r_R) over layers | 0.993 | 0.987 |
| Layers where the D-minus-R gap aligns with refusal beyond chance | 1 of 27 | 0 of 31 |

### After ablation

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Did the task, overall, base -> ablated | 36% -> 82% | 16% -> 84% |
| Aligned vs crossed, ablated | 84% vs 79%, p = 0.24 | 88% vs 82%, p = 0.04 |
| Did the task by severity 1 / 2 / 3, ablated | 62% / 84% / 97% | 66% / 85% / 100% |

### Main points

1. **No overall party difference** in what either model does at baseline
   (Democrat vs Republican: p = 0.06 Qwen, p = 0.79 Llama).
2. **Both models favor your own side's claim.** Aligned requests get done
   more often than crossed ones in both models, and crossed requests sit
   closer to refusal on the direction. The effect is modest, roughly a tenth
   of the full "this is a refusal case" signal.
3. **One shared direction, not two** (secondary). cos(r_D, r_R) is at least
   0.987 at every layer, and the D-minus-R gap is within the chance band along
   the refusal direction at almost every layer.
4. **Ablation is causal and mostly removes the asymmetry.** It lifts "did the
   task" to about 80% or more for every identity, leaves benign prompts at 0%, and
   shrinks the aligned-vs-crossed gap to a few points. With the direction
   removed, compliance rises with severity.
5. **The paper's safety judge misses this category.** Llama Guard 2 does not
   treat election delegitimization as unsafe at all.

### 3B pilot

The pilot used 384 prompts (four identities, 12 templates) and a Qwen2.5-7B
LLM judge, and found the same shape: no party difference at baseline, and in
Llama-3.2-3B a crossed-vs-aligned gap after ablation (55% vs 38% not done,
p = 0.001). Its files are kept as they were in `results/Qwen2.5-3B-Instruct/`
and `results/Llama-3.2-3B-Instruct/`, re-scored with the Llama Guard judge;
`results/summary_identity_effects.csv` holds its cross-model tests. Qwen-3B
needed a relaxed KL cutoff (0.5); its first unfiltered run is kept in
`results/Qwen2.5-3B-Instruct_run1_kl_unfiltered/`.

## Running

```
uv sync
uv run python src/prompts.py
uv run python scripts/run.py --model Qwen/Qwen2.5-7B-Instruct --stage all
uv run python scripts/guard.py --results Qwen2.5-7B-Instruct --guard meta-llama/Llama-Guard-3-8B --election-policy
uv run python scripts/patch.py --model Qwen/Qwen2.5-7B-Instruct
uv run python scripts/guard.py --results Qwen2.5-7B-Instruct --guard meta-llama/Llama-Guard-3-8B --election-policy --input patch_alignment.jsonl
uv run python scripts/headline.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
```

`scripts/pod_run.sh` runs all of this for both models on one GPU pod (about
an hour on an A40, with judging). Each step leaves a `.done` marker in
`/workspace/logs/`, so a rerun skips finished steps; delete the markers to
regenerate. Gated Meta models (Llama-3.1-8B, Llama Guard) need `HF_TOKEN` in
the environment. The 3B models run on an M-series Mac in about 30 minutes;
`unsloth/Llama-3.2-3B-Instruct` is an ungated copy of Meta's weights.

Results go to `results/<model>/`. Tensors (`*.pt`) are not in git; see
HANDOFF.md for the file index and formats.

## Known limits

- **Judge:** validated on 3B replies only; over-counts debunking replies as
  compliant; reference labels are AI-written and human-reviewed.
- **The committed 7B/8B results predate the `fan` identity.** They were made
  from the first 736 prompts. The fan control and projection matching need
  one more GPU run.
- **Qwen-7B's baseline is non-monotonic in severity** (34% at level 3 vs 21%
  at level 2). Read those replies before putting severity on a slide.
- **The "none" baseline differs in wording** from the identity prompts. Use
  voter or fan as the no-party baseline.
- **Two models.** Results may not hold for larger or different-family models.
