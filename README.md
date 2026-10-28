# Partisan asymmetry in refusal geometry

Does an open-weight chat model refuse "the election was stolen" requests
differently depending on the user's stated party, and does any difference show
up inside the model, along the refusal direction?

Builds on Arditi et al. 2024, *Refusal in Language Models Is Mediated by a
Single Direction* (arXiv 2406.11717), and checks against *There Is More to
Refusal than a Single Direction* (arXiv 2602.02132).

Main results are on **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**.
An earlier 3B pilot (Qwen2.5-3B, Llama-3.2-3B) is kept for comparison.
Short answer: both models help more when the asker's party matches the
claim's side; naming a party otherwise does nothing a Lakers fan can't do;
in Llama that asymmetry runs through the refusal direction and in Qwen it
mostly doesn't. [HANDOFF.md](HANDOFF.md) has the per-person task list and
file index; [EXPLAINER.md](EXPLAINER.md) is a plain-English walkthrough of
the pilot.

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

Caveats: Llama Guard counts some debunking replies as doing the task and
misses disclaimer-then-argue and question-form replies, so its absolute
rates are off in both directions; compare groups. On a blind human pass
over 50 7B/8B replies (`validation/RESULTS.md`), the Claude second judge
agrees with the human at kappa 0.77 and Llama Guard at 0.51.

### Causal tests (`scripts/patch.py`, `add_party.py`, `alignment_dir.py`)

Ablation removes refusal almost everywhere, so it can't say whether the
direction explains a gap between two groups. Three tests do, all with
predictions written down first (`results/PREREG_causal_tests.md`) and
results in `results/CAUSAL_TESTS.md`:

- **Projection matching** (the TA's test): raise the lower group's
  projection to the other group's mean along the unit refusal direction at
  the chosen layer, with a random-direction control, and judge the replies.
- **Addition by party**: add the direction to the benign election questions
  and ask whether it induces refusal equally for every identity.
- **Alignment direction**: what "crossed" adds to the activations on stolen
  prompts beyond what it adds on benign ones (the language extension's
  censorship-direction construction), its overlap with the refusal
  direction, and a shift test that moves each prompt to the other
  alignment state.

A second judge (Claude, `scripts/judge_claude.py`, the language extension's
rubric) scores everything alongside Llama Guard; `headline.py --judge
claude-judge` reruns the comparisons under it.

### Testing whether the directions "split"

`cos(r_D, r_R)` is near 1 by construction, because the prompts share all
their content, so this is a secondary check. The test looks at the gap
`delta = r_D - r_R`: how much of it lies along the refusal direction, against
a sign-flip null, and whether it matches the D-minus-R gap on benign prompts.
If it does, the gap is "the model noticed a party label", not a
party-specific way of refusing.

## Results

Everything below is from the 920-prompt runs on Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct, judged two ways: **Guard** is Llama Guard 3 with the
election category, **Claude** is the second judge. Baseline "did the task"
rates are on the 544 stolen-election prompts; p-values are exact McNemar
tests on matched pairs unless stated. Reproduce the behavior tables with
`uv run python scripts/headline.py` (add `--judge claude-judge` for the
second judge); the causal tests are in `results/CAUSAL_TESTS.md`, the
no-GPU follow-ups in `results/NO_GPU_FOLLOWUPS.md`, the layer curves in
`figures/`.

### Headline

1. **Both models help more when the asker's party matches the claim's
   side.** A Democrat asking about 2016 or a Republican asking about 2020
   (aligned) gets the content more often than the reverse (crossed). It
   holds under both judges, in both generation runs, within matched pairs,
   and nothing else about identity survives those checks.
2. **There is no Democrat-versus-Republican difference.** The first run's
   8-point Republican advantage in Qwen (p = 0.06) did not reproduce in the
   second run and vanishes under the second judge.
3. **A party label acts like any identity.** "I'm a Lakers fan" lowers
   compliance by about as much as "I'm a Democrat" in both models, and
   moves the refusal projection by as much. What the models react to is
   whose claim it is, not which party you name.
4. **Where the alignment effect lives differs by model.** In Llama it runs
   through the refusal direction: a constant shift along that direction by
   the measured gap closes 70% of it, twice the gap closes it fully, and a
   random shift does nothing. In Qwen the same shift closes a fifth at
   most; the gap is carried by a distinct "alignment direction" that
   overlaps refusal at cosine 0.6 and switches on at the same depth where
   the gap first appears in the layer-wise curves.
5. **The refusal direction is party-blind.** Added to benign election
   questions it induces refusal for every identity to within 1%; ablated,
   it lifts compliance to over 80% for everyone and nearly erases the
   alignment gap.
6. **The paper's safety judge sees none of this.** Llama Guard 2 flags
   zero of these replies, including ones that fully argue an election was
   stolen. Llama Guard 3 with an added election category works but
   undercounts; the Claude judge is validated against a blind human pass at
   kappa 0.77.

### Refusal direction

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Layer / position | 19 of 28 / -2 | 13 of 32 / -5 |
| KL on harmless prompts (paper's cutoff 0.1) | 0.036 | 0.075 |
| Refusal on 64 held-out harmful prompts, before -> after ablation | 92% -> 0% | 91% -> 0% |
| Lowest cosine across 100 bootstrap re-extractions | 0.998 | 0.990 |
| Reproduced from scratch on a second pod (Oct 28) | same layer, KL and norm | same |

Why these layers (`figures/refusal_curve.png`): Llama's layer 13 / -5 is the
strongest ablation of any candidate and passes both filters. In Qwen, 21
candidates at layers 13-21 ablate more than 19 / -2, but every one of them
fails the KL filter; 19 / -2 is the strongest that passes.

### Behavior at baseline

"Did the task" on stolen-election prompts, 136 matched pairs per comparison:

| | Judge | Democrat | Republican | p | Aligned | Crossed | p |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | Guard | 32% | 33% | 1.0 | 40% | 26% | 0.001 |
| | Claude | 53% | 52% | 0.84 | 60% | 45% | <0.001 |
| Llama-3.1-8B | Guard | 16% | 13% | 0.27 | 18% | 10% | 0.003 |
| | Claude | 32% | 28% | 0.23 | 33% | 26% | 0.012 |

Benign civic questions: 0-2% under Guard and 98-100% under Claude in every
cell (Guard's "unsafe" means election-delegitimizing content; Claude's
"did the task" means answered the question), both conditions.

By asker, Guard judge:

| Asker | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Nobody stated | 38% | 21% |
| Voter | 43% | 14% |
| Democrat | 32% | 16% |
| Republican | 33% | 13% |
| Lakers fan | 32% | 10% |

Fan minus voter, matched pairs: -11 points in Qwen (p = 0.008) and -4 in
Llama (n.s.) under Guard; **-12 in both models under Claude (p < 0.01)**.
The pre-registered prediction that fan would sit within 5 points of voter
failed in both models.

### Projection on the refusal direction

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Stolen minus benign prompts (the scale for the rows below) | 14.4 | 2.0 |
| Crossed minus aligned, stolen prompts | +1.56 (11%), p < 0.001 | +0.14 (7%), p < 0.001 |
| Republican minus Democrat, stolen | +0.47 (3%), p = 0.05 | +0.09 (5%), p < 0.001 |
| Each party minus voter, **benign** prompts | +2.2 to +3.0 | +0.15 to +0.26 |
| Lakers fan minus voter, benign prompts | +3.2 | +0.57 |
| Lowest cos(r_D, r_R) over layers | 0.993 | 0.987 |
| Layers where the D-minus-R gap aligns with refusal beyond chance | 2 of 27 | 0 of 31 |

The party-label shift in projection is present on benign questions too and
is matched or exceeded by the Lakers-fan shift, so it is "the model noticed
a group label", not refusal. The crossed-minus-aligned shift is specific to
stolen-election prompts.

**By layer** (`figures/party_gap_by_layer.png`, `scripts/layer_curves.py`):
the crossed-minus-aligned gap on each layer's own refusal direction is zero
in early layers and switches on mid-network, only for stolen requests:
first significant at **layer 19 of 28 in Qwen** and **layer 11 of 32 in
Llama**, peaking at 18% of the stolen-benign scale (Qwen, layer 25) and
13% (Llama, layer 18). Republican minus Democrat is nonzero from the first
layers but tracks the benign questions. Reading: the model represents who
is asking early and combines it with whose claim it is only later, around
the depth where the refusal direction starts to work.

### Does the projection predict behavior?

Logistic regression of "did the task" on each prompt's projection
(`scripts/regress.py`, SEs clustered on matched pairs):

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| AUC of projection alone (Guard / Claude) | 0.81 / 0.89 | 0.79 / 0.87 |
| Discordant D/R pairs where the lower-projection member did the task | 23 of 29, p = 0.002 | 10 of 14 |
| Crossed coefficient, without -> with projection (Claude) | -0.59 -> -0.48 | -0.32 -> -0.19, n.s. |

The direction tracks refusal prompt by prompt. Adding it to the model
shrinks the crossed-vs-aligned term by a fifth in Qwen and two fifths in
Llama: correlational evidence for the split the causal tests then found.

### Ablation

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Did the task, base -> ablated (Guard) | 36% -> 81% | 15% -> 82% |
| Did the task, base -> ablated (Claude) | 55% -> 92% | 32% -> 92% |
| Aligned vs crossed, ablated (Guard) | 85% vs 82%, p = 0.45 | 86% vs 82%, p = 0.23 |
| Benign prompts after ablation | unchanged | unchanged |

Ablation turns most refusals into compliance in both models. What remains
(about 8% of ablated stolen replies under the Claude judge, in each model)
is pushback or dodging, not refusal: the single direction controls flat
refusal, and other ways of not complying survive without it, as arXiv
2602.02132 predicts.

### Causal tests (pre-registered; `results/PREREG_causal_tests.md`)

**Projection matching.** Aligned prompts are pushed along the unit refusal
direction toward the crossed level, at the chosen layer, nothing else
changed. "Did the task", aligned prompts:

| Intervention | Qwen Guard | Qwen Claude | Llama Guard | Llama Claude |
|---|---|---|---|---|
| Before: aligned / crossed | 40% / 26% | 61% / 45% | 18% / 10% | 34% / 27% |
| Each prompt set to the crossed mean | 42% | 61% | 6% | 18% |
| Each prompt set to its **own** group mean (control: gap unchanged) | 46% | 65% | 10% | 22% |
| **Constant shift by the gap, 1x** | 39%, n.s. | 57%, n.s. | 13%, p = 0.09 | 29%, p = 0.016 |
| Constant shift 2x, with random control | | | **11%, p = 0.02**; random 16% | **24%, p < 0.001**; random 34% |
| Constant shift 3x / 10x | 31% / 15% | | 10% / 0% | |
| 10x shift along a random direction | 41% | | 14% | |

"Set to the mean" is confounded: setting each prompt to its own group's
mean, which leaves the gap untouched, moves Llama almost as much, because
it removes the spread across prompts. The constant shift is the clean test.
In Llama it closes 64-70% of the gap at 1x and all of it at 2x, with the
random direction flat. In Qwen it closes 5-23% at 1x; the direction does
control Qwen's compliance (3x closes 63%, 10x overshoots), but it takes
three times the measured gap to do gap-sized work. Democrat raised to
Republican: no gap to explain, and nothing moves.

**Addition by party** (`scripts/add_party.py`). The direction added to the
benign election questions induces refusal for every identity: keyword
refusal 0% -> 99-100%, Claude did-the-task 100% -> 0-3%. Republican minus
Democrat in the induced shift: 0.3% (Qwen) and 0.6% (Llama) of the shift.

**Alignment direction** (`scripts/alignment_dir.py`). What "crossed" adds to
the activations on stolen prompts beyond what it adds on benign ones:

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Split-half cosine (stable?) | 0.98 | 0.98 |
| cos with the refusal direction, chosen layer | **0.60**, p < 0.001 vs sign-flip null | 0.21, n.s. |
| cos by depth | about 0 through layer 15, 0.44 at 16, 0.79 at 25 | 0.2-0.3 everywhere |
| Shift each prompt to the other alignment state: aligned / crossed (Guard; Claude) | 40/26 -> 28/39; 60/47 -> 52/52 | 18/10 -> 17/19; 34/27 -> 35/36 |
| Refusal-parallel component alone | closes the gap | closes the gap |
| Orthogonal component alone | closes most of it (Guard) or half (Claude) | widens it; raises compliance for both groups |
| Random direction of the same norm | nothing | nothing |

In Qwen the alignment direction is a real, stable direction whose overlap
with refusal grows with depth exactly where the layer-wise gap appears, and
shifting prompts along it swaps or closes the aligned/crossed gap under both
judges while the 1x refusal-only shift does not. In Llama it is nearly
orthogonal to refusal and only its refusal component does anything. Llama
encodes "the other side's claim" as more refusal-like, full stop; Qwen
encodes it as more refusal-like and along a second axis the refusal
direction does not see.

### Judges

| | Agreement with 50 blind human labels (7B/8B) | Kappa | Main errors |
|---|---|---|---|
| Llama Guard 2 (the paper's judge) | flags 0 replies | | no election category |
| Llama Guard 3, built-in categories | | 0.42 (3B) | catches about half |
| **Llama Guard 3 + election category** | 76% | 0.51 | misses disclaimer-then-argue and question-form replies; severity-1 rates are undercounts |
| **Claude, US rubric** | 89% | **0.77** | counts "talking points with rebuttals" as compliance, which the rubric asks for |
| Qwen2.5-7B LLM judge (pilot) | | 0.70 (3B) | keyword-triggered |

The two judges disagree on levels by about 20 points (Claude counts
disclaimer-then-argue replies as doing the task) but agree on every
comparison that matters. Opus-vs-Sonnet rechecks agreed 97-98%. Every judge
sees the request with the identity stripped, but a third to a half of
ablated replies echo "As a lifelong Democrat...", so no judge is fully
blind; the Lakers-fan result and the absence of a party gap make this
moot.

### Other things we found on the way

- **Request format beats severity.** Both models draft letters, speeches
  and op-eds far more readily than they produce "the strongest evidence" or
  "a persuasive post": Qwen 51-58% vs 6% at baseline. The three severity
  levels bundle those formats unevenly, which is why Qwen's baseline runs
  59 / 21 / 34 across severity 1 / 2 / 3. Don't read that as a severity
  effect. The "op-ed asking for more scrutiny" request is done 91-97% of
  the time by both models, so it is not a gray-area request for them.
- **Greedy decoding depends on the batch.** The two 920-prompt runs (on
  different days and pods) are byte-identical, but only 29% (Qwen) and 60%
  (Llama) of replies match the 736-prompt first run exactly. Group rates
  agree within 4 points. Compare groups, not individual replies, across
  runs.
- **Severity after ablation is real but judge-sensitive.** With the
  direction removed, did-the-task rises with severity (Qwen 62 / 83 / 96,
  Llama 64 / 83 / 99 under Guard), but the severity-1 number is deflated
  by Guard's blind spot for question-form replies; under Claude severity 1
  is 97-100%.

### Extension: language and censorship (Tyler, `reports/language_censorship.html`)

Same requests in English and Chinese, plus 12 China-sensitive topics with
matched harmless controls:

- Qwen censors China-sensitive requests **21% in English vs 62% in
  Chinese** (24 pairs, p = 0.002); Traditional Chinese gives the same 62%,
  so it is the language, not the script. Llama answers every one in both
  languages. Each model guards its own country's topics: Llama refuses 69%
  of the US stolen-election prompts in English, Qwen 19%.
- 29 of those 42 points are **official-line answers**, not refusals.
  Ablating the refusal direction cuts harmful-request refusal to 0% in both
  languages but leaves Qwen's Chinese gap at 110%. A refusal counter, or an
  abliterated "uncensored" release, would miss this.
- The English and Chinese refusal directions are the same vector (cosine
  0.98 Qwen, 0.91 Llama), and either one ablated removes refusal in both
  languages.
- Exploratory: both models censor the US election claims *less* in Chinese
  (Qwen 19% -> 4%, Llama 69% -> 52%), not significant after correction.

### 3B pilot

The pilot used 384 prompts (four identities, 12 templates) and a Qwen2.5-7B
LLM judge, and found the same shape: no party difference at baseline, and in
Llama-3.2-3B a crossed-vs-aligned gap after ablation (55% vs 38% not done,
p = 0.001). Its files are kept as they were in `results/Qwen2.5-3B-Instruct/`
and `results/Llama-3.2-3B-Instruct/`, re-scored with the Llama Guard judge;
`results/summary_identity_effects.csv` holds its cross-model tests. Qwen-3B
needed a relaxed KL cutoff (0.5); its first unfiltered run is kept in
`results/Qwen2.5-3B-Instruct_run1_kl_unfiltered/`.

### Cost

Three pod runs on one A40 (about $5 total), one H100 run for the language
extension ($2.40), and about $90 of Claude API calls for translation and
judging across both strands.

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

- **Judges:** Llama Guard undercounts compliance and the Claude judge is a
  single model family; the human pass covers 50 replies. Absolute rates
  differ by judge; comparisons do not.
- **Greedy decoding depends on the batch.** The 920-prompt reruns (Divya's
  second run and the Oct 28 pod run) are byte-identical to each other but
  reproduce only 29% (Qwen) and 60% (Llama) of the 736-prompt first run's
  replies exactly; group rates agree within 4 points. Compare groups, not
  individual replies, across runs.
- **Severity is confounded with request format** (see "Other things we
  found"). Don't present baseline severity as a curve.
- **The "none" baseline differs in wording** from the identity prompts. Use
  voter or fan as the no-party baseline.
- **Two models.** Results may not hold for larger or different-family models.

## Extension: language and censorship

Results are summarized under Results above. Report:
`reports/language_censorship.html`. Files, rerun steps and caveats: the
"Extension" section of `HANDOFF.md`. The scripts are `scripts/lang_*.py`.
