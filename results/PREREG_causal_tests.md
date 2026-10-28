# Pre-registered predictions for the causal tests

Written 2026-10-27, before the GPU run that produces them. Same idea as the
language extension's pass bars: say what we expect, with a number, so the
result reads as a test. Basis for each prediction is the committed 7B/8B
results and the regression in `NO_GPU_FOLLOWUPS.md`.

"Did the task" below is the Llama Guard 3 + election judge unless stated.
Baseline rates for reference (stolen-election prompts):

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Aligned / crossed, base | 40% / 25% | 18% / 10% |
| Republican / Democrat, base | 37% / 29% | 13% / 15% |
| Crossed minus aligned projection | +1.59 (0.18 SD) | +0.14 (0.20 SD) |
| Democrat minus Republican projection | -0.47 (0.05 SD) | -0.09 (0.12 SD) |

## A. Projection matching (`scripts/patch.py`, judged with guard.py)

**A1. Alignment.** Crossed prompts are raised to the aligned group's mean
projection at the chosen layer.

- Prediction: the patched crossed rate closes **less than 25%** of the
  aligned-minus-crossed gap (Qwen: moves less than 4 points toward 40%;
  Llama: less than 2 points toward 18%), and the random-direction control
  lands within 5 points of the patched rate.
- Why: in the regression, adding projection to the model does not shrink
  the crossed coefficient at all, and the projection shift being applied is
  a fifth of a standard deviation. The language extension moved Llama's
  election refusal by 23 points with a shift of over one standard deviation,
  so the machinery works when the shift is large; this one is small.
- Pass bar for the opposite conclusion: if the patched rate closes **50% or
  more** of the gap and the random control does not, the alignment
  asymmetry does run through the direction and conclusion 1 of
  `NO_GPU_FOLLOWUPS.md` is wrong.

**A2. Party.** Democrat prompts (the lower-projection group in both models)
are raised to the Republican mean.

- Prediction: Democrats' patched rate moves **toward refusal** by under 5
  points, i.e. away from the Republican rate in Qwen, not toward it. Raising
  projection can only reduce compliance, and in Qwen the party gap goes the
  other way (Republicans sit higher on the direction and comply more).
- This test cannot confirm a party mechanism in Qwen under any outcome; it
  is included because the TA asked for it and because a large move would
  mean the direction is more sensitive than the regression suggests.

## B. Addition by party (`scripts/add_party.py`)

The refusal direction is added (coeff 1.0, the direction stage's scale) to
the benign election questions, which nobody refuses at baseline, and the
induced rise in the logit refusal score is compared across identities.

- Prediction: the induced rise is **symmetric**. The paired Republican-minus-
  Democrat difference in the added score is under 10% of the mean induced
  rise, sign-flip p > 0.05, in both models.
- Why: no party difference in behavior at baseline, and the direction was
  built from non-political prompts, so it has no party content to act on.
- If asymmetric (p < 0.05 and over 25% of the rise): the direction interacts
  with party, which would be a new finding and would need the coeff sweep
  (0.5, 2.0) to show it is not a saturation artifact.

## C. Non-political identity control ("Lakers fan")

- Prediction: fan behaves like voter. |fan minus voter| did-the-task is
  under 5 points in both conditions and both models, and the fan-minus-voter
  projection gap is inside the voter-minus-none band. Then the Democrat and
  Republican effects are about party, not about stating any group identity.
- If fan differs from voter by as much as the parties do: the "identity
  effects" are partly "any self-description" effects and the README's
  framing needs to change.

## D. Second judge (`scripts/judge_claude.py`, Claude with the US rubric)

- D1. The aligned-over-crossed gap at baseline **holds under the second
  judge**: aligned higher than crossed in both models, McNemar p < 0.05 in
  Qwen, same direction in Llama.
- D2. Severity-1 "did the task" rates **rise by 20 points or more** under
  the Claude judge, since it reads question-form and scrutiny-only content
  as doing the task and Llama Guard does not.
- D3. Qwen's Republican-vs-Democrat baseline gap (8 points, p = 0.06)
  **shrinks** under the Claude judge, consistent with the Guard judge's
  small-sample tilt toward Republican askers. Llama's stays null.
- D4. Opus-vs-Sonnet did_task agreement on the 10% recheck is 90% or
  higher, as in the language extension.

## What we will do with each outcome

- A1 null as predicted + B symmetric: write up "the direction is causal for
  refusal but not for the partisan asymmetry", then build the alignment
  direction (what being crossed adds on stolen prompts beyond what it adds
  on benign ones, the language extension's censorship-direction method) and
  test it with the same patching.
- A1 closes the gap: the direction carries the asymmetry; drop the
  alignment-direction plan and report the projection-matching result as the
  headline causal finding.
- D1 fails: the aligned-vs-crossed finding is judge-dependent and goes on
  the slide with that caveat, whichever judge a reader prefers.

---

## Amendment, 2026-10-27 (still before the GPU run)

The second judge (D) was run first because it needed no GPU. All four D
predictions held (`NO_GPU_FOLLOWUPS.md` section 4). But the regression
under the Claude judge, which the direction predicts better (AUC 0.89 vs
0.81), shows partial mediation of the crossed effect by projection: the
crossed coefficient shrinks 19% (Qwen) and 40% (Llama) when projection is
added. The Guard-judge regression that prediction A1 was built on showed
0%.

Amended A1 expectation: projection matching closes **about 20% (Qwen) to
40% (Llama)** of the aligned-minus-crossed gap when judged with Claude, and
less under Guard. Pass bars are unchanged: **50% or more** closed with the
random control under 25% means the gap runs through the direction; **under
25%** means it does not; between is partial mediation, reported as such.
Both judges will be run on the patched replies (`guard.py --input` and
`judge_claude.py --input patch_alignment.jsonl`) and both reported.

A2, B and C are unchanged. B's symmetry prediction is, if anything,
strengthened: the party gap itself is gone under the Claude judge.


---

## Verdicts (2026-10-28)

Results and the verdict table are in `CAUSAL_TESTS.md` section F. Short
form: A1 held in Qwen (0% closed) and was exceeded in Llama (the gap runs
through the direction, 150-220% closed, random control 0%); B symmetric to
under 1% of the induced shift; C failed in Qwen (a Lakers fan is treated like
a party member); D all held.
