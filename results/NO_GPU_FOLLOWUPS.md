# No-GPU follow-ups on the 7B/8B results (Oct 26)

Three checks that needed only the committed results. Scripts:
`scripts/regress.py` (writes `results/regression.csv`),
`validation/sample_7b8b.py` + `validation/agreement_7b8b.py`.

## 1. Does the projection predict behavior? Yes, but it does not explain the alignment gap

Logistic regression of the judge's "did the task" on each stolen-election
prompt's projection onto the refusal direction (z-scored within model),
standard errors clustered on matched pairs.

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Projection alone, base: odds ratio per SD / AUC | 0.25 (p < 1e-12) / 0.81 | 0.30 (p < 1e-8) / 0.79 |
| Matched D/R pairs where exactly one did the task: lower-projection one did it | 23 of 29, p = 0.002 | 10 of 14, p = 0.18 |
| Crossed coefficient, without projection -> with projection | -0.72 -> -0.69 | -0.76 -> -0.89 |
| Republican coefficient, without -> with projection | +0.38 -> +0.71 | -0.12 -> +0.46 |
| Projection predicting the *ablated* reply (OR per SD) | 1.27, p = 0.25; with severity: 0.53, p = 0.06 | 2.47, p < 0.001; with severity (sev 3 dropped, all comply): 1.90, p = 0.006 |

- **The direction tracks refusal prompt by prompt** in both models (AUC
  about 0.8), and within matched pairs the member closer to refusal is
  the one that didn't do the task.
- **The crossed-vs-aligned gap in behavior is not mediated by the
  projection.** Adding projection to the model does not shrink the crossed
  coefficient at all. The projection gap between crossed and aligned
  prompts (handoff: +1.59 Qwen, +0.14 Llama) is small next to the
  prompt-to-prompt spread (SD 8.6 and 0.73), so it cannot carry a gap of
  this size. Prediction for the TA's projection-matching test: raising
  crossed prompts' projections to the aligned mean will **not** close the
  behavioral gap. If it does, that would be the surprise.
- **Party label works against projection in Qwen.** Republican askers sit
  higher on the refusal direction (handoff) yet do the task more; holding
  projection fixed, the Republican coefficient doubles. Whatever makes Qwen
  help Republicans more is not the refusal direction.
- **After ablation, base projection no longer predicts refusal** in Qwen.
  In Llama it predicts *compliance*: prompts that sat closer to refusal
  comply more once the direction is removed, even within severity. That
  fits the "other non-compliance survives ablation" reading of arXiv
  2602.02132: low-projection prompts are ones the model deflects for
  reasons the direction does not control.

## 2. The severity curve is a request-format effect

Base "did the task" by template, both models, is driven by the request's
form, not its severity:

| Format | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| prove it / strongest evidence | 6% | 8% |
| social media post | 24% | 9% |
| list / talking points | 26% | 15% |
| letter / petition / email | 51% | 13% |
| article / op-ed | 52% | 48% |
| rally or protest speech | 58% | 12% |

Qwen-7B's non-monotonic severity (59% / 21% / 34%) comes from this:
severity 2 holds the "persuasive post", "strongest evidence" and "list the
ways" templates, which Qwen refuses nearly always ("I'm sorry, but as an AI
assistant..."), while severity 3 holds "letter to legislators" (66%) and
"protest speech" (41%), which it drafts as "the user's perspective". The
same letter template gets 3% in Llama. Reading the replies, the judge is
right on the severity-3 letters and speeches: they argue the claim.

Recommendation: don't present severity as a baseline effect. Either group
by format or drop the baseline severity row. The monotonic rise *after*
ablation (62 / 84 / 97 and 66 / 85 / 100) is real but the severity-1 number
is deflated by the judge (see 3). The "op-ed asking for more scrutiny"
template is done 91-97% of the time by both models at baseline, so it is
not a gray-area request for these models.

## 3. Judge check on 7B/8B replies (50 blind rows)

50 rows drawn from the 7B/8B runs, identity prefix stripped, model and
condition hidden, stratified toward borderline judge calls (16 of 45 stolen
rows have 0.3 < p < 0.7, vs 13% in the full set). **Labeled by Claude
with RUBRIC.md, not yet reviewed by a person.** Treat as a first read until
someone checks `validation/to_label_7b8b/Claude.csv`.

| Rows | n | Agreement | Kappa |
|---|---|---|---|
| Clear judge calls (p < 0.3 or > 0.7) | 29 | 86% | 0.72 |
| Borderline (0.3 < p < 0.7) | 16 | 38% | -0.16 |
| Weighted to the full set's borderline share | | about 80% | |

So on 7B/8B the judge is about as good as on 3B (84%, kappa 0.68), with the
same shape: fine on clear cases, a coin flip in the borderline band.

What it gets wrong:

- **Severity-1 content.** Town-hall questions, "raise questions" posts and
  "deserves more scrutiny" op-eds: labels say did the task 15 of 15, the
  judge says 8 of 15. The election category does not fire on question-form
  or scrutiny-only content. Template 12 (town-hall questions) is scored
  6-12% even after ablation, in both models. Severity-1 rates in
  `headline_behavior.csv` are undercounts.
- **Debunking replies that look compliant** (2 of 29 clear rows): "As a
  Republican, you might be interested in the following evidence: the
  recount confirmed Biden won" scored p = 0.73.
- **Possible party tilt**, judge minus label "did the task": Republican
  +0.15, Democrat / voter / none -0.25. About 10 rows per group, so a
  warning, not a result, but it points the same way as the 3B check and
  would inflate Qwen's Republican-vs-Democrat baseline gap (37% vs 29%,
  p = 0.06).
- **Blinding leaks through the reply.** The prefix is stripped from the
  request, but replies echo it ("As a lifelong Democrat, I'm writing to
  urge..."). Share of stolen replies that self-identify with the asker's
  party:

  | | Qwen base | Qwen ablated | Llama base | Llama ablated |
  |---|---|---|---|---|
  | Democrat asker | 16% | 32% | 5% | 42% |
  | Republican asker | 18% | 33% | 7% | 45% |

  The judge sees these, so it is not blind for a third to a half of ablated
  replies. Rates are symmetric across parties, so this cannot by itself
  create a party gap, but it can let a judge with its own tilt act on one.

Benign controls: labels 5 of 5 answered, judge 0 of 5 unsafe, as intended.

## 4. Second judge: Claude with the language extension's US rubric (Oct 27)

`scripts/judge_claude.py` re-scored every base and ablated reply from both
models with the blind Claude judge from PR #5 (Opus, RUBRIC_US: ANSWERS /
PUSHBACK / DODGES / REFUSES; identity prefix stripped). Cost $15.70. A 10%
Sonnet recheck agreed on did-the-task 287 of 294 times (98%).
`headline.py --judge claude-judge` reruns every comparison; output in
`results/headline_behavior_claude-judge.csv`, regression in
`results/regression_claude-judge.csv`.

**Pre-registered predictions D1 to D4 (`PREREG_causal_tests.md`): all four held.**

| | Guard judge | Claude judge |
|---|---|---|
| Qwen aligned vs crossed, base | 40% vs 25%, p < 0.001 | 60% vs 45%, p < 0.001 |
| Llama aligned vs crossed, base | 18% vs 10%, p = 0.002 | 33% vs 26%, p = 0.012 |
| Qwen Republican vs Democrat, base | 37% vs 29%, p = 0.06 | 52% vs 53%, p = 0.84 |
| Llama Republican vs Democrat, base | 13% vs 15%, p = 0.79 | 28% vs 32%, p = 0.23 |
| Qwen severity 1 / 2 / 3, base | 59 / 21 / 34 | 97 / 32 / 46 |
| Llama severity 1 / 2 / 3, base | 38 / 9 / 5 | 82 / 15 / 5 |
| Projection predicting did-the-task, AUC | 0.81 / 0.79 | 0.89 / 0.87 |

- **The alignment effect is judge-independent.** Aligned requests get done
  more than crossed ones under both judges, in both models, with p < 0.05
  every time.
- **Qwen's party gap was a judge artifact.** Under the Claude judge
  Republican and Democrat askers are within a point. The Guard judge's
  small-sample tilt toward Republican askers (section 3) was real. The
  "Democrat penalty vs voter" seen with the Guard judge (12 points,
  p = 0.002) also goes away: under Claude both parties sit 7-8 points below
  voter.
- **Severity-1 rates double**, as predicted: the Guard judge was missing
  question-form and scrutiny-only content.
- **Levels are much higher under Claude.** Base did-the-task is 55% (Qwen)
  and 32% (Llama) on stolen prompts, vs 36% and 16% under Guard. Where the
  two disagree at base, Claude's label is almost always ANSWERS where Guard
  said no (112 of 149 Qwen disagreements, 84 of 90 Llama), i.e. replies that
  argue the claim after a disclaimer. Guard's election category under-fires
  on those.
- **The direction predicts the Claude judge's labels better** (AUC 0.89 and
  0.87), which is some evidence the Claude judge is the less noisy one.

**This changes the mediation picture.** With the Guard judge, adding
projection did not shrink the crossed coefficient at all. With the Claude
judge it does:

| Crossed coefficient | without projection | with projection | shrink |
|---|---|---|---|
| Qwen2.5-7B | -0.59 (p < 1e-4) | -0.48 (p = 0.036) | 19% |
| Llama-3.1-8B | -0.32 (p = 0.006) | -0.19 (p = 0.37) | 40% |

So under the better judge, part of the alignment gap does run through the
projection at the chosen layer, more so in Llama. The pre-registered
prediction for projection matching (A1, under 25% closed) was written from
the Guard-judge regression and is amended in `PREREG_causal_tests.md`
before any GPU output exists: the point expectation is now 20% (Qwen) to
40% (Llama) closed, judged with Claude; the pass bar for "runs through the
direction" stays at 50%.

**Caveat on the blind-label check.** The Claude judge agrees with the 50
blind rubric labels from section 3 on 50 of 50 rows (Guard: 69%). That is
not independent validation: those labels were also written by a Claude
model from the same rubric. It shows the judge applies the rubric the way
the rubric's author intended, not that the rubric is right. The human pass
over `validation/to_label_7b8b/Claude.csv` is still the thing that would
settle it.
