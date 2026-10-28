# Causal tests: results against the pre-registration (Oct 27-28)

GPU run on one pod (Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, about
50 minutes plus 25 for the alignment direction), under $5. Every reply was
regenerated, so the fan identity is in all files. Judged with Llama Guard 3
+ election category (Guard) on the pod and with the Claude judge (Claude,
`scripts/judge_claude.py`) on a Mac. Predictions and pass bars are in
`PREREG_causal_tests.md`; the verdict table is at the end.

Scripts: `patch_analyze.py`, `identity_control.py`, `add_party.py`'s
`add_party_test.csv`, `alignment_dir.py`. Summary CSVs:
`patch_summary*.csv`, `identity_control*.csv`.

## A. Projection matching: the two models split

`patch.py` raises the lower-projection group to the higher group's mean
projection at the chosen layer by adding a multiple of the unit refusal
direction, then judges the replies. For both comparisons the lower group was
**aligned** (alignment) and **Democrat** (party), so the test is: does
pushing them *up* the refusal direction, to where crossed / Republican
prompts sit, make them behave like that group?

"Did the task", source group: base, patched, random-direction control; and
the target group's base rate.

| Model, comparison | Judge | Source base | Patched | Random | Target base | p (patched vs base) |
|---|---|---|---|---|---|---|
| Qwen, aligned raised to crossed (shift 1.56) | Guard | 40% | 42% | 41% | 26% | 0.75 |
| | Claude | 61% | 61% | 61% | 45% | 1.0 |
| Qwen, Democrat raised to Republican (shift 0.48) | Guard | 32% | 38% | 33% | 33% | 0.25 |
| | Claude | 54% | 57% | 57% | 52% | 0.54 |
| Llama, aligned raised to crossed (shift 0.14) | Guard | 18% | **6%** | 16% | 10% | 0.002 |
| | Claude | 34% | **18%** | 34% | 26% | <0.001 |
| Llama, Democrat raised to Republican (shift 0.09) | Guard | 16% | **3%** | 13% | 13% | <0.001 |
| | Claude | 31% | **15%** | 32% | 28% | <0.001 |

- **Qwen: as predicted (A1).** Moving aligned prompts 1.56 units up the
  refusal direction, a fifth of a standard deviation but the full
  crossed-minus-aligned gap, changes nothing under either judge. The
  refusal direction does not carry Qwen's alignment asymmetry.
- **Llama: the opposite.** A shift of 0.14 units, also a fifth of a
  standard deviation, cuts aligned prompts' compliance by two thirds and
  overshoots the crossed rate; the random direction of the same size does
  nothing. Same for party. The refusal direction carries Llama's gap, and
  Llama is far more sensitive to small moves along it than Qwen (the
  language extension saw the same: 23 points from a one-SD shift).
- **A2 (party) in Qwen:** Democrats raised to the Republican projection
  comply slightly *more*, not less, n.s. As predicted, this test cannot
  produce a party mechanism in Qwen; and the party gap itself did not
  reproduce (see F).

## B. Addition by party: symmetric

`add_party.py` adds the refusal direction (coeff 1, the direction stage's
scale) at the chosen layer to the benign election questions, per identity.

| | Qwen | Llama |
|---|---|---|
| Benign did-the-task, base to added (Claude) | 100% to 3% | 100% to 0% |
| Keyword refusal rate, base to added | 0% to 99% | 0% to 100% |
| Induced rise in logit refusal score, all identities | about 14 units | about 8.6 units |
| Republican minus Democrat in the added score, 48 pairs | -0.04, p = 0.29 | +0.05, p = 0.03 |
| Same as share of the induced rise | 0.3% | 0.6% |

The direction induces refusal for every identity, including fan. The party
difference in how much it induces is under 1% of the shift in both models.
The paired test with 48 cells is powerful enough to call Llama's 0.6%
significant, and Qwen's at coeff 2 (0.5%, p = 0.006); the pre-registered
bar was "under 10% of the rise and p > 0.05", so B passes on magnitude and
fails on the p-value in Llama. Read: symmetric to within noise that is
detectable only because the design is paired. At baseline, Republican askers
sit higher on the logit refusal score than Democrats on the same benign
questions (+0.36 Qwen, +0.45 Llama, the label effect from the geometry).

## C. Non-political identity control: prediction failed in Qwen

Each identity against voter, matched on (election, template). Guard judge;
Claude for the fan and voter rows is in section G.

| | Qwen: did task, base | Qwen: projection, benign | Llama: did task, base | Llama: projection, benign |
|---|---|---|---|---|
| fan minus voter | **-11 pts, p = 0.008** | +3.2, p < 0.001 | -4 pts, p = 0.21 | +0.57, p < 0.001 |
| Democrat minus voter | -10 pts, p = 0.002 | +2.2, p < 0.001 | +2 pts, n.s. | +0.15, p < 0.001 |
| Republican minus voter | -10 pts, p = 0.01 | +3.0, p < 0.001 | -2 pts, n.s. | +0.26, p < 0.001 |

- In Qwen, "I'm a Lakers fan" lowers compliance by as much as either party
  label does, and on benign prompts it sits as high on the refusal direction
  as the parties do. So the identity effects on projection, and Qwen's
  identity effects on compliance, are **"stated any group membership"
  effects, not party effects.** Prediction C (fan within 5 points of voter)
  fails for Qwen.
- In Llama the behavioral identity effects are all within noise; the
  projection effects are small and present for every identity.
- The alignment effect is untouched: it is a within-party comparison, and
  fan has no alignment. It is the one identity finding that is specifically
  about party.

## D. The alignment direction (`alignment_dir.py`)

a = mean over stolen cells of (crossed minus aligned activations) minus the
same over benign cells, at the chosen position: the model's representation
of "this party is asking about the other side's claim", net of the party
label. Split-half cosine of a is 0.98 in both models, so it is a stable
direction, not noise.

| | Qwen (layer 19) | Llama (layer 13) |
|---|---|---|
| cos(a, refusal direction) at the chosen layer | **0.60**, sign-flip null 95% up to 0.53, p < 0.001 | 0.21, null up to 0.29, p = 0.40 |
| cos by depth | near 0 through layer 15, then 0.44 at 16 rising to 0.79 at 25 | 0.2-0.3 everywhere |
| norm of a relative to the refusal direction | 0.07 | 0.21 |
| a's component along refusal (units of projection) | 2.0 | 0.19 |

Causal test, Guard judge, Democrat/Republican stolen prompts
("did the task", aligned / crossed):

| Condition | Qwen | Llama |
|---|---|---|
| base | 40% / 26% | 18% / 10% |
| **shift**: aligned get +a, crossed get -a, chosen layer | **28% / 39%** (the two groups swap) | 17% / 19% (crossed rises, aligned does not fall) |
| a projected out everywhere | 82% / 82% | 15% / 13% |
| random same-norm direction projected out | 39% / 30% | 15% / 13% |
| benign prompts, any condition | 0% | 0% |

Cross-fitted per-prompt projection on a, in the regression with the
refusal projection (Guard, base):

| | Qwen | Llama |
|---|---|---|
| refusal projection alone, AUC | 0.77 | 0.86 |
| alignment projection alone, AUC | 0.79 | 0.60 |
| both: refusal coef / alignment coef | +0.65 (p = 0.33) / -1.85 (p = 0.01) | -1.73 (p < 1e-7) / +0.24 (p = 0.43) |
| crossed term, with refusal only to with both | -0.56 (p = 0.01) to -0.36 (p = 0.11) | -0.65 (p = 0.02) to -1.21 (p < 0.001) |

**Qwen.** The alignment direction is real, overlaps the refusal direction
(cos 0.6, growing with depth), and is causally sufficient: shifting each
prompt by a (3.4 units, 7% of the refusal direction's norm) swaps the two
groups' compliance rates. Yet shifting along the refusal direction alone by
1.56 units (A) did nothing, and a's own refusal component is only 2.0
units. So the active part of a is its component orthogonal to the refusal
direction. The cross-fitted projection on a predicts compliance better than
the refusal projection and absorbs it. Projecting a out everywhere lifts
both groups to the ablation ceiling, because removing a unit direction with
cos 0.6 to refusal removes most of the refusal signal; that condition is
not informative about alignment and the random control confirms only that
an arbitrary direction does nothing.

**Llama.** The alignment direction is nearly orthogonal to the refusal
direction and its projection adds nothing to the refusal projection as a
predictor. Causally it half-works: subtracting it from crossed prompts
brings them up to the aligned rate, adding it to aligned prompts does not
bring them down. In Llama the alignment gap lives on the refusal direction
itself (A), and a is a weaker, partly redundant handle on it.

## E. Reproducibility of the behavioral results

The pod regenerated every reply with the same greedy decoding on a
different GPU than the Oct 25 run.

| | Qwen | Llama |
|---|---|---|
| Replies byte-identical to Oct 25 | 29% | 60% |
| Same first 40 characters | 91% | 92% |
| Guard label agreement, old vs new | 90-96% | 97-99% |
| Aligned / crossed, base, Guard, old to new | 40 / 25 to 40 / 26 | 18 / 10 to 18 / 10 |
| Republican / Democrat, base, Guard, old to new | **37 / 29 to 33 / 32** | 13 / 15 to 13 / 16 |

Greedy decoding is not bit-reproducible across hardware, but group rates
are. The alignment effect reproduces to the point. Qwen's Republican-vs-
Democrat gap (p = 0.06 on Oct 25) does not reproduce under the same judge,
which settles it as noise, on top of its disappearance under the Claude
judge.

## F. Pre-registration verdicts

| Prediction | Verdict |
|---|---|
| A1 Qwen: projection matching closes < 25% of the alignment gap | **Held.** 0% under both judges. |
| A1 Llama: (amended) about 40% closed under Claude | **Exceeded.** 220% under Claude, 150% under Guard; random control 0%. The gap runs through the direction. |
| A2: party matching moves Democrats toward refusal by < 5 points | Qwen: moved the other way, n.s. Llama: moved toward refusal by 13-15 points, overshooting Republicans. **Llama fails the "small" prediction in the informative direction.** |
| B: induced refusal symmetric, < 10% of rise, p > 0.05 | **Magnitude held** (< 1%); p-value bar failed in Llama at coeff 1 and Qwen at coeff 2. |
| C: fan within 5 points of voter | **Failed in Qwen** (-11, p = 0.008); held in Llama (-4, n.s.). |
| D1-D4: second judge | All held (NO_GPU_FOLLOWUPS.md section 4). |

## What this means for the headline

1. **Both models favor the asker's own side's claim, and the effect is
   robust**: two judges, two generation runs, within-pair.
2. **Where it lives differs by model.** In Llama it is the refusal
   direction: a tiny push along it flips aligned prompts to crossed
   behavior. In Qwen the refusal direction is inert for this, and the gap
   is carried by a distinct direction that overlaps refusal by cos 0.6 and
   whose orthogonal part does the work.
3. **The party-label effects were never about party.** A Lakers fan gets
   the same treatment as a Democrat or a Republican in Qwen, in projection
   and in behavior. Qwen's apparent Republican advantage was judge tilt plus
   generation noise.
4. **The refusal direction treats both parties alike** when added: it
   induces refusal for everyone, to within 1%.

## Limits and tomorrow's one GPU item

- The shift test in D has no random-direction control of its own; the
  closest control is the random shift in A (1.56 units in Qwen, 0.14 in
  Llama), which did nothing. A same-norm random shift at 3.4 units in Qwen
  should be run.
- The decomposition the Qwen result implies, shifting by a's component
  orthogonal to refusal vs its parallel component, has not been run. It is
  one script change and ten minutes on a pod, and it would turn "inferred"
  into "shown".
- Guard 2 was skipped (no access); it flags nothing in this task.
- Section G below depends on the Claude judge for the fan rows and the
  alignment-direction replies.

## G. Claude judge on the alignment-direction replies and the fan control

PENDING
