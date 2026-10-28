# Causal tests: results against the pre-registration (Oct 27-28)

GPU run on one pod (Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, about
50 minutes plus 25 for the alignment direction), under $5. Every reply was
regenerated, so the fan identity is in all files. Judged with Llama Guard 3
+ election category (Guard) on the pod and with the Claude judge (Claude,
`scripts/judge_claude.py`) on a Mac. Both judges are validated against a
blind human pass on 50 7B/8B replies: Claude kappa 0.77, Guard 0.51
(`validation/RESULTS.md`). Predictions and pass bars are in
`PREREG_causal_tests.md`; the verdict table is at the end.

Scripts: `patch_analyze.py`, `identity_control.py`, `add_party.py`'s
`add_party_test.csv`, `alignment_dir.py`. Summary CSVs:
`patch_summary*.csv`, `identity_control*.csv`.

## A. Projection matching: Llama mostly through the direction, Qwen mostly not

**Credit and provenance.** Divya ran projection matching on Oct 25-26
(PR #2, branch `blind-judge`), with three controls that the pre-registered
version did not have: setting each prompt to its *own* group's mean (which
leaves the gap untouched), a constant shift by the group-mean gap (which
keeps each prompt's place in the spread), and a dose series (3x and 10x
the gap, 10x also along a random direction). The pod run on Oct 27-28
reproduced her second run byte for byte (same prompts, code and GPU
class; see E), so the "set to the target mean" numbers below are the same
in both runs. The constant shift is the clean test; the set-to-mean
version is confounded, as the own-mean control shows.

All comparisons raise the lower-projection group (**aligned**, and
**Democrat**) along the unit refusal direction at the chosen layer.
"Did the task", aligned prompts (136), both judges:

| Intervention | Qwen Guard | Qwen Claude | Llama Guard | Llama Claude |
|---|---|---|---|---|
| Before: aligned / crossed | 40% / 26% | 61% / 45% | 18% / 10% | 34% / 27% |
| Each prompt set to the crossed mean (pre-registered test) | 42%, p = 0.75 | 61%, p = 1.0 | 6%, p = 0.002 | 18%, p < 0.001 |
| Each prompt set to its **own** group mean (control: gap unchanged) | 46%, p = 0.24 | 65%, p = 0.33 | 10%, p = 0.04 | 22%, p = 0.005 |
| **Constant shift by the gap, 1x (the clean test)** | **39%, p = 1.0** | **57%, p = 0.23** | **13%, p = 0.09** | **29%, p = 0.016** |
| share of the gap closed by the 1x shift | 5% | 23% | 64% | 70% |
| Constant shift 2x, with random control (Oct 28, Divya's follow-up) | | | **11%, p = 0.02**; random 16%, n.s. | **24%, p = 0.00024**; random 34%, n.s.; 130% of gap |
| Constant shift 3x | 31%, p = 0.02 | | 10%, p = 0.008 | |
| Constant shift 10x | 15%, p < 1e-7 | | 0%, p < 1e-7 | |
| 10x shift along a random direction | 41% | | 14% | |
| Democrat raised to Republican, 1x constant shift | 31% (from 32%) | 53% (from 56%) | 15% (from 16%) | 30% (from 31%) |

- **The set-to-mean test is confounded.** Setting each prompt to its own
  group's mean, which does not move the gap, drops Llama's compliance from
  18% to 10% (Guard) and 34% to 22% (Claude). So most of what "set to the
  crossed mean" measures in Llama is the removal of spread across prompts,
  not the gap. The constant shift is the number to quote.
- **Llama: the refusal direction carries the gap.** A constant shift by
  exactly the measured gap (0.14 units) closes 64-70% of it: significant
  under the Claude judge (p = 0.016), short of it under Guard (p = 0.09)
  at 136 prompts. Divya's suggested follow-up, a 2x shift with a random
  control, run Oct 28: closes 91% of the gap under Guard (18% to 11%, p = 0.02) and
  130% under Claude (34% to 24%, p = 0.00024), landing on the crossed rate,
  while the same shift along a random direction does nothing (34% Claude, 16% Guard, n.s.).
- **Qwen: the refusal direction carries little of the gap.** The 1x shift
  (1.56 units) closes 5% (Guard) to 23% (Claude), neither significant.
  The direction does control Qwen's compliance: 3x closes 63% and 10x
  overshoots, with the random direction flat. But it takes three times
  the measured gap to produce a gap-sized effect, so the gap itself mostly
  comes from somewhere else. Section D finds where.
- **Party:** there is no party gap to explain (32% vs 33% in Qwen), and
  the 1x shift changes nothing in either model under either judge.

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

## E. Reproducibility: two runs, two prompt sets

Three generation runs exist for the 7B/8B models: Divya's first (Oct 25,
736 prompts), Divya's second (Oct 25-26, 920 prompts with the fan
identity, in PR #2), and the Oct 27-28 pod run (920 prompts, this
branch). The second and third are **byte-identical** in every reply,
judge label and projection: same prompts, code and GPU class give the same
greedy output. The first differs from both:

| First run vs rerun | Qwen | Llama |
|---|---|---|
| Replies byte-identical | 29% | 60% |
| Same first 40 characters | 91% | 92% |
| Guard label agreement | 90-96% | 97-99% |
| Aligned / crossed, base, Guard | 40 / 25 to 40 / 26 | 18 / 10 to 18 / 10 |
| Republican / Democrat, base, Guard | **37 / 29 to 33 / 32** | 13 / 15 to 13 / 16 |

The difference between the first run and the rerun is the prompt set
(736 vs 920 changes the batch composition and so the padding and numerics),
not the hardware. Group rates agree within 4 points; the alignment effect
reproduces to the point; Qwen's Republican-vs-Democrat gap (p = 0.06 in
the first run) does not, which with its disappearance under the Claude
judge settles it as noise. Compare groups across runs, not individual
replies.

## F. Pre-registration verdicts

| Prediction | Verdict |
|---|---|
| A1 Qwen: projection matching closes < 25% of the alignment gap | **Held.** Clean 1x shift: 5% (Guard), 23% (Claude), neither significant. |
| A1 Llama: (amended) about 40% closed under Claude | **Exceeded, with Divya's clean test.** 1x constant shift closes 70% (Claude, p = 0.016) / 64% (Guard, p = 0.09); 2x closes 91% (Guard, p = 0.02), random control flat. The pre-registered set-to-mean version overshoots (150-220%) but is confounded by spread removal (own-mean control). |
| A2: party matching moves Democrats toward refusal by < 5 points | **Held** with the clean shift: 1-3 points in both models, n.s. The set-to-mean version in Llama (16% to 3%) is the same spread artifact. |
| B: induced refusal symmetric, < 10% of rise, p > 0.05 | **Magnitude held** (< 1%); p-value bar failed in Llama at coeff 1 and Qwen at coeff 2. |
| C: fan within 5 points of voter | **Failed in both models** under Claude (-12 in each, p < 0.01); under Guard failed in Qwen (-11) and held in Llama (-4, n.s.). |
| D1-D4: second judge | All held (NO_GPU_FOLLOWUPS.md section 4). |

## What this means for the headline

1. **Both models favor the asker's own side's claim, and the effect is
   robust**: two judges, two generation runs, within-pair.
2. **Where it lives differs by model.** In Llama the refusal direction
   plausibly carries most of the gap: a constant shift by the measured gap
   closes about 70% of it, and the decomposition shows only the refusal
   component of the alignment direction moves the gap. In Qwen the same
   1x test closes a fifth at most; the gap is carried by an alignment
   direction that overlaps refusal by cos 0.6, and *both* its refusal
   component and its orthogonal component close the gap on their own (H),
   with a same-size random direction doing nothing. The refusal direction
   controls compliance in both models (dose series), but only in Llama
   does the measured gap-sized shift do gap-sized work.
3. **The party-label effects were never about party.** A Lakers fan is
   helped 12 points less than a plain voter in both models, as much as or
   more than a party label, and sits as high on the refusal direction.
   Qwen's apparent Republican advantage was judge tilt plus generation
   noise.
4. **The refusal direction treats both parties alike** when added: it
   induces refusal for everyone, to within 1%.

## Limits and tomorrow's one GPU item

- The shift test in D has no random-direction control of its own; the
  closest controls are the random shifts in A (1.56 units and 15.6 units
  in Qwen, 0.14 and 1.4 in Llama), all flat. `alignment_decompose.py`
  adds a same-norm random control.
- The decomposition (H) uses a bidirectional shift, so its dose is about
  2.6x the measured gap; a 1x version of the par/perp split would match
  Divya's clean test exactly.
- Guard 2 was skipped (no access); it flags nothing in this task.

## G. Claude judge on the alignment-direction replies and the fan control

Opus-vs-Sonnet did-the-task agreement on the rechecks: 142 of 146 on the
alignment replies, 68 of 72 on the fan/voter rows.

**Alignment-direction shift, Claude judge** (aligned / crossed, did the
task; McNemar for shift vs base within group):

| | base | shift (+a to aligned, -a to crossed) | a projected out | random projected out |
|---|---|---|---|---|
| Qwen | 60% / 47% | **52% / 52%** (aligned down, p = 0.02; crossed up, p = 0.23) | 93% / 90% | 60% / 50% |
| Llama | 34% / 27% | **35% / 36%** (aligned flat; crossed up, p < 0.001) | 33% / 28% | 33% / 25% |

Under Claude the Qwen shift closes the 13-point gap to zero rather than
reversing it as under Guard, and the Llama shift closes the 7-point gap by
raising crossed prompts only. Both judges agree on the shape: in Qwen the
alignment direction moves both groups, in Llama it moves crossed prompts
up and leaves aligned prompts alone. Benign prompts stay at 98-100% under
every condition, so the shift is not a general refusal or compliance knob.

**Fan vs voter, Claude judge** (new replies, matched pairs):

| | Qwen | Llama |
|---|---|---|
| fan minus voter, did the task, base | **-12 pts, p = 0.002** | **-12 pts, p < 0.001** |
| fan minus voter, ablated | -7 pts, p = 0.006 | -3 pts, n.s. |
| benign, any condition | 0 | 0 |

The Claude judge confirms Qwen's fan penalty and finds the same size in
Llama, where Guard had it at -4 and n.s. For comparison, under Claude on
the Oct 25 replies, Democrat and Republican askers sat 7-8 points below
voter in Qwen and within 3 points of voter in Llama. So "I'm a Lakers fan"
lowers compliance at least as much as a party label in both models, and in
Llama more. Prediction C fails in both models under the better judge. The
party labels carry no penalty beyond what any stated group identity
carries; what party carries is the alignment effect.

## H. Decomposing the alignment direction (`alignment_decompose.py`)

Section D inferred that the part of Qwen's alignment direction a
orthogonal to the refusal direction does the work. This tests it: aligned
prompts get +v and crossed prompts -v at the chosen layer, for v = a's
component along the unit refusal direction (par), its orthogonal component
(perp), a itself (full), and a random direction of the same norm as a,
orthogonal to both (rand). Note the relative dose between the groups is
2|v|, so the par condition is a shift of about 2.6x the measured
aligned-crossed projection gap in both models, comparable to Divya's 3x
constant shift, not her 1x.

| | Qwen | Llama |
|---|---|---|
| norm of a / par / perp | 3.37 / 2.03 / 2.69 | 0.90 / 0.19 / 0.88 |

"Did the task", aligned / crossed, and the resulting gap, under both judges:

| Condition | Qwen Guard | gap | Qwen Claude | gap | Llama Guard | gap | Llama Claude | gap |
|---|---|---|---|---|---|---|---|---|
| base | 40% / 26% | +14 | 62% / 49% | +13 | 18% / 10% | +8 | 33% / 26% | +7 |
| par: refusal component only | 35% / 34% | **+1** | 56% / 54% | **+2** | 13% / 17% | **-4** | 26% / 30% | **-4** |
| perp: orthogonal component only | 35% / 36% | **-1** | 57% / 51% | +7 | 27% / 14% | +13 | 42% / 26% | +16 |
| full: a | 28% / 39% | -11 | 54% / 51% | +2 | 17% / 19% | -2 | 35% / 35% | -1 |
| rand: same norm, random | 40% / 27% | +13 | 60% / 48% | +13 | 15% / 15% | +1 | 32% / 29% | +2 |

Opus-vs-Sonnet did-the-task agreement on the rechecks: 134 of 136.

- **Qwen: the refusal component closes the gap; the orthogonal component
  does part of it.** The refusal component alone closes the gap under
  both judges (at a 2.6x dose, consistent with Divya's 3x shift closing
  63%). The orthogonal component alone closes it under Guard and about
  half of it under Claude (13 to 7, not significant on its own), while a
  random direction of the same size does nothing under either judge, so it
  is a real direction with a real but smaller effect. The section-D
  inference that the orthogonal part "does the work" was too strong: at
  this dose the refusal component does the work, and the orthogonal part
  adds to it.
- **Llama: only the refusal component carries the gap.** The orthogonal
  component raises compliance in both groups whichever way it is applied
  (aligned 18 to 27, crossed 10 to 14), which is an off-direction
  perturbation, not alignment. The gap lives on the refusal direction, as
  A found.
- **Model split, restated.** Llama encodes "the other side's claim" as
  more refusal-like, full stop. Qwen encodes it as more refusal-like *and*
  along a second direction that the refusal direction does not see; the
  refusal-like part moves behavior at this dose, the second part moves it
  less. Why Divya's 1x refusal shift does so little in Qwen while the
  alignment direction's refusal component does so much is the dose: ±2.03
  units between the groups is 2.6x the measured 1.56-unit gap.
