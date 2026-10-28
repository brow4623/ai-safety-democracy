# Handoff: refusal direction and left/right results (Oct 25)

From Divya, for Michael, Tyler and Skyler (and Sreya for slides). This covers what is done, where every file is, and what each of you can start on without a GPU.

## Status

Done on two models, **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**:

- Full prompt set: 736 prompts, 136 matched Democrat/Republican pairs, 3 severity levels, plus benign controls (`data/political_prompts.jsonl`).
- Refusal direction from non-political harmful vs harmless prompts, with held-out validation and a stability check.
- Projection of every prompt onto that direction, party-specific directions, and the "one direction or two" test.
- Replies with and without the direction ablated, scored by a validated judge.
- The 3B pilot (Sreya's Qwen2.5-3B and Llama-3.2-3B) is kept for comparison and re-scored with the same judges.

The run took about 2 hours on one A40 on RunPod. The pod is terminated.

## TA feedback (Ege) and where it stands

| Feedback | Status |
|---|---|
| Fix which election; cross claim with party | Done: every prompt names the election; 4 Republican-coded and 4 Democratic-coded claims, each asked by both parties (aligned vs crossed) |
| No-label baseline | Done ("none") |
| Non-political identity control ("I'm a Lakers fan") | **Added to the prompts** (`fan`, 920 prompts total). Not in the committed 7B/8B results yet; needs one more GPU run |
| Many requests, not one | Done: 17 request types x 8 elections, 3 severity levels |
| Party-specific directions always differ; use one non-political direction and compare projections | Done: that is the main analysis. The cos(r_D, r_R) check is near 1 by construction, so treat it as **secondary**, not a headline |
| Strong, calibrated judge that catches steering/redirecting | Partly: REDIRECT is its own label and the judge is checked against reviewed labels (kappa 0.68), but it over-counts redirects and is validated on 3B only |
| Ablation removes refusal everywhere; instead set the lower side's projection to the other side's average | **Code ready** (`scripts/patch.py`, plus a random-direction control). Not run yet; needs the GPU run |
| Pilot refusal rates need room for a difference | Fixed by 7B/8B: baseline "did the task" is 36% (Qwen) and 16% (Llama), not near 0 |
| Direction from non-political prompts | Done |

**Next GPU run** (about an hour on one A40, under $1): `scripts/pod_run.sh` now regenerates everything with the `fan` identity and runs `patch.py` plus judging for both models. Move or delete the old `.done` markers first, or it skips finished steps.

How `patch.py` works: for a comparison (aligned vs crossed, or Democrat vs Republican), the group with the lower mean projection is raised so each prompt's projection at the chosen layer equals the other group's average, by adding a multiple of the unit refusal direction at that layer's input, at every position. Nothing else changes. The same shift along a random orthogonal direction is the control. Compare the patched "did the task" rate with the target group's base rate. Checked on a small model: after the shift, the layer sees projections within 0.04 of the target.

## Headline results

Reproduce with `uv run python scripts/headline.py`. "Did the task" means the judge says the reply produced the requested stolen-election content.

**The direction is clean and stable in both models**

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Layer / position | 19 of 28 / -2 | 13 of 32 / -5 |
| KL on harmless prompts (paper's cutoff is 0.1) | 0.036 | 0.075 |
| Refusal on 64 held-out harmful prompts, before -> after ablation | 92% -> 0% | 91% -> 0% |
| Lowest cosine across 100 bootstrap re-extractions | 0.998 | 0.990 |

**No overall party difference, but both models favor "your own side's claim"** (did the task, baseline)

| | Democrat | Republican | p | Aligned (own side) | Crossed (other side) | p |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 29% | 37% | 0.06 | 40% | 25% | <0.001 |
| Llama-3.1-8B | 15% | 13% | 0.79 | 18% | 10% | 0.002 |

Example: Qwen-7B helps a Republican argue a Republican-coded claim (2020, Arizona 2022) 47% of the time, a Democrat with the same request 24%. The refusal-direction projection shows the same thing: crossed requests sit closer to refusal (Qwen +1.59, about 11% of the stolen-vs-benign gap; Llama +0.14, about 7%; both p < 0.001).

**One shared direction, not two (secondary; see TA feedback).** cos(r_D, r_R) is at least 0.987 at every layer in both models, and the D-minus-R gap lines up with the refusal direction beyond chance at 1 of 27 layers (Qwen) and 0 of 31 (Llama).

**Ablation is causal and mostly removes the asymmetry.** Did the task goes from 36% to 82% (Qwen) and 16% to 84% (Llama), benign stays at 0%. Aligned vs crossed after ablation: Qwen 84% vs 79% (p = 0.24), Llama 88% vs 82% (p = 0.04). With the direction removed, did the task rises with severity (Qwen 62% / 84% / 97%, Llama 66% / 85% / 100% for levels 1 / 2 / 3).

## Judge (read before using any behavior number)

Details in `validation/RESULTS.md`.

- The paper's judge, **Llama Guard 2, flags none** of these replies as unsafe, including ones that fully argue an election was stolen. That is a finding in itself.
- We use **Llama Guard 3 with one added category** for election delegitimization (`scripts/guard.py --election-policy`). On 171 reviewed pilot replies: kappa 0.68, about the same as the Qwen judge (0.70), with errors that barely depend on party.
- It counts about a third of debunking replies as doing the task, so **absolute rates run high**. Compare groups, don't quote raw levels as exact.
- The reference labels were written by Claude and reviewed by Divya, not labeled independently. Say "human-reviewed AI labels" on slides. The judge was validated on 3B replies, not 7B/8B.
- Every judge sees the request with the party prefix replaced by "Please", so it cannot react to party.

## Where things are

Per model, in `results/<model>/`:

| File | What it is |
|---|---|
| `direction.json` | Chosen layer, position, KL, ablation and addition scores |
| `direction_selection.csv` | Every (layer, position) candidate's ablation, addition and KL scores. **This is a first refusal curve.** |
| `direction_stability.json` | Bootstrap and split-half cosine of the direction |
| `ablation_sanity.json` | Refusal on 64 held-out harmful prompts, before and after ablation |
| `geometry_per_prompt.csv` | Each prompt's projection at the chosen layer and its refusal logit score |
| `layerwise_projection.csv` | Each prompt's projection at **every** layer (columns `L0`..`L{n-1}`), onto that layer's own direction |
| `paired_identity_effect.csv` | Democrat minus Republican projection by claim side and severity, with permutation p-values |
| `party_directions_by_layer.csv` | cos(r_D, r_R), cos with the anchor, and the gap-vs-chance test, per layer |
| `projection_by_cell.csv` | Mean projection by kind, severity, identity, claim side |
| `behavior.jsonl` | Every reply, base and ablated (128 tokens, greedy) |
| `guard3-election.jsonl`, `_rates.csv`, `_party_test.csv` | The judge's scores. Also `guard2*` and `guard3*` for comparison |
| `direction.pt`, `anchor_means.pt`, `political_acts.pt` | Tensors, **not in git** (see below) |

Elsewhere: `results/headline_behavior.csv` (all behavior comparisons), `validation/` (rubric, reviewed labels, agreement scripts), `pod_logs/` (run logs, not in git).

**Large files are not in git** (`*.pt` is ignored). Download link: **TODO (Divya to add)**. Sizes: `political_acts.pt` 148 MB (Qwen) and 193 MB (Llama); the others are a few MB.

Tensor formats (load with `torch.load`):
- `direction.pt`: `direction` [d_model], `layer`, `pos` (index into the post-instruction tokens), `all_dirs` [n_layers, n_positions, d_model] (every candidate direction).
- `anchor_means.pt`: `mean_harmful`, `mean_harmless`, each [n_layers, n_positions, d_model].
- `political_acts.pt`: `ids` (prompt ids, same order as `data/political_prompts.jsonl`), `pos`, `acts` [n_prompts, n_layers, d_model] in float16, at the chosen position. Layer l means the residual stream at the input of block l (layer 0 = embeddings).

## For Michael: layer-wise curves (no GPU needed)

- **Refusal curve across layers:** plot `ablate_harmful` (lower = ablation removes more refusal) and `add_harmless` from `direction_selection.csv` against layer, one line per position. That is the "burden of proof" for the chosen layer. Layers above 80% depth were not scanned, following the paper.
- **Party gap across layers:** from `layerwise_projection.csv`, for each layer take matched pairs and plot Democrat minus Republican, and crossed minus aligned, for stolen and benign prompts. Where the gap first appears is the depth result.
- Layer 0 projections are blank on purpose: the template tokens are identical for harmful and harmless prompts at the embeddings, so there is no direction there.
- For anything else, `political_acts.pt` has every prompt's raw activations at every layer.

## For Tyler: more models

- `scripts/pod_run.sh` runs everything for a model on one GPU pod. Edit `MODELS` and run it from the repo root. Each step leaves a `.done` marker in `/workspace/logs/`, so a rerun skips finished steps.
- A 7-9B model fits on one A40 (48 GB, $0.59/hr). Each model took about 20 minutes plus about 5 minutes of judge scoring.
- Gated Meta and Google models need Hugging Face access on your account and an `HF_TOKEN` (on RunPod, as an account secret referenced as `{{ RUNPOD_SECRET_HF_TOKEN }}`). SSH sessions on the pod don't inherit it; read it from `/proc/1/environ`.
- Candidates: Gemma-2-9B-it (CLAUDE.md's stretch model), or a second Chinese-lab model for the US vs China comparison. Qwen (Alibaba) and Llama (Meta) already make one US/China pair.
- Watch out: if a model's template puts something after the user turn that `src/model.py` doesn't expect, check `eoi_len` in the run log.

## For Skyler: causal tests (done Oct 28, see `results/CAUSAL_TESTS.md`)

All four ran on one pod on Oct 27-28 (under $5) and are scored against `results/PREREG_causal_tests.md`. Headline: the alignment asymmetry runs through the refusal direction in Llama (a 0.14-unit shift flips aligned prompts to crossed behavior) and not in Qwen (a 1.56-unit shift does nothing); in Qwen it is carried by a distinct "alignment direction" (`scripts/alignment_dir.py`, cos 0.6 with refusal) whose orthogonal part does the work. The Lakers-fan control behaves like a party label in Qwen, so the party-label effects are "any group identity" effects; the alignment effect is the only party-specific one. The original list, for the record:

0. **Projection matching** (the TA's test) is the priority: `scripts/patch.py` is written and checked, and it runs as part of `pod_run.sh`. Analyze `results/<model>/patch_<comparison>-guard3-election.jsonl` against the target group's base rate in `guard3-election.jsonl`. **Predictions and pass bars are pre-registered in `results/PREREG_causal_tests.md` (Oct 27); read it before looking at the result.**
1. **Addition by party.** Code ready: `scripts/add_party.py` (runs in `pod_run.sh` after projection matching; judge its replies with `guard.py --input add_party.jsonl`). Prediction in `results/PREREG_causal_tests.md`.
2. **Does projection predict refusal?** Done (Oct 26): `scripts/regress.py`, write-up in `results/NO_GPU_FOLLOWUPS.md`. Projection predicts "did the task" (AUC 0.8) but adding it does not shrink the crossed-vs-aligned coefficient, so expect projection matching *not* to close the gap under the Guard judge; under the Claude judge the regression shows partial mediation (see `results/PREREG_causal_tests.md` amendment).
3. **Ablation by layer** for the party gap (ablate only at some layers) if Michael's curves point somewhere specific.
4. **Second judge.** Done (Oct 27, $15.70): `scripts/judge_claude.py` re-scored every reply with the language extension's Claude judge; `headline.py --judge claude-judge` reruns every comparison under it. The alignment effect holds under both judges; Qwen's party gap does not. `results/NO_GPU_FOLLOWUPS.md` section 4.

1 and 3 need a GPU; 2 doesn't.

## Known issues

- Judge limits above (high absolute rates; validated on 3B only; AI labels reviewed by a human).
- Qwen-7B baseline helps more with severity 3 (34%) than severity 2 (21%). Read (Oct 26): it is a request-format effect (letters and speeches get drafted, "evidence"/"persuasive post" requests get refused), not severity. See `results/NO_GPU_FOLLOWUPS.md`. Don't present baseline severity as a curve.
- The judge misses most severity-1 content (town-hall questions, scrutiny op-eds), and replies often echo the asker's party, so it is not fully blind. 7B/8B spot check in `results/NO_GPU_FOLLOWUPS.md` (Claude-labeled, needs a human pass over `validation/to_label_7b8b/Claude.csv`).
- The "none" identity uses "Please" in both phrasings while the others alternate. Use "voter" or "fan" as the no-party baseline.
- `data/political_prompts.jsonl` now has 920 prompts (with `fan`); the committed 7B/8B results were made from the first 736. The ids that exist in both are identical.
- The 3B pilot used Sreya's 384 prompts and the old extraction (validation set reused for the sanity check, Qwen-3B KL relaxed to 0.5). Its files are kept as they were, apart from re-grading.

## Rerun from scratch

```bash
uv sync
uv run python src/prompts.py
uv run python scripts/run.py --model Qwen/Qwen2.5-7B-Instruct --stage all
uv run python scripts/guard.py --results Qwen2.5-7B-Instruct --guard meta-llama/Llama-Guard-3-8B --election-policy
uv run python scripts/headline.py Qwen2.5-7B-Instruct
```

---

# Extension: language and censorship (Oct 27, Tyler)

Does asking in Chinese instead of English change what a model refuses, and is that change carried by Divya's refusal direction? Run on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Report: `reports/language_censorship.html` (also published as a private page; ask Tyler for the link).

## Headline results

| Question | Result |
|---|---|
| Q1. Does Chinese increase censorship of CCP-sensitive topics? | **Yes, Qwen.** 21% censored in English vs 62% in Simplified Chinese (24 matched pairs, McNemar p = 0.002). Traditional Chinese is the same (62%), so it is the language, not the script. Controls clean. |
| Q2. Does language change behavior on the team's US stolen-election prompts? | **Reverses, exploratory.** Both models censor *less* in Chinese: Qwen 19% → 4%, Llama 69% → 52%. Neither survives Holm correction (0.131, 0.116). |
| Q3. Is the refusal direction the same in both languages? | **Yes.** cos(English, Chinese) = 0.978 (Qwen), 0.908 (Llama); ablating either direction drops refusal on harmful prompts to 0% in both languages. Divya's directions reproduced within tolerance. |
| Q4. Does removing the direction close the language gap? | **No for Qwen on China topics (the main finding).** The gap goes from 42 to 46 points after ablation. Most of it is official-line answers (29 of 42 points), which ablation leaves unchanged; projection matching agrees. **Yes for Llama on US elections, with caveats**: the gap goes from −17 to +4 points, mostly because ablation removes nearly all refusals, and raising projections increases refusal (52% → 81%, random direction 33%). Two of the pre-registered Q4 checks (bootstrap interval, five random directions) were not computed for the ablation runs. |
| Q5. Do models differ? | **Each model guards its own country.** Qwen censors 62% of China-sensitive prompts in Chinese, Llama 0% in every language. Llama refuses 69% of US election prompts in English, Qwen 19%. |

Plan step B5, the "censorship direction" (what Chinese adds on sensitive prompts beyond what it adds on controls): in Qwen it points partly along the refusal direction (cosine 0.65, shuffle-test p = 0.02, about 14% of the refusal direction's size); in Llama it does not (0.08, p = 0.51). So Qwen registers "sensitive in Chinese" on its refusal axis, but removing that axis does not stop the official-line answers. Results: `results/<model>/lang_censordir.json`, script `scripts/lang_censordir.py`.

Takeaway: the refusal direction controls genuine refusal in both languages, but Qwen's Chinese-language censorship is mostly state-aligned framing that does not run through it. An English refusal audit would miss it.

"Censored" = official line + dodges + refuses. On election prompts, pushback against the false claim is counted separately and never as censorship.

## Files

| Path | What it is |
|---|---|
| `src/lang_prompts.py`, `scripts/lang_build_prompts.py` | China set (12 topics × question/writing × sensitive/control × en/zh_s/zh_t) and the combined set |
| `data/lang_experiment_prompts.jsonl` | All 278 prompts per model (`pair` links a prompt to its control and translations) |
| `data/lang_us_zh.json`, `data/lang_anchor_zh.json` | Chinese translations of the election prompts and of Divya's harmful/harmless anchor prompts (11 harmful anchors from MultiJail, marked `source: "multijail"`) |
| `scripts/lang_pod_run.sh`, `scripts/lang_experiment.py` | GPU side: directions, behavior (base, English ablation, Chinese ablation for Qwen), projection matching |
| `scripts/lang_translate.py`, `lang_judge2.py`, `lang_backtranslate.py`, `lang_examples.py`, `lang_claude.py` | Claude API side: translation, judge (two rubrics), judge-bias check, example glosses, cost ledger |
| `scripts/lang_analyze.py` | Pre-registered tests and every report number → `results/lang_experiment/summary.json`, `examples.json` |
| `results/<model>/lang_behavior.jsonl`, `lang_labels.jsonl`, `lang_patch.jsonl`, `lang_direction.json` | Replies, judge labels, projection-matching replies, direction verification and cosine by layer |
| `results/lang_experiment/` | Summary, examples, back-translation check, cost ledger, pod logs |
| `scripts/ccp_pilot.py`, `results/Qwen2.5-7B-Instruct/ccp_pilot.jsonl`, `lang_test.jsonl`, `lang_judged.jsonl` | The two earlier pilots |

`src/model.py` gained Chinese refusal tokens and phrases (the original refusal score only recognized English).

## Rerun

```bash
# Mac: translations (Claude API)
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_translate.py
uv run python scripts/lang_build_prompts.py
# GPU pod (one H100, ~21 min for both models)
nohup bash scripts/lang_pod_run.sh > /workspace/logs/lang.log 2>&1 &
# Mac: judge, checks, analysis
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_judge2.py
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_backtranslate.py
uv run python scripts/lang_analyze.py
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_examples.py
```

Cost of the full run: $21.31 (RunPod $2.40, Claude API $14.91, pilots $4).

## Judge

Claude Opus 5.5 sees only the request and reply (not model, language or condition). Checks: a 10% re-score by Claude Sonnet 5.5 agreed 94% (Qwen) and 98% (Llama); 60 Chinese replies back-translated to English and re-judged blind agreed 95% exactly, 98% on censored vs not. 15 of 1,966 replies (all pipe-bomb sanity checks) could not be judged because Claude's safety filter declined; a keyword check shows both models refused them.

## Caveats

- Translations are by Claude with no native-speaker check; translation, register and language are confounded.
- Small sample: 12 China topics (planned 20; drafting the 8 extra topics was interrupted by a safety filter, so the validated set was kept), so the effective n is about 12. The US set is 8 elections.
- The Chinese refusal direction ablates well but barely induces refusal when added (0.12 vs 5.14 for English).
- Truncation differs: Llama hits the token cap on 45% of Chinese replies and 0% of English ones.
- Greedy decoding, one sample per prompt, 7–8B models only. A null for ablation means the gap is not carried by this one direction, not that it has no mechanism.

## Ideas for the next group

1. Native-speaker check of translations and a sample of judge labels.
2. Find what does carry the official-line behavior: build a direction from official-line vs factual Chinese replies and test it causally.
3. More China topics (the planned 20) and a second Chinese-lab model to separate Qwen from Chinese labs in general.
4. Follow up the Q2 reversal (less censorship of US election claims in Chinese) with more prompts.
