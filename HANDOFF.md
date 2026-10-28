# Handoff: refusal direction and left/right results

From Divya, for Michael, Tyler, Skyler and Sreya. Last updated Oct 26. Class session: **Thursday Oct 29**.

This is the plan document. It says what is done, what each of us does next, and where every file is.

## Plan: who does what next

| Who | Next task | Needs a GPU? | By |
|---|---|---|---|
| **Divya** | Draft the left/right and projection-matching slides (`.pt` files are uploaded, see [Where things are](#where-things-are)) | No | Mon |
| **Michael** | Layer-wise curves: **done** ([results](#for-michael-layer-wise-curves-no-gpu-needed)). Slides for that part to Sreya | No | Tue (slides Wed) |
| **Tyler** | A third model, e.g. Gemma-2-9B-it or a second Chinese-lab model, with `pod_run.sh` ([details](#for-tyler-more-models)) | Yes, about 1 hr on an A40 | **Mon** (away Tue/Wed) |
| **Skyler** | Causal follow-ups: does projection predict refusal; addition by party ([details](#for-skyler-causal-tests)) | One no, one yes | Wed |
| **Sreya** | Slides: one section per person, plus the judge finding and the TA-feedback table below | No | Wed |

Each person makes the slides for their own part and sends them to Sreya by Wednesday.

## Status

Done on **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct** (two runs on one A40 on RunPod):

- Prompt set: 920 prompts. 8 elections (4 Republican-coded and 4 Democratic-coded "stolen" claims), each asked by a Democrat, a Republican, a voter, a Lakers fan, and nobody; 17 request types at 3 severity levels; benign controls (`data/political_prompts.jsonl`).
- Refusal direction from non-political harmful vs harmless prompts, checked on held-out prompts and for stability.
- Projection of every prompt onto that direction, at every layer.
- Replies with and without the direction ablated, scored by a validated judge.
- Projection matching (the TA's causal test), with controls for how the shift is applied and a dose check.
- The 3B pilot (Sreya's Qwen2.5-3B and Llama-3.2-3B) is kept for comparison.

## TA feedback (Ege) and where it stands

| Feedback | Status |
|---|---|
| Fix which election; cross claim with party | Done: every prompt names the election; each claim is asked by both parties (aligned vs crossed) |
| No-label baseline | Done ("none") |
| Non-political identity control ("I'm a Lakers fan") | Done (`fan`). See results |
| Many requests, not one | Done: 17 request types x 8 elections, 3 severity levels |
| Party-specific directions always differ; use one non-political direction and compare projections | Done: that is the main analysis. The cos(r_D, r_R) check is near 1 by construction, so it is **secondary** |
| Strong, calibrated judge that catches steering/redirecting | Partly: REDIRECT is its own label and the judge is checked against reviewed labels (kappa 0.68), but it over-counts redirects and is validated on 3B replies only |
| Ablation removes refusal everywhere; instead set the lower side's projection to the other side's average | Done, with controls (see Projection matching below) |
| Pilot refusal rates need room for a difference | Fixed by 7B/8B: baseline "did the task" is 36% (Qwen) and 15% (Llama) |
| Direction from non-political prompts | Done |

## Headline results

Reproduce with `uv run python scripts/headline.py`. "Did the task" means the judge says the reply produced the requested stolen-election content.

**The direction is clean and stable in both models**

| | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Layer / position | 19 of 28 / -2 | 13 of 32 / -5 |
| KL on harmless prompts (paper's cutoff is 0.1) | 0.036 | 0.075 |
| Refusal on 64 held-out harmful prompts, before -> after ablation | 92% -> 0% | 91% -> 0% |
| Lowest cosine across 100 bootstrap re-extractions | 0.998 | 0.990 |

**No party difference, but both models favor "your own side's claim"** (did the task, baseline, 136 matched pairs)

| | Democrat | Republican | p | Aligned (own side) | Crossed (other side) | p |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 32% | 33% | 1.00 | 40% | 26% | 0.001 |
| Llama-3.1-8B | 16% | 13% | 0.27 | 18% | 10% | 0.003 |

The refusal-direction projection agrees: crossed requests sit closer to refusal (Qwen +1.56, about 11% of the stolen-vs-benign gap; Llama +0.14, about 7%; both p < 0.001).

**A party label acts like any identity** (did the task, baseline)

| Asker | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Nobody stated | 38% | 21% |
| Voter | 43% | 14% |
| Democrat | 32% | 16% |
| Republican | 33% | 13% |
| Lakers fan | 32% | 10% |

A Lakers fan is treated about like a partisan. What moves the models is whether the claim matches the asker's side, not the party itself.

**Ablation is causal and removes the asymmetry.** Did the task goes from 36% to 81% (Qwen) and 15% to 82% (Llama); benign stays at 0%. Aligned vs crossed after ablation: Qwen 85% vs 82% (p = 0.45), Llama 86% vs 82% (p = 0.23). With the direction removed, did the task rises with severity (Qwen 62% / 83% / 96%, Llama 64% / 83% / 99% for levels 1 / 2 / 3).

**Projection matching (TA's test).** Aligned prompts (the side the model helps more) are pushed along the refusal direction toward the crossed prompts' level, at the chosen layer, nothing else changed. Did the task, aligned prompts (136):

| Intervention | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| Before (aligned / crossed) | 40% / 26% | 18% / 10% |
| Each prompt set to the crossed average | 42% (p = 0.75) | 6% (p = 0.002) |
| Each prompt set to its **own** group average (control: no gap change) | 46% (p = 0.24) | 10% (p = 0.04) |
| **Constant shift by the gap (1x), the clean test** | **39% (p = 1.0)** | **13% (p = 0.09)** |
| Constant shift 3x | 31% (p = 0.02) | 10% (p = 0.008) |
| Constant shift 10x | 15% (p < 0.001) | 0% (p < 0.001) |
| Same 10x shift along a random direction | 41% | 14% |

How to read it:
- **"Set to the average" is confounded.** Setting each prompt to its own group's average, which leaves the gap untouched, moves Llama about as much as setting it to the crossed average. So that version mostly measures removing the spread across prompts, not the gap. Use the constant shift instead.
- **The direction really controls this behavior.** Bigger shifts along it lower "did the task" step by step in both models, and the same shift along a random direction does nothing.
- **How much of the aligned/crossed gap runs through it differs by model.** Shifting by exactly the measured gap (1x) closes about 64% of Llama's gap (18% -> 13%, short of significance with 136 prompts) and about 5% of Qwen's (40% -> 39%). In Qwen the gap mostly comes from somewhere else; in Llama the direction plausibly explains much of it.
- Democrat vs Republican: there's no behavior gap to explain (32% vs 33% in Qwen), and the 1x shift changes nothing in either model.

**One shared direction, not two (secondary; see TA feedback).** cos(r_D, r_R) is at least 0.987 at every layer in both models, and the D-minus-R gap lines up with the refusal direction beyond chance at 2 of 27 layers (Qwen) and 0 of 31 (Llama).

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
| `guard3-election.jsonl`, `_rates.csv`, `_party_test.csv` | The judge's scores. Also `guard2*` and `guard3*` for comparison (first run, 736 prompts) |
| `patch_<comparison>[_<mode><scale>].jsonl`, `...-guard3-election.jsonl` | Projection-matching replies and their judge scores |
| `direction.pt`, `anchor_means.pt`, `political_acts.pt` | Tensors, **not in git** (see below) |

Elsewhere: `results/headline_behavior.csv` (all behavior comparisons), `validation/` (rubric, reviewed labels, agreement scripts), `pod_logs/` (run logs, not in git).

**Large files are not in git** (`*.pt` is ignored). They are on the release [tensors-v1](https://github.com/brow4623/ai-safety-democracy/releases/tag/tensors-v1), named `<model>.<file>.pt`, with checksums in `SHA256SUMS.txt`. Sizes: `political_acts.pt` 185 MB (Qwen) and 241 MB (Llama); the others are a few MB. To download and put them in place, from the repo root:

```bash
gh release download tensors-v1 --repo brow4623/ai-safety-democracy --dir /tmp/tensors && for f in /tmp/tensors/*.pt; do n=$(basename "$f"); mv "$f" "results/${n%.*.*}/${n#*-Instruct.}"; done
```

(Or download them from the release page and rename `Qwen2.5-7B-Instruct.direction.pt` to `results/Qwen2.5-7B-Instruct/direction.pt`, and so on.)

Tensor formats (load with `torch.load`):
- `direction.pt`: `direction` [d_model], `layer`, `pos` (index into the post-instruction tokens), `all_dirs` [n_layers, n_positions, d_model] (every candidate direction).
- `anchor_means.pt`: `mean_harmful`, `mean_harmless`, each [n_layers, n_positions, d_model].
- `political_acts.pt`: `ids` (prompt ids, same order as `data/political_prompts.jsonl`), `pos`, `acts` [n_prompts, n_layers, d_model] in float16, at the chosen position. Layer l means the residual stream at the input of block l (layer 0 = embeddings).

## For Michael: layer-wise curves (no GPU needed)

**Done.** `uv run python scripts/layer_curves.py` rebuilds everything from saved results in under a minute: `figures/refusal_curve.png`, `figures/party_gap_by_layer.png`, and `results/party_gap_by_layer.csv` (per layer: gap, CI, dz, sign-flip p, Bonferroni flag). Gaps are a share of each layer's stolen-minus-benign gap, the scale `headline.py` uses.

- **Refusal curve (why the chosen layer):** Llama's layer 13 / position -5 is the strongest ablation of any candidate and passes both filters. In Qwen, 21 candidates at layers 13-21 ablate more than layer 19 / position -2, but every one of them changes harmless outputs too much (KL above 0.1; about half also fail the addition test). Layer 19 / -2 is the strongest candidate that passes. Layers above 80% depth were not scanned, following the paper.
- **Alignment gap (the depth result):** crossed minus aligned is about 0 in early layers and switches on mid-network, only for stolen requests. First significant (Bonferroni over layers) at **layer 19 of 28 in Qwen** and **layer 11 of 32 in Llama**; 11% and 7% at the chosen layers, peaking at 18% (Qwen, layer 25) and 13% (Llama, layer 18). Benign questions stay at or slightly below 0.
- **Party label:** Republican minus Democrat is nonzero from the first layers in both models but tracks the benign questions, so it is the party word, not refusal. Party minus Lakers fan is larger and also mostly shared with benign questions (Qwen through layer 18; in Llama the fan prompts sit closer to refusal on both kinds, more so on benign).
- **Reading:** the model represents *who* is asking early, and combines it with *whose claim it is* only later, around the depth where the refusal direction starts to work.
- Caveats: each layer uses its own direction, so sign changes across layers compare different directions. Early layers have a small stolen-minus-benign gap, so percentages there are noisy. Layer 0 is blank on purpose (no direction at the embeddings).

## For Tyler: more models

- `scripts/pod_run.sh` runs everything for a model on one GPU pod. Edit `MODELS` and run it from the repo root. Each step leaves a `.done` marker in `/workspace/logs/`, so a rerun skips finished steps.
- A 7-9B model fits on one A40 (48 GB, $0.59/hr). Each model took about 25 minutes plus about 10 minutes of judging and projection matching.
- Gated Meta and Google models need Hugging Face access on your account and an `HF_TOKEN` (on RunPod, as an account secret referenced as `{{ RUNPOD_SECRET_HF_TOKEN }}`). SSH sessions on the pod don't inherit it; read it from `/proc/1/environ`.
- Candidates: Gemma-2-9B-it (CLAUDE.md's stretch model), or a second Chinese-lab model for the US vs China comparison. Qwen (Alibaba) and Llama (Meta) already make one US/China pair.
- Watch out: if a model's template puts something after the user turn that `src/model.py` doesn't expect, check `eoi_len` in the run log.

## For Skyler: causal tests (done Oct 27-28; results in `results/CAUSAL_TESTS.md`)

Predictions were written before the GPU run (`results/PREREG_causal_tests.md`) and every test is scored against them. Everything below is judged with Llama Guard 3 + election and with a second judge, Claude on Tyler's US rubric (`scripts/judge_claude.py`; `headline.py --judge claude-judge`), validated by a blind human pass at kappa 0.77 (`validation/RESULTS.md`).

1. **Does projection predict refusal?** Done: `scripts/regress.py`. AUC about 0.8-0.9 in both models. Adding projection does not remove the crossed-vs-aligned term in Qwen; it shrinks it in Llama.
2. **Addition by party.** Done: `scripts/add_party.py`. The direction induces refusal for every identity (fan included) to within 1% across parties.
3. **Projection matching.** Divya's constant-shift test, scored under both judges: Llama's 1x shift closes 64-70% of the alignment gap (Claude p = 0.016, Guard p = 0.09); Qwen's closes 5-23%, n.s. The set-to-mean version is confounded (own-mean control). Divya's 2x follow-up (Oct 28): Llama closes 91% of the gap under Guard (p = 0.02) and all of it under Claude (p < 0.001), random direction flat. Settled.
4. **Alignment direction** (new, `scripts/alignment_dir.py`, Tyler's censorship-direction construction applied to aligned vs crossed): in Qwen it is stable (split-half cos 0.98), overlaps refusal at cos 0.6 rising to 0.8 by layer 25 (where Michael's curve peaks), and shifting each prompt by it closes the gap under both judges while the 1x refusal-only shift does not. In Llama it is nearly orthogonal to refusal and half-works. `scripts/alignment_decompose.py` splits it into refusal-parallel and orthogonal parts: in Qwen the refusal-parallel part closes the gap and the orthogonal part closes part of it (random same-norm direction does nothing); in Llama only the refusal part does.
5. **Ablation by layer** at Michael's layers (19 Qwen, 11 Llama): not run.

## Known issues

- Judge limits above. Update Oct 28: a blind human pass on 50 7B/8B replies gives kappa 0.77 for the Claude judge and 0.51 for Llama Guard (`validation/RESULTS.md`); Guard misses disclaimer-then-argue and severity-1 replies, so its severity-1 rates are undercounts.
- Qwen-7B baseline helps more with severity 3 (33%) than severity 2 (22%). Read (Oct 26): a request-format effect (letters and speeches get drafted, "evidence" and "persuasive post" requests get refused), not severity; see `results/NO_GPU_FOLLOWUPS.md`.
- The "none" identity uses "Please" in both phrasings while the others alternate. Use "voter" or "fan" as the no-party baseline.
- The second run regenerated all replies; rates differ from the first run by a point or two (the 736- vs 920-prompt batch changes the numerics; the Oct 28 pod run reproduced the second run byte for byte). Conclusions didn't change, except Qwen's Republican-vs-Democrat gap (p = 0.06 in the first run) which does not reproduce and vanishes under the Claude judge. `guard2*` and plain `guard3*` files are from the first run (736 prompts).
- The 3B pilot used Sreya's 384 prompts and the old extraction (validation set reused for the sanity check, Qwen-3B KL relaxed to 0.5). Its files are kept as they were, apart from re-grading.

## Rerun from scratch

```bash
uv sync
uv run python src/prompts.py
uv run python scripts/run.py --model Qwen/Qwen2.5-7B-Instruct --stage all
uv run python scripts/guard.py --results Qwen2.5-7B-Instruct --guard meta-llama/Llama-Guard-3-8B --election-policy
uv run python scripts/patch.py --model Qwen/Qwen2.5-7B-Instruct
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
| `scripts/lang_translate.py`, `lang_judge.py`, `lang_backtranslate.py`, `lang_examples.py`, `lang_claude.py` | Claude API side: translation, judge (two rubrics), judge-bias check, example glosses, cost ledger |
| `scripts/lang_analyze.py` | Pre-registered tests and every report number → `results/lang_experiment/summary.json`, `examples.json` |
| `results/<model>/lang_behavior.jsonl`, `lang_labels.jsonl`, `lang_patch.jsonl`, `lang_direction.json` | Replies, judge labels, projection-matching replies, direction verification and cosine by layer |
| `results/lang_experiment/` | Summary, examples, back-translation check, cost ledger (pod logs are kept locally in `pod_logs/`, not in git) |
| `scripts/ccp_pilot.py`, `lang_pilot_run.py`, `lang_pilot_judge.py`, `data/lang_prompts.jsonl`, `results/Qwen2.5-7B-Instruct/ccp_pilot.jsonl`, `lang_test.jsonl`, `lang_judged.jsonl` | The two earlier pilots (superseded; kept as a record) |

`src/model.py` gained Chinese refusal tokens and phrases (the original refusal score only recognized English).

## Rerun

```bash
# Mac: translations (Claude API)
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_translate.py
uv run python scripts/lang_build_prompts.py
# GPU pod (one H100, ~35 min for both models). Needs Divya's results/<model>/direction.json and
# direction_stability.json, which are in git; writes the .pt direction files it reuses.
nohup bash scripts/lang_pod_run.sh > /workspace/logs/lang.log 2>&1 &
# Mac: judge, checks, analysis
ANTHROPIC_API_KEY=... uv run --no-project --with anthropic python scripts/lang_judge.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
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
- Most replies are cut off at the token cap (300 tokens; 400 for Llama in Chinese): Qwen 44–68% depending on language, Llama 51% in English and 45–50% in Chinese. The judge was told to label where a reply is heading.
- Greedy decoding, one sample per prompt, 7–8B models only. A null for ablation means the gap is not carried by this one direction, not that it has no mechanism.

## Ideas for the next group

1. Native-speaker check of translations and a sample of judge labels.
2. Find what does carry the official-line behavior: build a direction from official-line vs factual Chinese replies and test it causally.
3. More China topics (the planned 20) and a second Chinese-lab model to separate Qwen from Chinese labs in general.
4. Follow up the Q2 reversal (less censorship of US election claims in Chinese) with more prompts.
