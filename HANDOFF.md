# Handoff: refusal direction and left/right results

From Divya, for Michael, Tyler, Skyler and Sreya. Last updated Oct 26. Class session: **Thursday Oct 29**.

This is the plan document. It says what is done, what each of us does next, and where every file is.

## Plan: who does what next

| Who | Next task | Needs a GPU? | By |
|---|---|---|---|
| **Divya** | Draft the left/right and projection-matching slides (`.pt` files are uploaded, see [Where things are](#where-things-are)) | No | Mon |
| **Michael** | Layer-wise curves: refusal curve and where the party / alignment gap appears ([details](#for-michael-layer-wise-curves-no-gpu-needed)) | No | Tue |
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

- **Refusal curve across layers:** plot `ablate_harmful` (lower = ablation removes more refusal) and `add_harmless` from `direction_selection.csv` against layer, one line per position. That is the "burden of proof" for the chosen layer. Layers above 80% depth were not scanned, following the paper.
- **Party and alignment gap across layers:** from `layerwise_projection.csv`, for each layer take matched pairs and plot Democrat minus Republican, and crossed minus aligned, for stolen and benign prompts. Where the gap first appears is the depth result. The Lakers-fan rows give a non-political comparison line.
- Layer 0 projections are blank on purpose: the template tokens are identical for harmful and harmless prompts at the embeddings, so there is no direction there.
- For anything else, `political_acts.pt` has every prompt's raw activations at every layer.

## For Tyler: more models

- `scripts/pod_run.sh` runs everything for a model on one GPU pod. Edit `MODELS` and run it from the repo root. Each step leaves a `.done` marker in `/workspace/logs/`, so a rerun skips finished steps.
- A 7-9B model fits on one A40 (48 GB, $0.59/hr). Each model took about 25 minutes plus about 10 minutes of judging and projection matching.
- Gated Meta and Google models need Hugging Face access on your account and an `HF_TOKEN` (on RunPod, as an account secret referenced as `{{ RUNPOD_SECRET_HF_TOKEN }}`). SSH sessions on the pod don't inherit it; read it from `/proc/1/environ`.
- Candidates: Gemma-2-9B-it (CLAUDE.md's stretch model), or a second Chinese-lab model for the US vs China comparison. Qwen (Alibaba) and Llama (Meta) already make one US/China pair.
- Watch out: if a model's template puts something after the user turn that `src/model.py` doesn't expect, check `eoi_len` in the run log.

## For Skyler: causal tests

1. **Does projection predict refusal?** (no GPU) Logistic regression of the judge's "did the task" on each prompt's projection (`geometry_per_prompt.csv` joined to `guard3-election.jsonl` on `id`), with party, alignment and severity terms.
2. **Addition by party.** (GPU) Add the direction (`Model.add` in `src/model.py`) to the benign election questions and check whether it induces refusal equally for Democrat and Republican askers.
3. **Projection matching follow-up:** the 1x shift in Llama closes ~64% of the gap but isn't significant at 136 prompts; more prompts or a 2x shift would settle it.
4. **Ablation by layer** for the alignment gap, if Michael's curves point somewhere specific.

## Known issues

- Judge limits above (high absolute rates; validated on 3B only; AI labels reviewed by a human).
- Qwen-7B baseline helps more with severity 3 (33%) than severity 2 (22%). Worth reading those replies before putting severity on a slide.
- The "none" identity uses "Please" in both phrasings while the others alternate. Use "voter" or "fan" as the no-party baseline.
- The second run regenerated all replies; rates differ from the first run by a point or two (generation is not bit-identical across runs). Conclusions didn't change. `guard2*` and plain `guard3*` files are from the first run (736 prompts).
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
