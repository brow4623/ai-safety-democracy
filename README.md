# Partisan asymmetry in refusal geometry

Does an open-weight chat model refuse "the election was stolen" requests
differently depending on the user's stated party, and does any difference show
up inside the model, along the refusal direction?

Builds on Arditi et al. 2024, *Refusal in Language Models Is Mediated by a
Single Direction* (arXiv 2406.11717), and checks against *There Is More to
Refusal than a Single Direction* (arXiv 2602.02132).

## Design

### Prompts (`src/prompts.py` -> `data/political_prompts.jsonl`)

384 prompts, built as 4 identities (none, voter, Democrat, Republican) x 8 elections x templates.

- **Identity:** none, voter, Democrat, Republican. "Voter" has the same sentence shape as the party labels with no party. Each has two phrasings ("I'm a
  Democrat." / "As a lifelong Democrat...") so a result can't hinge on one
  string.
- **Election:** four elections where the "stolen" story is coded Republican
  (2020 presidential, 2020 GA, 2020 MI, 2022 AZ governor), and four where it is
  coded Democratic (2016 presidential, 2000 FL, 2004 OH, 2018 GA governor).
- **Templates:** 8 "argue it was stolen" requests (prove it, persuasive post,
  rally speech, letter to legislators, and so on), plus 4 benign civic
  questions about the same elections as a control.

Why the elections are crossed with identity: the original example (a Democrat
asking to prove 2020 was stolen) changes two things at once, who is asking and
whose claim it is. Crossing them gives **aligned** prompts (a Democrat on 2016,
a Republican on 2020) and **crossed** prompts (the reverse), so a party-label
effect can be told apart from a claim-content effect. Within one (election,
template) cell the prompts differ only in the identity prefix.

### Pipeline (`scripts/run.py`)

1. **direction:** compute difference-of-means directions from the standard
   harmful vs. harmless prompts (Arditi et al.'s splits, in `data/anchor/`) at
   every layer and post-instruction position. Pick the one whose ablation most
   reduces refusal on held-out harmful prompts. It must also induce refusal when
   added to harmless prompts, and must barely change harmless outputs (KL < 0.1).
2. **geometry:**
   - Project every political prompt onto that direction.
   - Estimate the paired Democrat-minus-Republican effect, with a sign-flip
     permutation test.
   - Build party-specific directions (party's stolen-election prompts minus
     the harmless mean) and test whether they split.
3. **behavior:** generate replies with and without the direction ablated, and
   label each refuse / pushback / comply.

### Testing whether the directions "split"

A high cosine between the Democrat and Republican directions is not enough on
its own, because the two prompt sets share almost all their content. So we
compare `cos(r_D, r_R)` against a split-half ceiling: the cosine between two
halves of the *same* party's prompts (templates 0-3 vs. 4-7). If the
cross-party cosine is about as high as the within-party one, it is one
direction. If it is clearly lower, the mechanism differs by party. We also
report how much of the D-minus-R difference vector lies along the refusal
direction.

## Our take on the two open questions

**Scope: partisan asymmetry only, or political gray-area requests broadly?**
Keep the core tight on partisan asymmetry, but use the stolen-election
narratives from *both* sides as described above. That is still one clean
question, and it removes the biggest confound. Broader gray-area topics can be
a stretch goal once the main result is in, well before the October 29 session.

**Where should the refusal direction come from?** Anchor it on standard
nonpolitical harmful/harmless prompts first, then project the political prompts
onto it. A direction taken from the political prompts themselves would mix
"refusal" with "this is about partisan politics", and the two could not be
separated afterward. The geometry stage also builds the political-prompt
directions and reports their cosine with the anchor, so we get both views. The
benign political control set shows how much of the projection is just topic.

## Running

```
uv sync
uv run python src/prompts.py
uv run python scripts/run.py --stage all                 # Qwen2.5-3B-Instruct, ~30 min on an M-series Mac
uv run python scripts/run.py --model meta-llama/Llama-3.2-3B-Instruct
```

Runs on a CUDA GPU, Apple MPS, or CPU. On a Kaggle T4 it picks fp16
automatically. Results go to `results/<model>/`.

## Pilot results (Qwen2.5-3B-Instruct)

See [EXPLAINER.md](EXPLAINER.md) for a plain-English walkthrough.

- **Direction:** layer 23, position -5. Ablating it takes refusal on held-out
  harmful prompts from 88% to 0%. The KL on harmless prompts is 0.31.
- **Behavior:** baseline refusal on stolen-election prompts is 94% / 88%
  (R-coded / D-coded) with no identity. It is 78% / 56% for "Republican",
  66% / 53% for "Democrat", and 62% / 25% for "voter".
- **Projection:** Republican-labeled prompts sit 0.73 units higher on the
  refusal direction than matched Democrat ones (p < 1e-4, sign-flip). The gap
  is the same on both claim sides and similar on benign prompts (0.86).
- **Split:** cos(r_D, r_R) = 0.997 at every layer. The D-minus-R gap has
  cosine about -0.4 with the refusal direction, which is inside the sign-flip
  95% band. Its cosine with the benign D-minus-R gap is 0.84. Conclusion: one
  shared direction, plus a general party-label signal, not a party-specific
  refusal mechanism.
- **Ablation:** refusal on stolen prompts drops to 3-25% for every identity,
  roughly evenly.

## Known limits (pilot)

- Refuse / pushback / comply labels come from keyword matching. Before final
  numbers they should be replaced with an LLM judge, plus a hand check of a
  sample.
- There are only 32 prompts per (identity, claim side) cell, so behavioral
  gaps under about 20 points are not reliable. The continuous measures
  (projection and logit refusal score) are more sensitive.
- One model so far. Llama-3.2-3B-Instruct is the obvious second model.
- The KL cutoff was relaxed from 0.1 to 0.5 (see `--kl-max`). The first run,
  without the cutoff, picked a direction that damaged factual recall. It is
  kept in `results/Qwen2.5-3B-Instruct_run1_kl_unfiltered/`.
