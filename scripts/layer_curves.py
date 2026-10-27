"""Layer-wise curves: where the refusal direction works, and where the party gap appears.

From saved results only (no GPU):
  refusal curve  every scanned (layer, position) candidate in
                 direction_selection.csv: refusal score on held-out harmful
                 prompts with the direction ablated (lower = removes more
                 refusal), on harmless prompts with it added (higher = induces
                 more), and KL on harmless prompts. The chosen position is
                 highlighted, the others are gray.
  party gap      at every layer, the paired gap on that layer's own refusal
                 direction (layerwise_projection.csv): crossed minus aligned,
                 Republican minus Democrat, and (when the run has the fan
                 identity) party label minus Lakers fan, the non-political
                 control, for stolen and benign prompts. Gaps are a share of that layer's stolen-minus-benign gap (the
                 scale headline.py uses), with ~95% CIs (mean +- 1.96 SE over
                 pairs) and sign-flip p-values. Positive = closer to refusal.

Writes figures/refusal_curve.png, figures/party_gap_by_layer.png (plus
slide_alignment_gap.png and slide_party_gap.png, the same rows split for
slides) and results/party_gap_by_layer.csv, and prints the first layer where each gap
clears p < 0.05 Bonferroni-corrected over layers.

Usage: uv run python scripts/layer_curves.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from headline import comparisons, paired  # noqa: E402
from summarize import sign_flip_p  # noqa: E402

FIG = ROOT / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS, SHADE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec")
CHOSEN, OTHER = "#2a78d6", "#c3c2b7"
KIND_COLOR = {"stolen": "#2a78d6", "benign": "#eb6834"}
KIND_LABEL = {"stolen": "stolen-election requests", "benign": "benign election questions"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    })


def load_direction(model):
    return json.load(open(ROOT / "results" / model / "direction.json"))


def ref_line(ax, y, text, x):
    ax.axhline(y, color=MUTED, lw=0.8, zorder=1)
    ax.annotate(text, (x, y), xytext=(0, 2), textcoords="offset points", va="bottom",
                fontsize=8, color=MUTED)


# ------------------------------------------------------------ refusal curve

def refusal_curve(models):
    rows = [("ablate_harmful", "Harmful prompts, direction ablated\nrefusal score (lower = less refusal)"),
            ("add_harmless", "Harmless prompts, direction added\nrefusal score (higher = more refusal)"),
            ("kl_harmless", "KL on harmless prompts, ablated\n(log scale, lower = fewer side effects)")]
    fig, axes = plt.subplots(3, len(models), figsize=(5.4 * len(models), 8.2), sharey="row", squeeze=False)
    for j, model in enumerate(models):
        d = load_direction(model)
        sel = pd.read_csv(ROOT / "results" / model / "direction_selection.csv")
        layer, pos, n = int(d["layer"]), int(d["pos"]), int(d["n_layers"])
        best = sel[(sel.layer == layer) & (sel.pos == pos)].iloc[0]
        for i, (col, label) in enumerate(rows):
            ax = axes[i, j]
            for p, s in sel.groupby("pos"):
                if p != pos:
                    ax.plot(s.layer, s[col], color=OTHER, lw=0.9, zorder=2)
            s = sel[sel.pos == pos]
            ax.plot(s.layer, s[col], color=CHOSEN, lw=1.6, zorder=3)
            ax.scatter([layer], [best[col]], s=55, facecolor=SURFACE, edgecolor=CHOSEN, lw=1.6, zorder=4)
            # Layers above 80% depth were not scanned, following the paper.
            ax.axvspan(sel.layer.max() + 0.5, n - 0.5, color=SHADE, lw=0, zorder=0)
            ax.set_xlim(0.5, n - 0.5)
            if j == 0:
                ax.set_ylabel(label)
        # Reference labels go in the unscanned strip, where no curve can cover them.
        top, mid, bot = axes[:, j]
        x = sel.layer.max() + 0.8
        ref_line(top, d["baseline_harmful"], "no ablation", x)
        ref_line(mid, d["baseline_harmless"], "no addition", x)
        ref_line(mid, 0, "must be > 0", x)
        bot.set_yscale("log")
        ref_line(bot, d["kl_cutoff_used"], f"KL cutoff {d['kl_cutoff_used']:g}", x)
        top.set_title(f"{model} ({n} layers)\nchosen: layer {layer}, position {pos}", loc="left")
        top.text((sel.layer.max() + n) / 2, 0.03, "not scanned\n(> 80% depth)", transform=top.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color=MUTED)
        bot.set_xlabel("Layer")
    handles = [Line2D([], [], color=CHOSEN, lw=1.6, label="chosen post-instruction position"),
               Line2D([], [], color=OTHER, lw=0.9, label="other positions"),
               Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=CHOSEN, mew=1.6,
                      label="selected direction")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Refusal score = log-odds that the reply starts with a refusal token, averaged over "
             "validation prompts. Each candidate is that layer's harmful-minus-harmless mean difference.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(FIG / "refusal_curve.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- party gap

def pair_gaps(sub, col):
    """Matched-pair gaps: headline.py's two comparisons, plus party label vs the fan control."""
    out = []
    for name, a, b in comparisons(sub):
        x, y = paired(sub, col, a, b)
        out.append((name, (x - y).values))
    p = sub.pivot(index="pair_id", columns="identity", values=col)
    if "fan" in p:
        out.append(("party vs fan", ((p.democrat + p.republican) / 2 - p.fan).values))
    return out


def layer_gaps(model):
    df = pd.read_csv(ROOT / "results" / model / "layerwise_projection.csv")
    st = df[df.kind == "stolen"]
    cols = [c for c in df.columns if c[0] == "L" and c[1:].isdigit() and df[c].notna().any()]
    rows = []
    for c in cols:
        # Same scale as headline.py: stolen prompts (any stated identity) minus benign.
        scale = st[st.identity != "none"].groupby("pair_id")[c].mean().mean() - df[df.kind == "benign"][c].mean()
        for kind in ["stolen", "benign"]:
            for name, gap in pair_gaps(df[df.kind == kind], c):
                se = gap.std(ddof=1) / np.sqrt(len(gap))
                lo, hi = sorted([(gap.mean() - 1.96 * se) / scale, (gap.mean() + 1.96 * se) / scale])
                rows.append({"model": model, "layer": int(c[1:]), "kind": kind, "comparison": name,
                             "n_pairs": len(gap), "gap": gap.mean(), "se": se, "scale": scale,
                             "share": gap.mean() / scale, "share_lo": lo, "share_hi": hi,
                             "dz": gap.mean() / gap.std(ddof=1), "p": sign_flip_p(gap)})
    out = pd.DataFrame(rows)
    out["sig_bonferroni"] = out.p < 0.05 / len(cols)
    return out


def label_ends(ax, ends):
    """Direct-label line ends, nudged apart when they would overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for (x, _, text), y in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2, annotation_clip=False)


def party_gap(gaps, models, keep=None, out="party_gap_by_layer.png"):
    rows = [("crossed vs aligned", "Crossed minus aligned\n(% of stolen-minus-benign gap)"),
            ("Republican vs Democrat", "Republican minus Democrat\n(% of stolen-minus-benign gap)"),
            ("party vs fan", "Party label minus Lakers fan\n(% of stolen-minus-benign gap)")]
    rows = [r for r in rows if r[0] in set(gaps.comparison) and (keep is None or r[0] in keep)]
    height = 3.1 * len(rows) + 0.2
    fig, axes = plt.subplots(len(rows), len(models), figsize=(5.4 * len(models), height),
                             sharey="row", squeeze=False)
    for j, model in enumerate(models):
        d = load_direction(model)
        n = int(d["n_layers"])
        for i, (name, label) in enumerate(rows):
            ax = axes[i, j]
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
            ax.axvline(d["layer"], color=MUTED, lw=0.8, zorder=1)
            for kind, color in KIND_COLOR.items():
                g = gaps[(gaps.model == model) & (gaps.comparison == name) & (gaps.kind == kind)]
                ax.fill_between(g.layer, 100 * g.share_lo, 100 * g.share_hi, color=color, alpha=0.15, lw=0)
                ax.plot(g.layer, 100 * g.share, color=color, lw=1.6, zorder=3)
            ax.set_xlim(0.5, n - 0.5)
            if j == 0:
                ax.set_ylabel(label)
        axes[0, j].set_title(f"{model} ({n} layers)", loc="left")
        axes[0, j].annotate(f"refusal direction\n(layer {int(d['layer'])})", (d["layer"], 0),
                            xycoords=axes[0, j].get_xaxis_transform(), xytext=(-4, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=8, color=MUTED)
        axes[-1, j].set_xlabel("Layer")
    # Label line ends once every row's y-limits are final (sharey).
    for j, model in enumerate(models):
        for i, (name, _) in enumerate(rows):
            g = gaps[(gaps.model == model) & (gaps.comparison == name)]
            last = g[g.layer == g.layer.max()]
            label_ends(axes[i, j], [(r.layer, 100 * r.share, r.kind) for r in last.itertuples()])
    handles = [Line2D([], [], color=c, lw=1.6, label=KIND_LABEL[k]) for k, c in KIND_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Each layer uses its own refusal direction (harmful-minus-harmless mean at that layer). "
             "Positive = closer to refusal. Bands: ~95% CI over matched pairs (136 stolen, 48 benign).",
             fontsize=8, color=MUTED)
    # Fixed room in inches for the legend above and the note below, whatever the height.
    fig.tight_layout(rect=(0, 0.22 / height, 0.97, 1 - 0.42 / height))
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


def print_onsets(gaps, models):
    for model in models:
        layer = int(load_direction(model)["layer"])
        print(f"\n===== {model} (refusal direction at layer {layer})")
        for (name, kind), g in gaps[gaps.model == model].groupby(["comparison", "kind"]):
            sig = g[g.sig_bonferroni]
            at = g[g.layer == layer].iloc[0]
            peak = g.loc[g.share.abs().idxmax()]
            first = f"layer {sig.layer.min()}" if len(sig) else "never"
            print(f"  {name:24s} {kind:7s} first significant: {first:9s} "
                  f"at layer {layer}: {100 * at.share:+.1f}%  "
                  f"largest: {100 * peak.share:+.1f}% at layer {int(peak.layer)}")


def main():
    models = sys.argv[1:] or ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
    style()
    FIG.mkdir(exist_ok=True)
    refusal_curve(models)
    gaps = pd.concat([layer_gaps(m) for m in models])
    gaps.to_csv(ROOT / "results" / "party_gap_by_layer.csv", index=False)
    party_gap(gaps, models)
    # Slide versions: one story per figure, larger type.
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11}):
        party_gap(gaps, models, ["crossed vs aligned"], "slide_alignment_gap.png")
        party_gap(gaps, models, ["Republican vs Democrat", "party vs fan"], "slide_party_gap.png")
    print_onsets(gaps, models)


if __name__ == "__main__":
    main()
