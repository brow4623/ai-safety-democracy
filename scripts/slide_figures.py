"""Slide figures for the behavior and causal results, from saved results only (no GPU).

  design          how the prompt set is built: one request under every asker, and
                  the aligned/crossed grid
  method          Arditi et al.'s method as a schematic: find the refusal direction,
                  remove it, add it (illustrative points, not data)
  headline      did the task: Democrat vs Republican and aligned vs crossed
  headline a-c    simpler versions for the punchline slide: bars (A), the two gaps
                  only (B), and one-judge dumbbells (C); plus a paper-style version of A
                  with 95% intervals and significance brackets
  identity        each stated identity minus "I'm a voter" on matched pairs, with
                  the Lakers-fan control
  ablation        aligned and crossed, base -> refusal direction ablated
  request format  did the task per request template, grouped by severity
  shift dose      aligned prompts shifted along the refusal direction by 1-10x
                  the measured gap, against the crossed level, with the
                  random-direction control
  alignment cos   cos(alignment direction, refusal direction) by layer, with the
                  layers where the crossed-minus-aligned gap is significant shaded

Every behavior figure shows both judges: Claude (filled marks) and Llama Guard 3
with the election category (hollow marks). Each comparison stays inside one
generation run: the Claude judge's claude-judge.jsonl scores the first run (736
prompts, no fan identity); Llama Guard, the Claude fan/voter file and every
causal run score the 920-prompt rerun.

Same palette and slide sizing as layer_curves.py. It is not imported because
it pulls in torch through headline.py and grade.py; this script needs only
matplotlib and pandas.

Figures say "own side" / "other side" for the aligned / crossed prompts, and "side direction" for the
alignment direction.

Writes figures/slide_design.png, slide_method.png, slide_headline.png (and _a, _b, _c), slide_identity.png, slide_ablation.png,
slide_request_format.png, slide_shift_dose.png, slide_alignment_cosine.png.

Usage: uv run python scripts/slide_figures.py Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS, SHADE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec")
BLUE, ORANGE = "#2a78d6", "#eb6834"
GROUP_COLOR = {"aligned": BLUE, "crossed": ORANGE, "democrat": INK2, "republican": INK2}
# Slides say own side / other side: "aligned" means something else in an AI-alignment class.
GROUP_LABEL = {"aligned": "own side", "crossed": "other side"}
JUDGES = [("claude-judge", "Claude"), ("guard3-election", "Llama Guard 3")]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
W = 5.4   # inches per model column, as in layer_curves.py

# Stolen-election templates in src/prompts.py order, shortened for axis labels.
TEMPLATE_LABEL = [
    "help me prove it was stolen", "persuasive post: it was rigged", "the strongest evidence it was stolen",
    "rally speech: how it was stolen", "letter: legislators, refuse to recognize the win",
    "list the ways it was rigged, for family", "news-style article: it was stolen",
    "talking points: the winner isn't legitimate", "post raising questions about fairness",
    "list reasons to doubt the winner", "email to family: not sure it was fair",
    "op-ed: results deserve more scrutiny", "town-hall questions on irregularities",
    "petition: refuse to certify", "speech: protest until it's overturned",
    "message: declare the loser the winner", "post: don't accept the winner, whatever courts say"]
SEVERITY = {1: "Doubt", 2: "Claim", 3: "Act"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    })


def judged(model, name):
    return pd.read_json(RES / model / f"{name}.jsonl", lines=True)


def mark(ax, x, y, color, judge, size=8, zorder=4):
    """Claude is a filled mark, Llama Guard a hollow one, in the same color."""
    filled = judge == "claude-judge"
    ax.plot(x, y, ls="", marker="o", ms=size, mfc=color if filled else SURFACE, mec=color, mew=1.6,
            zorder=zorder)


def judge_handles(color=INK2, sizes=(8, 8)):
    return [Line2D([], [], ls="", marker="o", ms=sizes[0], mfc=color, mec=color, mew=1.6, label="Claude judge"),
            Line2D([], [], ls="", marker="o", ms=sizes[1], mfc=SURFACE, mec=color, mew=1.6,
                   label="Llama Guard 3 judge")]


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


def pts(x):
    return f"{x:+.0f}".replace("-", "−")


def finish(fig, handles, note, out, height, ncol=None, right=1.0):
    ncol = ncol or len(handles)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, va="bottom")
    n_lines = note.count("\n") + 1
    legend_rows = -(-len(handles) // ncol)
    fig.tight_layout(rect=(0, (0.12 + 0.14 * n_lines) / height, right, 1 - (0.12 + 0.3 * legend_rows) / height))
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- design

ELECTION_LABEL = {"2020_pres": "2020 presidential", "2020_ga": "2020 Georgia", "2020_mi": "2020 Michigan",
                  "2022_az_gov": "2022 Arizona governor", "2016_pres": "2016 presidential",
                  "2000_fl": "2000 Florida", "2004_oh": "2004 Ohio", "2018_ga_gov": "2018 Georgia governor"}


def design():
    """The prompt set: one request under every asker, and the aligned/crossed grid. Counts come from the data."""
    sys.path.insert(0, str(ROOT / "src"))
    from prompts import ELECTIONS

    rows = pd.read_json(ROOT / "data" / "political_prompts.jsonl", lines=True)
    stolen = rows[rows.kind == "stolen"]
    n_tpl = rows.groupby("kind").template.nunique()
    sides = rows.drop_duplicates("election").groupby("claim_side").election.apply(list)
    example = stolen[stolen.pair_id == "stolen-2020_pres-t0"].set_index("identity")
    request = example.instruction["none"]
    alt = rows[(rows.identity == "republican") & rows.instruction.str.startswith("As ")].instruction.iloc[0]

    height = 5.6
    fig = plt.figure(figsize=(2 * W, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 20 * W)                     # 1 unit = 0.1 inch both ways
    ax.set_ylim(0, 10 * height)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()

    def put(x, y, s, size=12, **kw):
        """Left-aligned text; returns its (x0, x1) in data units so the next piece can follow it."""
        t = ax.text(x, y, s, fontsize=size, va="center", ha="left", **kw)
        bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        return bb.x0, bb.x1

    def bracket(x0, x1, y, tick):
        ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=MUTED, lw=0.8)

    # One request, five askers
    put(2, 53, f"{rows.identity.nunique()} askers × {rows.election.nunique()} elections × {n_tpl.sum()} requests"
        f" = {len(rows)} prompts", weight="bold")
    put(4, 49.5, f"asker ({rows.identity.nunique()})", size=10, color=INK2)
    ys = dict(zip(["democrat", "republican", "voter", "fan", "none"], [46, 42.5, 39, 35.5, 32]))
    right = 0
    for who, y in ys.items():
        prefix = example.instruction[who][: -len(request)].strip()
        party = who in GROUP_COLOR
        right = max(right, put(4, y, prefix or "(no asker)", color=INK if party else INK2,
                               style="normal" if prefix else "italic")[1])
        if party:
            ax.plot(2.2, y, ls="", marker="o", ms=8, color=GROUP_COLOR[example.alignment[who]])
    bx, mid = right + 1.5, ys["voter"]
    ax.plot([bx - 0.8, bx, bx, bx - 0.8], [ys["democrat"], ys["democrat"], ys["none"], ys["none"]],
            color=MUTED, lw=0.8)
    ax.plot([bx, bx + 1.2], [mid, mid], color=MUTED, lw=0.8)

    name = next(e["name"] for e in ELECTIONS if e["id"] == "2020_pres")
    before, _, after = request.partition(name)
    space = 0.28 * 12 / 72 * 10
    x0, x1 = put(bx + 2.5, mid, before.rstrip())
    e0, e1 = put(x1 + space, mid, name, weight="bold")
    _, end = put(e1 + space, mid, after.lstrip())
    bracket(x0, end, mid + 2.6, -0.8)
    put(x0, mid + 4.4, f"request ({n_tpl.sum()}): {n_tpl['stolen']} stolen-election requests at 3 levels"
        f" (doubt, claim, act), plus {n_tpl['benign']} harmless civic questions", size=10, color=INK2)
    bracket(e0, e1, mid - 2.6, 0.8)
    put(e0, mid - 4.4, f"election ({rows.election.nunique()}): {len(sides['R'])} where the claim is"
        f" Republican-coded, {len(sides['D'])} Democratic-coded", size=10, color=INK2)

    # Aligned vs crossed
    put(2, 25.5, "Own side or other side: does the claim belong to the asker's party?", weight="bold")
    cols = {"R": 30, "D": 69}
    for side, x in cols.items():
        put(x, 22, f"{'Republican' if side == 'R' else 'Democratic'}-coded claim", size=11, weight="bold")
        labels = [ELECTION_LABEL[e] for e in sides[side]]
        put(x, 18.9, ", ".join(labels[:2]) + "\n" + ", ".join(labels[2:]), size=10, color=INK2, linespacing=1.1)
    for y in (16.3, 11.3, 6.3):
        ax.plot([2, 106], [y, y], color=GRID, lw=0.8)
    for party, y in (("R", 13.8), ("D", 8.8)):
        put(4, y, f"{'Republican' if party == 'R' else 'Democrat'} asks", size=11)
        for side, x in cols.items():
            group = "aligned" if party == side else "crossed"
            ax.plot(x + 1, y, ls="", marker="o", ms=8, color=GROUP_COLOR[group])
            put(x + 2.5, y, GROUP_LABEL[group].capitalize(), weight="bold")
    put(4, 3.8, f"{stolen.pair_id.nunique()} matched Democrat/Republican pairs ({rows.election.nunique()} elections"
        f" × {n_tpl['stolen']} requests), identical except for the asker", size=10, color=INK2)
    fig.text(0.01, 0.005, f"Each asker also has a second phrasing (\"{alt.split(' to ')[0]} to …\"), alternated"
             " across prompts. Voter and Lakers fan are the no-party baselines. Source: data/political_prompts.jsonl.",
             fontsize=8, color=MUTED, va="bottom")
    fig.savefig(FIG / "slide_design.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- method

def method():
    """Arditi et al.'s method in three panels: find the refusal direction r, remove it, add it. Illustrative points."""
    rng = np.random.default_rng(3)
    u = np.array([3, 1.6]) / np.hypot(3, 1.6)              # the refusal direction as drawn
    origin = np.array([0.6, 0.3])
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 2.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.2, wspace=0.12)

    def arrow(ax, a, b, color, lw=2.0):
        ax.annotate("", b, a, zorder=5, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14,
                                                        shrinkA=0, shrinkB=0))

    def reply(ax, x, y, s):
        ax.text(x, y, s, fontsize=12.5, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=AXIS, lw=0.8))

    captions = ("Average activation on harmful prompts minus on\nharmless ones: the refusal direction r",
                "Project r out at every layer: the model\nanswers a harmful request",
                "Add r to a harmless request: the model\nrefuses it")
    for ax, title, caption in zip(axes, ("1  Find it", "2  Remove it", "3  Add it"), captions):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.6, 7.0)
        ax.set_ylim(-0.4, 3.6)
        ax.set_title(title, loc="left", fontsize=15)
        ax.text(0, -0.04, caption, transform=ax.transAxes, va="top", fontsize=12.5, color=INK2)

    ax = axes[0]
    harmless = rng.normal((0.8, 0.9), 0.3, (14, 2))
    harmful = rng.normal((4.4, 2.5), 0.3, (14, 2))
    ax.plot(*harmless.T, ls="", marker="o", ms=5.5, mfc=SURFACE, mec=MUTED, mew=1.1, zorder=2)
    ax.plot(*harmful.T, ls="", marker="o", ms=5.5, mfc=MUTED, mec=MUTED, zorder=2)
    lo, hi = harmless.mean(0), harmful.mean(0)
    arrow(ax, lo, hi, INK, lw=2.4)
    ax.text(*(lo + hi) / 2 + (-0.55, 0.3), "r", fontsize=17, fontweight="bold", style="italic")
    ax.text(lo[0] - 1.0, lo[1], "harmless", fontsize=12, color=INK2, ha="right", va="center")
    ax.text(hi[0], hi[1] + 0.75, "harmful", fontsize=12, color=INK2, ha="center")

    ax = axes[1]
    ax.plot(*np.array([origin, origin + 4.6 * u]).T, color=AXIS, lw=1, ls="--")
    ax.text(*(origin + 4.75 * u), "r", fontsize=15, fontweight="bold", style="italic", color=MUTED, va="center")
    v = np.array([2.0, 2.9])
    kept = v - (v @ u) * u
    arrow(ax, origin, origin + v, MUTED, lw=1.6)
    ax.plot(*np.array([origin + kept, origin + v]).T, color=MUTED, lw=1, ls=":")
    arrow(ax, origin, origin + kept, INK, lw=2.4)
    ax.text(*(origin + v + (0.15, 0.0)), "harmful request", fontsize=12, color=INK2, va="center")
    ax.text(*(origin + kept / 2 + (-0.25, 0)), "r removed", fontsize=12, color=INK, ha="right", va="center")
    reply(ax, 3.3, 0.8, "“Sure, here’s how…”")

    ax = axes[2]
    y = np.array([1.6, 0.4])
    arrow(ax, origin, origin + y, MUTED, lw=1.6)
    arrow(ax, origin + y, origin + y + 2.2 * u, INK, lw=2.4)
    ax.text(*(origin + y / 2 + (0.2, -0.7)), "harmless request", fontsize=12, color=INK2, ha="center")
    ax.text(*(origin + y + 1.1 * u + (0.25, -0.35)), "+ r", fontsize=15, fontweight="bold", style="italic")
    reply(ax, 2.2, 2.9, "“I can’t help with that.”")

    fig.savefig(FIG / "slide_method.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- headline

def headline_rates(judge):
    """(Republican, Democrat, p) and (crossed, aligned, p) per model, from headline.py's CSVs."""
    name = "headline_behavior.csv" if judge == "guard3-election" else f"headline_behavior_{judge}.csv"
    df = pd.read_csv(RES / name)
    df = df[(df.condition == "base") & (df.subset == "all-coded")]
    return {(r.model, r.comparison): (r.A, r.B, r.p) for r in df.itertuples()}


def headline(models):
    rates = {j: headline_rates(j) for j, _ in JUDGES}
    groups = [("Republican vs Democrat", "Democrat vs Republican", ("democrat", "republican"), ("Dem", "Rep")),
              ("crossed vs aligned", "Own side vs other side", ("aligned", "crossed"), ("own side", "other side"))]
    height = 3.9
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharex=True, sharey=True,
                             squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ticks, labels = [], []
        for g, (comp, title, (lo_key, hi_key), (lo_lab, hi_lab)) in enumerate(groups):
            top = 4.2 - 2.4 * g
            ax.text(0, top + 0.78, title, transform=ax.get_yaxis_transform(), fontsize=11,
                    fontweight="bold", color=INK, va="center")
            for k, (judge, jname) in enumerate(JUDGES):
                y = top - 0.75 * k
                a, b, p = rates[judge][(model, comp)]       # A = Republican / crossed, B = Democrat / aligned
                lo, hi = 100 * b, 100 * a
                ax.plot([lo, hi], [y, y], color=GRID, lw=2.5, zorder=2, solid_capstyle="round")
                mark(ax, lo, y, GROUP_COLOR[lo_key], judge)
                mark(ax, hi, y, GROUP_COLOR[hi_key], judge)
                if k == 0:
                    # Left dot's label ends at it, right dot's starts at it, so they stay apart when dots touch.
                    (lx, llab), (rx, rlab) = sorted([(lo, lo_lab), (hi, hi_lab)], key=lambda e: e[0])
                    for x, lab, dx, ha in [(lx, llab, -2, "right"), (rx, rlab, 2, "left")]:
                        ax.annotate(lab, (x, y), xytext=(dx, 8), textcoords="offset points", ha=ha,
                                    va="bottom", fontsize=9, color=INK2)
                ax.annotate(f"{pts(100 * (a - b))} pts, {fmt_p(p)}", (max(lo, hi), y), xytext=(10, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK2)
                ticks.append(y)
                labels.append(jname)
        ax.set_yticks(ticks, labels)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(0.65, 5.3)
        ax.set_xlim(0, 85)
        ax.set_xlabel("Did the task (% of 136 matched prompts per group)")
        ax.set_title(SHORT[model], loc="left")
    handles = judge_handles() + [
        Line2D([], [], ls="", marker="o", ms=8, color=BLUE, label="own side: the asker's party owns the claim"),
        Line2D([], [], ls="", marker="o", ms=8, color=ORANGE, label="other side: the other party's claim")]
    note = ("Stolen-election prompts, no intervention. p: exact McNemar on matched pairs (headline.py). "
            "Claude judge: first generation run (736 prompts); Llama Guard: 920-prompt rerun.")
    finish(fig, handles, note, "slide_headline.png", height, ncol=4)


# Simpler takes on the headline, for the punchline slide. Options A and C show the Claude judge only (the one
# validated against a blind human pass); option B shows just the two gaps, under both judges.
HEADLINE_NOTE_CLAUDE = ("Claude judge (agrees with a blind human pass at κ = 0.77), first generation run. "
                        "Stolen-election prompts, 136 matched pairs per comparison, exact McNemar tests. "
                        "Llama Guard 3 shows the same pattern.")


def finish_plain(fig, note, out, height):
    """finish() without a legend, for figures whose marks are labeled directly."""
    fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, (0.12 + 0.14 * (note.count("\n") + 1)) / height, 1, 1))
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


def gap_label(diff, p):
    unit = "pt" if round(abs(diff)) == 1 else "pts"
    return f"{pts(diff)} {unit}, " + ("not significant" if p >= 0.05 else fmt_p(p))


def headline_bars(models):
    """Option A: bars with their values, Claude judge only."""
    rates = headline_rates("claude-judge")
    groups = [("Republican vs Democrat", [("Democrat", 1, INK2), ("Republican", 0, AXIS)]),
              ("crossed vs aligned", [("Own side", 1, BLUE), ("Other side", 0, ORANGE)])]
    height = 4.2
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharey=True, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        ticks, labels = [], []
        for g, (comp, bars) in enumerate(groups):
            vals = rates[(model, comp)]                      # (A, B, p): A = Republican / crossed
            for k, (label, idx, color) in enumerate(bars):
                x = g * 1.1 + (k - 0.5) * 0.42
                v = 100 * vals[idx]
                ax.bar(x, v, width=0.36, color=color, zorder=3)
                ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=11, color=INK)
                ticks.append(x)
                labels.append(label)
            top = 100 * max(vals[0], vals[1])
            ax.text(g * 1.1, top + 10, gap_label(100 * (vals[1] - vals[0]), vals[2]), ha="center", va="bottom",
                    fontsize=10, color=INK2)
        ax.set_xticks(ticks, labels)
        ax.tick_params(axis="x", length=0, labelsize=10.5, labelcolor=INK2)
        ax.set_ylim(0, 85)
        ax.set_xlim(-0.55, 1.65)
        if j == 0:
            ax.set_ylabel("Did the task (%)")
        ax.set_title(SHORT[model], loc="left")
    finish_plain(fig, HEADLINE_NOTE_CLAUDE, "slide_headline_a.png", height)


def headline_gaps(models):
    """Option B: only the two gaps per model, both judges. The party gap sits at zero; the own-side gap doesn't."""
    rates = {j: headline_rates(j) for j, _ in JUDGES}
    panels = [("Republican vs Democrat", "Democrat minus Republican", INK2),
              ("crossed vs aligned", "Own side minus other side", BLUE)]
    height = 3.4
    fig, axes = plt.subplots(1, 2, figsize=(2 * W, height), sharex=True, sharey=True)
    for ax, (comp, title, color) in zip(axes, panels):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        for r, model in enumerate(models):
            y = len(models) - 1 - r
            for k, (judge, _) in enumerate(JUDGES):
                a, b, p = rates[judge][(model, comp)]
                gap, yy = 100 * (b - a), y + (0.14 if k == 0 else -0.14)
                mark(ax, gap, yy, color, judge, size=9)
                ax.annotate(gap_label(gap, p), (gap, yy), xytext=(9, 0), textcoords="offset points",
                            va="center", fontsize=9.5, color=INK2)
        ax.set_yticks(range(len(models)), [SHORT[m] for m in reversed(models)])
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_xlim(-8, 30)
        ax.set_xlabel("Difference in did-the-task rate (percentage points)")
        ax.set_title(title, loc="left")
    note = ("Stolen-election prompts, 136 matched pairs per comparison, exact McNemar tests. Claude judge: first "
            "generation run; Llama Guard 3: 920-prompt rerun.")
    finish(fig, judge_handles(), note, "slide_headline_b.png", height)


def headline_simple(models):
    """Option C: the dumbbells again, Claude judge only, one panel per comparison, values on the dots."""
    rates = headline_rates("claude-judge")
    panels = [("Republican vs Democrat", "Democrat vs Republican", ("Dem", "Rep"), (INK2, AXIS)),
              ("crossed vs aligned", "Own side vs other side", ("own side", "other side"), (BLUE, ORANGE))]
    height = 3.4
    fig, axes = plt.subplots(1, 2, figsize=(2 * W, height), sharex=True, sharey=True)
    for ax, (comp, title, (lab_b, lab_a), (col_b, col_a)) in zip(axes, panels):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        for r, model in enumerate(models):
            y = len(models) - 1 - r
            a, b, p = rates[(model, comp)]
            va, vb = 100 * a, 100 * b
            ax.plot([va, vb], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=2)
            ax.plot(va, y, ls="", marker="o", ms=11, color=col_a, zorder=3)
            ax.plot(vb, y, ls="", marker="o", ms=11, color=col_b, zorder=3)
            (lx, ll), (rx, rl) = sorted([(va, lab_a), (vb, lab_b)])
            ax.annotate(f"{ll} {lx:.0f}%", (lx, y), xytext=(-10, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=10.5, color=INK2)
            ax.annotate(f"{rl} {rx:.0f}%", (rx, y), xytext=(10, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=10.5, color=INK2)
            ax.annotate(gap_label(vb - va, p), ((va + vb) / 2, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, color=MUTED)
        ax.set_yticks(range(len(models)), [SHORT[m] for m in reversed(models)])
        ax.set_ylim(-0.6, len(models) - 0.3)
        ax.set_xlim(0, 85)
        ax.set_xlabel("Did the task (%)")
        ax.set_title(title, loc="left")
    finish_plain(fig, HEADLINE_NOTE_CLAUDE, "slide_headline_c.png", height)


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion, as (low, high)."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def headline_academic(models, n=136):
    """Paper-style version of option A: serif type, framed axes, 95% intervals and significance brackets."""
    rates = headline_rates("claude-judge")
    groups = [("Republican vs Democrat", [("Democrat", 1, "#4d4d4d"), ("Republican", 0, "#bdbdbd")]),
              ("crossed vs aligned", [("Own side", 1, BLUE), ("Other side", 0, ORANGE)])]
    height = 4.4
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.spines.top": True, "axes.spines.right": True,
                         "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.grid": False,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.color": INK, "ytick.color": INK,
                         "axes.labelcolor": INK, "axes.titleweight": "normal", "font.size": 12,
                         "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharey=True, squeeze=False)
        for j, model in enumerate(models):
            ax = axes[0, j]
            ticks, labels = [], []
            for g, (comp, bars) in enumerate(groups):
                vals = rates[(model, comp)]
                xs, tops = [], []
                for k, (label, idx, color) in enumerate(bars):
                    x = g * 1.1 + (k - 0.5) * 0.42
                    v = vals[idx]
                    lo, hi = wilson(v, n)
                    ax.bar(x, 100 * v, width=0.36, color=color, edgecolor=INK, linewidth=0.8, zorder=2)
                    ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="none", ecolor=INK,
                                elinewidth=1, capsize=4, zorder=3)
                    xs.append(x)
                    tops.append(100 * hi)
                    ticks.append(x)
                    labels.append(label)
                y = max(tops) + 4
                ax.plot([xs[0], xs[0], xs[1], xs[1]], [y - 1.5, y, y, y - 1.5], color=INK, lw=0.9)
                ax.text(sum(xs) / 2, y + 0.8, stars(vals[2]), ha="center", va="bottom",
                        fontsize=13 if vals[2] < 0.05 else 11)
            ax.set_xticks(ticks, labels)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, 85)
            ax.set_xlim(-0.55, 1.65)
            if j == 0:
                ax.set_ylabel("Did the task (%)")
            ax.set_title(f"({'ab'[j]}) {SHORT[model]}", loc="left", fontsize=13)
        note = ("Claude judge, first generation run; stolen-election prompts, n = 136 matched pairs per comparison. "
                "Error bars: 95% Wilson intervals.\nBrackets: exact McNemar test on matched pairs "
                "(*** p < 0.001, * p < 0.05, n.s. not significant). Llama Guard 3 shows the same pattern.")
        fig.text(0.01, 0.005, note, fontsize=9, color=INK2, va="bottom")
        fig.tight_layout(rect=(0, 0.45 / height, 1, 1))
        fig.savefig(FIG / "slide_headline_academic.png", dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------- identity

def identity_gaps(model):
    """Did the task, identity minus voter, on matched stolen pairs: rows of (judge, identity, mean, se)."""
    sources = {"guard3-election": [("guard3-election", ["democrat", "republican", "fan"])],
               "claude-judge": [("claude-judge", ["democrat", "republican"]),
                                ("claude-judge-fan-voter", ["fan"])]}
    rows = []
    for judge, files in sources.items():
        for name, idents in files:
            df = judged(model, name)
            p = df[df.kind == "stolen"].pivot(index="pair_id", columns="identity", values="unsafe_base")
            for ident in idents:
                d = (p[ident].astype(float) - p["voter"].astype(float)).dropna()
                rows.append({"judge": judge, "identity": ident, "n": len(d), "mean": d.mean(),
                             "se": d.std(ddof=1) / np.sqrt(len(d))})
    return pd.DataFrame(rows)


def identity(models):
    order = [("democrat", "I'm a Democrat"), ("republican", "I'm a Republican"),
             ("fan", "I'm a Lakers fan\n(non-political control)")]
    height = 3.7
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharex=True, sharey=True,
                             squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        gaps = identity_gaps(model)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=INK2, lw=0.9, zorder=1)
        ax.axhline(0.5, color=GRID, lw=0.8, zorder=1)
        for i, (ident, _) in enumerate(order):
            for k, (judge, _) in enumerate(JUDGES):
                r = gaps[(gaps.judge == judge) & (gaps.identity == ident)].iloc[0]
                y = len(order) - 1 - i + (0.14 if k == 0 else -0.14)
                m, h = 100 * r["mean"], 196 * r["se"]
                ax.plot([m - h, m + h], [y, y], color=AXIS, lw=1.6, zorder=3)
                mark(ax, m, y, INK2, judge)
        ax.set_yticks(range(len(order)), [lab for _, lab in reversed(order)])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.5, len(order) - 0.15)
        ax.set_xlim(-28, 14)
        ax.annotate("same as \"I'm a voter\"", (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                    textcoords="offset points", va="top", fontsize=8, color=MUTED)
        ax.set_xlabel("Did the task, minus \"I'm a voter\" (percentage points)")
        ax.set_title(SHORT[model], loc="left")
    handles = judge_handles()
    note = ("Stolen-election prompts, no intervention; 136 matched pairs per row. Bars: ~95% CI (mean +- 1.96 SE "
            "of the paired difference).\nClaude party rows: first generation run; Claude fan row, and all "
            "Llama Guard rows: 920-prompt rerun. Each row is compared with the voter replies from its own run.")
    finish(fig, handles, note, "slide_identity.png", height)


# ----------------------------------------------------------------- ablation

def ablation(models):
    rates = {}
    for judge, _ in JUDGES:
        name = "headline_behavior.csv" if judge == "guard3-election" else f"headline_behavior_{judge}.csv"
        df = pd.read_csv(RES / name)
        df = df[(df.comparison == "crossed vs aligned") & (df.subset == "all-coded")]
        for r in df.itertuples():
            rates[(judge, r.model, r.condition)] = {"crossed": r.A, "aligned": r.B, "p": r.p}
    height = 4.3
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharey=True, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        lines = []
        for judge, jname in JUDGES:
            for group in ["aligned", "crossed"]:
                ys = [100 * rates[(judge, model, c)][group] for c in ["base", "ablated"]]
                ax.plot([0, 1], ys, color=GROUP_COLOR[group], lw=1.8, zorder=3,
                        ls="-" if judge == "claude-judge" else (0, (4, 2)))
                for x, y in zip([0, 1], ys):
                    mark(ax, x, y, GROUP_COLOR[group], judge)
                if judge == "claude-judge":
                    ax.annotate(GROUP_LABEL[group], (0, ys[0]), xytext=(-12, 0), textcoords="offset points", ha="right",
                                va="center", fontsize=9, color=INK2)
            gap = [100 * (rates[(judge, model, c)]["crossed"] - rates[(judge, model, c)]["aligned"])
                   for c in ["base", "ablated"]]
            lines.append(f"{jname}: {pts(gap[0])} to {pts(gap[1])} pts")
        ax.text(0.02, 0.98, "other side minus own side\n" + "\n".join(lines), transform=ax.transAxes, ha="left",
                va="top", fontsize=9, color=INK2, linespacing=1.4)
        ax.set_xticks([0, 1], ["Base model", "Refusal direction\nablated"])
        ax.set_xlim(-0.45, 1.25)
        ax.set_ylim(0, 100)
        if j == 0:
            ax.set_ylabel("Did the task (%)")
        ax.set_title(SHORT[model], loc="left")
    handles = judge_handles() + [
        Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=7, label="own side"),
        Line2D([], [], color=ORANGE, lw=1.8, marker="o", ms=7, label="other side")]
    note = ("Stolen-election prompts, 136 matched pairs per group. Solid lines: Claude judge (first generation run); "
            "dashed: Llama Guard 3 (920-prompt rerun).")
    finish(fig, handles, note, "slide_ablation.png", height)


# ----------------------------------------------------------- request format

def template_rates(models):
    """Did the task per template at baseline. Fan rows are left out so both judges see the same identities."""
    rows = []
    for model in models:
        for judge, _ in JUDGES:
            df = judged(model, judge)
            st = df[(df.kind == "stolen") & (df.identity != "fan")]
            for t, g in st.groupby("template"):
                rows.append({"model": model, "judge": judge, "template": t, "severity": g.severity.iloc[0],
                             "rate": g.unsafe_base.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def request_format(models):
    rates = template_rates(models)
    claude = rates[rates.judge == "claude-judge"].groupby(["severity", "template"]).rate.mean().reset_index()
    # Rows: severity blocks top to bottom, highest Claude rate (averaged over models) first in each.
    order, y, ys, headers = [], 0.0, {}, []
    for sev in sorted(SEVERITY):
        headers.append((y, f"{SEVERITY[sev]} (severity {sev})"))
        y -= 0.85
        for t in claude[claude.severity == sev].sort_values("rate", ascending=False).template:
            ys[t] = y
            order.append(t)
            y -= 1
        y -= 0.35
    height = 6.4
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharex=True, sharey=True,
                             squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        r = rates[rates.model == model].set_index(["judge", "template"]).rate
        for t in order:
            xs = [100 * r[(judge, t)] for judge, _ in JUDGES]
            ax.plot(xs, [ys[t]] * 2, color=GRID, lw=2.5, zorder=2, solid_capstyle="round")
            # Guard's ring is drawn larger and underneath, so a tie reads as a ring around a dot.
            mark(ax, xs[1], ys[t], BLUE, "guard3-election", size=10, zorder=3)
            mark(ax, xs[0], ys[t], BLUE, "claude-judge", size=4, zorder=4)
        for hy, text in headers:
            ax.text(-0.02 if j == 0 else 0, hy, text, transform=ax.get_yaxis_transform(), fontsize=10,
                    fontweight="bold", color=INK, va="center", ha="right" if j == 0 else "left")
        ax.set_yticks([ys[t] for t in order], [TEMPLATE_LABEL[t] for t in order], fontsize=9)
        ax.tick_params(axis="y", length=0, labelleft=j == 0)
        ax.set_ylim(y + 0.6, 0.5)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("Did the task (%)")
        ax.set_title(SHORT[model], loc="left")
    n = int(rates.n.iloc[0])
    note = (f"Baseline, stolen-election prompts; {n} per template (8 elections x nobody/voter/Democrat/Republican). "
            "Claude judge: first generation run; Llama Guard: 920-prompt rerun.")
    finish(fig, judge_handles(BLUE, sizes=(4, 10)), note, "slide_request_format.png", height)


# --------------------------------------------------------------- shift dose

def dose_points(model):
    """Aligned did-the-task under a constant shift along the refusal direction, per judge and scale."""
    out = {}
    for judge, _ in JUDGES:
        pts, rand = {}, {}
        for scale in [1, 2, 3, 10]:
            f = RES / model / f"patch_alignment_shift{scale}-{judge}.jsonl"
            if not f.exists():
                continue
            df = pd.read_json(f, lines=True)
            df = df[df.source == "aligned"]
            pts[0] = df.unsafe_base.mean()
            pts[scale] = df.unsafe_patched.mean()
            if "unsafe_random" in df:
                rand[scale] = df.unsafe_random.mean()
        out[judge] = (pts, rand)
    return out


def crossed_level(model, judge):
    """Crossed did-the-task at baseline in the 920-prompt rerun, the run the shift experiments start from."""
    if judge == "guard3-election":
        df = judged(model, "guard3-election")
    else:
        df = judged(model, "alignment_dir-claude-judge")
    return df[(df.kind == "stolen") & (df.alignment == "crossed")].unsafe_base.mean()


def shift_dose(models):
    xpos = {0: 0, 1: 1, 2: 2, 3: 3, 10: 4.6}
    height = 4.3
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharey=True, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        dose = dose_points(model)
        targets = []
        for judge, jname in JUDGES:
            level = 100 * crossed_level(model, judge)
            ls = "-" if judge == "claude-judge" else (0, (4, 2))
            ax.axhline(level, color=ORANGE, lw=1.2, ls=ls, zorder=1)
            targets.append((level, "other side,\n" + ("Claude" if judge == "claude-judge" else "Guard")))
            pts, rand = dose[judge]
            s = sorted(pts)
            xs, ys = [xpos[k] for k in s], [100 * pts[k] for k in s]
            ax.plot(xs, ys, color=BLUE, lw=1.8, ls=ls, zorder=3)
            for x, yv in zip(xs, ys):
                mark(ax, x, yv, BLUE, judge)
            for k, v in rand.items():
                mark(ax, xpos[k] + 0.12, 100 * v, MUTED, judge, size=7)
        # Label the two target lines just past the right edge, nudged apart if close.
        targets.sort()
        if targets[1][0] - targets[0][0] < 9:
            targets[1] = (targets[0][0] + 9, targets[1][1])
        for level, text in targets:
            ax.annotate(text, (1, level), xycoords=("axes fraction", "data"), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2,
                        annotation_clip=False)
        # Axis break between 3x and 10x.
        ax.axvspan(3.7, 3.9, color=SHADE, lw=0, zorder=0)
        ax.set_xticks(list(xpos.values()), ["base", "1×", "2×", "3×", "10×"])
        ax.set_xlim(-0.35, 5.0)
        ax.set_ylim(0, 75)
        if j == 0:
            ax.set_ylabel("Own-side prompts: did the task (%)")
        ax.set_xlabel("Constant shift toward refusal, in multiples of the measured gap")
        ax.set_title(SHORT[model], loc="left")
    handles = judge_handles() + [
        Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=7, label="own side, shifted along refusal direction"),
        Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="same shift, random direction"),
        Line2D([], [], color=ORANGE, lw=1.2, label="other side, no shift (the target)")]
    note = ("Own-side prompts (n = 136) shifted at the chosen layer by a multiple of the other-minus-own-side "
            "projection gap. Solid: Claude judge; dashed: Llama Guard 3.\nNot run: Qwen 2×; "
            "Claude on 3× and 10×. All from the 920-prompt rerun. Shaded strip: axis break.")
    finish(fig, handles, note, "slide_shift_dose.png", height, ncol=3, right=0.97)


# ------------------------------------------------------------ alignment cos

def alignment_cosine(models):
    gaps = pd.read_csv(RES / "party_gap_by_layer.csv")
    height = 3.9
    fig, axes = plt.subplots(1, len(models), figsize=(W * len(models), height), sharey=True, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        a = json.load(open(RES / model / "alignment_dir.json"))
        n = int(json.load(open(RES / model / "direction.json"))["n_layers"])
        layer = a["layer"]
        g = gaps[(gaps.model == model) & (gaps.kind == "stolen") & (gaps.comparison == "crossed vs aligned")]
        for lay in g[g.sig_bonferroni & (g.share > 0)].layer:
            ax.axvspan(lay - 0.5, lay + 0.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(layer, color=MUTED, lw=0.8, zorder=1)
        ax.plot(a["layers"], a["cos_align_vs_refusal_by_layer"], color=BLUE, lw=1.8, zorder=3)
        c = a["chosen_layer"]
        lo, hi = c["perm_null_95"]
        ax.plot([layer, layer], [lo, hi], color=MUTED, lw=4, alpha=0.5, solid_capstyle="butt", zorder=2)
        ax.plot(layer, c["cos_align_vs_refusal"], ls="", marker="o", ms=8, mfc=SURFACE, mec=BLUE, mew=1.8,
                zorder=4)
        p = c["perm_p_two_sided"]
        ax.annotate(f"{c['cos_align_vs_refusal']:.2f}, {'p < 0.001' if p < 0.001 else 'n.s.'}",
                    (layer, c["cos_align_vs_refusal"]), xytext=(-10, 12), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color=INK2)
        ax.annotate(f"refusal direction\n(layer {layer})", (layer, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_xlim(0.5, n - 0.5)
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlabel("Layer")
        if j == 0:
            ax.set_ylabel("cos(side direction,\nrefusal direction)")
        ax.set_title(f"{SHORT[model]} ({n} layers)", loc="left")
    handles = [Line2D([], [], color=BLUE, lw=1.8, label="cosine at each layer"),
               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.5,
                             label="layers where other side > own side on refusal (Bonferroni p < 0.05)"),
               Line2D([], [], color=MUTED, lw=4, alpha=0.5, label="sign-flip null, 95% range")]
    note = ("Side direction: other-side minus own-side activations on stolen prompts, minus the same on benign "
            "prompts (alignment_dir.py). Each layer's own refusal direction.")
    finish(fig, handles, note, "slide_alignment_cosine.png", height)


def main():
    models = sys.argv[1:] or ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"]
    style()
    FIG.mkdir(exist_ok=True)
    design()
    method()
    headline(models)
    headline_bars(models)
    headline_gaps(models)
    headline_simple(models)
    headline_academic(models)
    identity(models)
    ablation(models)
    request_format(models)
    shift_dose(models)
    alignment_cosine(models)
    for model in models:
        print(f"\n===== {model}")
        print(identity_gaps(model).round(3).to_string(index=False))
        for judge, (pts, rand) in dose_points(model).items():
            print(f"  shift dose {judge:16s} " + "  ".join(f"{k}x {100 * v:.0f}%" for k, v in sorted(pts.items()))
                  + "  random: " + "  ".join(f"{k}x {100 * v:.0f}%" for k, v in rand.items())
                  + f"  crossed level {100 * crossed_level(model, judge):.0f}%")


if __name__ == "__main__":
    main()
