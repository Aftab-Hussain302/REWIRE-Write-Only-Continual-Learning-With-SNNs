#!/usr/bin/env python3
"""Everything the paper shows, generated from a finished run folder of the review package (scripts/run_all.py + merge):

    python3 make_plots.py --run /path/results_2026-10-09_0830 --out assets --style ieee --zip

Reads only the saved rows (rows.json files, the sweep and fidelity summaries, the paired tests); recomputes nothing from the learners.
Writes <out>/
  tables/*.tex        tabular bodies for \\input: main, ablation, cost, gate, recognition, fidelity, paired, full_<dataset>
  macros.tex          one \\newcommand per number the text may quote (\\AccRewireCifarHundred, \\EnergyRatioHlopMean, ...)
  numbers.json        the same numbers, for checking
  figs/*.{pdf,eps,png}  pareto, retention, chunk cost, stream, gate curve, recurrence sweep, ablation ladder
  summary.md          the pre-registered hypotheses with their pass / fail per dataset, and the sentences the numbers support
Change --style to re-render for another template (column widths, fonts) without touching the results."""
import argparse, glob, json, os, re, shutil, sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, MaxNLocator

# ------------------------------------------------------------------------------------------------------------ names and style
DS = ["mnist", "nmnist", "shd", "cifar10", "cifar100", "tinyimagenet"]   # raw-input datasets first, then the ResNet-feature ones
DSNAME = dict(mnist="MNIST", cifar10="CIFAR-10", nmnist="N-MNIST", shd="SHD", cifar100="CIFAR-100", tinyimagenet="Tiny-ImageNet", synthetic="Synthetic")
RAW = {"mnist", "nmnist", "shd"}                                       # no front end
LABEL = {"rewire": "REWIRE (ours)", "rewire_gated": "REWIRE-gated (ours)", "rewire_gated_maha": "REWIRE-gated, Mahalanobis (ours)",
         "rewire_gated_t095": "REWIRE-gated, $\\theta$=0.95 (ours)", "rewire_gated_b128": "REWIRE-gated, budget 128 (ours)",
         "rewire_v1": "REWIRE-v1: sketch only, ridge", "rewire_v1_gated": "REWIRE-v1 gated", "abl_sketch_lda": "Sketch only, LDA readout",
         "abl_dualview_ridge": "Dual view, ridge readout", "abl_width128": "REWIRE, sketch 128", "abl_width256": "REWIRE, sketch 256",
         "abl_select_errors": "REWIRE, selective write (errors)", "abl_select_random": "REWIRE, selective write (random)",
         "abl_v1_select_errors": "REWIRE-v1, selective write (errors)", "abl_v1_select_random": "REWIRE-v1, selective write (random)",
         "abl_v1_woodbury": "REWIRE-v1, recursive solve", "ctrl_slda_input": "SLDA on the inputs", "ctrl_relu": "ReLU layer instead of LIF",
         "ranpac": "RanPAC-style (ReLU, ridge)", "ridge_input": "Ridge on the inputs", "hd_memory": "HD bundling", "hlop": "HLOP",
         "hlop_adaptive": "HLOP-R (subspace refreshed)", "hlop_gated": "HLOP, competence-gated", "replay": "Rehearsal (20/class)",
         "replay_iso": "Rehearsal (iso-memory)", "replay_gated": "Rehearsal, competence-gated", "ewc": "EWC", "naive": "Fine-tuning", "dsd_snn": "DSD-SNN"}
SHORT = {"rewire": "REWIRE", "rewire_gated": "REWIRE-g", "rewire_gated_maha": "REWIRE-g$_{\\rm M}$", "rewire_gated_t095": "REWIRE-g$_{.95}$", "rewire_gated_b128": "REWIRE-g$_{128}$",
         "rewire_v1": "v1", "rewire_v1_gated": "v1-g", "abl_sketch_lda": "sk+LDA", "abl_dualview_ridge": "dual+ridge", "abl_width128": "d$_s$128", "abl_width256": "d$_s$256",
         "abl_select_errors": "sel-err", "abl_select_random": "sel-rnd", "abl_v1_select_errors": "v1 sel-err", "abl_v1_select_random": "v1 sel-rnd", "abl_v1_woodbury": "v1 Woodbury",
         "ctrl_slda_input": "SLDA-in", "ctrl_relu": "ReLU", "ranpac": "RanPAC", "ridge_input": "Ridge-in", "hd_memory": "HD", "hlop": "HLOP", "hlop_adaptive": "HLOP-R",
         "hlop_gated": "HLOP-g", "replay": "Reh.", "replay_iso": "Reh.-iso", "replay_gated": "Reh.-g", "ewc": "EWC", "naive": "FT", "dsd_snn": "DSD"}
GROUP = {a: ("proposed" if a.startswith("rewire") and not a.startswith("rewire_v1") else "ablation" if a.startswith("abl_") or a.startswith("rewire_v1") else
             "control" if a in ("ctrl_slda_input", "ctrl_relu", "ranpac", "ridge_input", "hd_memory") else "baseline") for a in LABEL}
# One hue per method family (blue = ours, green = write-only controls, red = HLOP, amber = rehearsal, grey = fine-tuning / EWC); the
# four hues pass the dataviz validator on white under all pairs (worst normal-vision dE 15.1, CVD 6.2 with distinct markers as the
# second encoding); grey is the deliberately neutral "no protection" reference. Markers differ inside a family.
GROUP_COLOR = {"proposed": "#2a78d6", "ablation": "#2a78d6", "control": "#008300", "baseline": "#e34948"}
COLOR = {"replay": "#c98500", "replay_iso": "#c98500", "replay_gated": "#c98500", "ewc": "#777570", "naive": "#777570"}
MARK = {"rewire": "*", "rewire_gated": "P", "rewire_gated_maha": "p", "rewire_gated_t095": "h", "rewire_gated_b128": "P", "rewire_v1": "o", "rewire_v1_gated": "D",
        "abl_sketch_lda": "v", "abl_dualview_ridge": "^", "abl_width128": "<", "abl_width256": ">", "abl_select_errors": "x", "abl_select_random": "+",
        "abl_v1_select_errors": "x", "abl_v1_select_random": "+", "abl_v1_woodbury": "1", "ctrl_slda_input": "s", "ctrl_relu": "D", "ranpac": "o", "ridge_input": "v",
        "hd_memory": "^", "hlop": "s", "hlop_adaptive": "D", "hlop_gated": "o", "replay": "s", "replay_iso": "D", "replay_gated": "o", "ewc": "^", "naive": "X"}
FILLED = {"rewire", "rewire_gated", "rewire_gated_b128", "hlop", "replay", "ewc", "naive", "ctrl_slda_input", "rewire_v1"}
ORDER = list(LABEL)
DSTYPE = dict(mnist="pixels", nmnist="events", shd="spikes", cifar10="ResNet features", cifar100="ResNet features", tinyimagenet="ResNet features", synthetic="")
STYLE = {"ieee": dict(col=3.45, full=7.1, font=7.5), "iclr": dict(col=5.5, full=5.5, font=8), "neurips": dict(col=5.5, full=5.5, font=8)}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
FIG_LABEL = {}                                                          # per-run display names for the figures (set in main: the chosen gate variant)


def col(a): return COLOR.get(a, GROUP_COLOR.get(GROUP.get(a, "baseline"), "#777570"))
def mark(a): return MARK.get(a, "o")
def label(a): return FIG_LABEL.get(a, LABEL.get(a, a))
def short(a): return SHORT.get(a, a)
def is_open(a): return a not in FILLED
def order(arms): return [a for a in ORDER if a in arms] + sorted(a for a in arms if a not in ORDER)
def tint(hex_color, f=0.55):
    """A lighter version of a colour (mixed with white) - used instead of alpha so that EPS files keep no transparency."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5)); return "#%02x%02x%02x" % tuple(int(v + (255 - v) * f) for v in (r, g, b))


def style(font):
    plt.rcParams.update({"font.size": font, "font.family": "sans-serif", "axes.titlesize": font, "axes.labelsize": font, "legend.fontsize": font - 0.5,
                         "xtick.labelsize": font - 0.5, "ytick.labelsize": font - 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#898781", "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": False, "legend.frameon": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True, "lines.linewidth": 1.0, "errorbar.capsize": 1.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})


def save(fig, figs, name):
    import logging; logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)
    fig.savefig(f"{figs}/{name}.pdf"); fig.savefig(f"{figs}/{name}.png", dpi=300)
    try: fig.savefig(f"{figs}/{name}.eps", format="eps")
    except Exception: pass
    plt.close(fig)


def point(ax, x, y, a, xerr=None, yerr=None, ms=None):
    op = is_open(a); ms = ms or (8.5 if a == "rewire" else 5.2 if op else 4.6)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=mark(a), color=col(a), ms=ms, mfc="white" if op else col(a), mec=col(a) if op else "black", mew=0.9 if op else 0.4,
                elinewidth=0.5, ecolor=col(a), capsize=1.2, label=label(a), zorder=8 if a == "rewire" else (4 if op else 6))


def log_ticks(ax, lo, hi, axis="x"):
    """At most four labelled ticks on a log axis, chosen from decades, then 1-3, then 1-2-5; minor ticks stay unlabelled."""
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.01); a = ax.xaxis if axis == "x" else ax.yaxis
    for mult in ((1,), (1, 3), (1, 2, 5)):
        t = [m * 10 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1) for m in mult if lo <= m * 10 ** e <= hi]
        if len(t) >= 2: break
    while len(t) > 4: t = t[::2]
    a.set_major_locator(FixedLocator(t)); a.set_major_formatter(FixedFormatter([f"{v:g}" for v in t])); a.set_minor_formatter(NullFormatter())

# ------------------------------------------------------------------------------------------------------------ figure helpers
# Every figure: a fixed, small set of methods (the two proposed variants, the strongest write-only control, the SNN and rehearsal
# baselines, fine-tuning as the floor), one legend for the whole figure placed outside the panels by the layout engine, units in
# mJ, panels with margins, at most four labelled ticks on a log axis. Everything else is in the tables.
PARETO_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "naive"]
LINE_ARMS = ["rewire", "rewire_gated", "ctrl_slda_input", "hlop", "replay", "naive"]
RET_ARMS = ["rewire", "ctrl_slda_input", "hlop", "replay", "naive"]
THETA_SHADES = ["#9cc3ee", "#4f8fdc", "#0f4f9c"]                        # one hue, light -> dark, for the ordered thresholds
NEUTRAL = "#777570"


def new_fig(width, height, nrow=1, ncol=1, sharey=False, sharex=False):
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), squeeze=False, sharey=sharey, sharex=sharex, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.08)
    return fig, axes


def finish(fig, axes, figs, name, ncol=4, where="lower", handles=None, labels=None):
    """One legend for the whole figure, outside the panels (the constrained layout reserves the room), then save."""
    if handles is None:
        handles, labels, seen = [], [], set()
        for ax in np.ravel(axes):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen: seen.add(l); handles.append(h); labels.append(l)
    if handles:
        try: fig.legend(handles, labels, loc=f"outside {where} center", ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.2, handlelength=1.8)
        except ValueError:                                             # matplotlib < 3.7: an ordinary figure legend below the panels
            fig.legend(handles, labels, loc=f"{where} center", bbox_to_anchor=(0.5, -0.02 if where == "lower" else 1.02), ncol=ncol, frameon=False)
    save(fig, figs, name)


def panel_title(ax, ds, font, full=True):
    ax.set_title(f"{DSNAME[ds]} · {DSTYPE.get(ds, '')}".rstrip(" ·") if full else DSNAME[ds], fontsize=font, loc="left", color=INK, pad=3)


def line(ax, x, y, a, n_marks=8, offset=0, label=True):
    """A series line with a marker every few points (staggered per series) so that markers never pile up."""
    every = max(1, len(x) // n_marks); kw = dict(color=col(a), lw=1.4 if a == "rewire" else 0.9, zorder=8 if a == "rewire" else 5)
    ax.plot(x, y, marker=mark(a), ms=5.5 if a == "rewire" else 3.6, mfc="white" if is_open(a) else col(a), mec=col(a) if is_open(a) else "black",
            mew=0.8 if is_open(a) else 0.35, markevery=(offset % every, every), label=_lbl(a) if label else None, **kw)


def ylim_from(ax, ys, pad=0.08, cap=100.0):
    lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys)); r = max(hi - lo, 1.0); ax.set_ylim(lo - pad * r, min(cap + 0.02 * r, hi + pad * r))


def set_log_x(ax, vals, pad=1.7):
    lo, hi = float(np.nanmin(vals)) / pad, float(np.nanmax(vals)) * pad; ax.set_xscale("log"); ax.set_xlim(lo, hi); log_ticks(ax, lo, hi)


_lbl = label                                                              # figure labels: the registry names ("REWIRE (ours)", ...)


# ------------------------------------------------------------------------------------------------------------ figures
def fig_pareto(sc, dss, figs, st):
    """Accuracy against lifetime energy, one panel per dataset. Fine-tuning is drawn at the bottom edge with its value when it sits
    far below the others, so that the panel keeps its resolution for the methods that retain."""
    arms = [a for a in PARETO_ARMS if a in sc.arm.unique()]; ncol = 3 if len(dss) > 2 else max(1, len(dss)); nrow = int(np.ceil(len(dss) / ncol))
    fig, axes = new_fig(st["full"], 1.65 * nrow + 0.35, nrow, ncol)
    for i, ds in enumerate(dss):
        ax = axes[i // ncol, i % ncol]; d = sc[sc.dataset == ds]; m = d.groupby("arm").mean(numeric_only=True); se = d.groupby("arm").sem(numeric_only=True)
        here = [a for a in arms if a in m.index]; keep = [a for a in here if a != "naive"] or here
        acc = {a: 100 * m.loc[a, "final_acc"] for a in here}; en = {a: m.loc[a, "lifetime_energy_uJ"] / 1000 for a in here}
        lo = min(acc[a] for a in keep); hi = max(acc[a] for a in keep); floor = lo - 0.35 * max(hi - lo, 2.0)
        for a in here:
            if a == "naive" and acc[a] < floor:                        # off the bottom of the panel: the marker at the edge, with its value
                point(ax, en[a], floor, a); ax.annotate(f"{acc[a]:.0f}%", (en[a], floor), xytext=(5, 0), textcoords="offset points", fontsize=st["font"] - 1.5, color=INK2, va="center")
                continue
            point(ax, en[a], acc[a], a, xerr=se.loc[a, "lifetime_energy_uJ"] / 1000, yerr=100 * se.loc[a, "final_acc"])
        ax.set_ylim(floor - 0.08 * max(hi - lo, 2.0), hi + 0.22 * max(hi - lo, 2.0)); set_log_x(ax, list(en.values())); panel_title(ax, ds, st["font"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True)); ax.grid(True, axis="both")
        if i % ncol == 0: ax.set_ylabel("accuracy (%)")
        if i // ncol == nrow - 1: ax.set_xlabel("lifetime energy (mJ, log)")
    for j in range(len(dss), nrow * ncol): axes[j // ncol, j % ncol].axis("off")
    finish(fig, axes, figs, "fig_pareto", ncol=4)


def fig_retention(rows, dss, figs, st, name="fig_retention", width=None):
    """Accuracy on the returning class at the moment it returns (before any re-admission write), recurrence by recurrence."""
    ret = retention(rows); arms = [a for a in RET_ARMS if any(k[1] == a for k in ret)]; width = width or st["full"]
    ncol = 3 if len(dss) > 2 else max(1, len(dss)); nrow = int(np.ceil(len(dss) / ncol))
    fig, axes = new_fig(width, (1.75 if width > 5 else 1.85) * nrow + 0.3, nrow, ncol, sharey=True); allys = []
    for i, ds in enumerate(dss):
        ax = axes[i // ncol, i % ncol]
        for k, a in enumerate(arms):
            if (ds, a) in ret: y = 100 * ret[(ds, a)]; allys.append(y); line(ax, np.arange(1, len(y) + 1), y, a, n_marks=6, offset=k)
        ax.margins(x=0.04); panel_title(ax, ds, st["font"], full=width > 5); ax.grid(True, axis="y"); ax.grid(False, axis="x")
        if i % ncol == 0: ax.set_ylabel("accuracy on the returning\nclass at re-admission (%)")
        if i // ncol == nrow - 1: ax.set_xlabel("recurrence index")
    if allys: ylim_from(axes[0, 0], np.concatenate(allys), pad=0.1)
    for j in range(len(dss), nrow * ncol): axes[j // ncol, j % ncol].axis("off")
    finish(fig, axes, figs, name, ncol=min(5, len(arms)) if width > 5 else 3)


def fig_chunk_cost(cc, figs, st):
    """The three costs per chunk, mean over datasets: admission, maintenance overhead (energy per chunk above fine-tuning; zero by
    construction for the write-only learners, so no bar) and re-admission, with the share of recurrences the gate skipped."""
    g = cc.groupby("arm").mean(numeric_only=True); arms = [a for a in PARETO_ARMS if a in g.index]; gd = cc.groupby(["dataset", "arm"]).mean(numeric_only=True)
    fig, axes = new_fig(st["col"], 2.2); ax = axes[0, 0]; x = np.arange(len(arms)); w = 0.27
    intro = g.loc[arms, "intro_uJ"].values / 1000; recur = g.loc[arms, "recur_uJ"].values / 1000
    maint = np.array([np.nanmean([gd.loc[(ds, a), "intro_uJ"] - gd.loc[(ds, "naive"), "intro_uJ"] for ds in cc.dataset.unique() if (ds, a) in gd.index and (ds, "naive") in gd.index]) / 1000
                      if GROUP.get(a) == "baseline" and a != "naive" else np.nan for a in arms])
    ax.bar(x - w, intro, w, color=[tint(col(a), 0.6) for a in arms], edgecolor=[col(a) for a in arms], linewidth=0.6, label="admission")
    ax.bar(x, np.where(np.isfinite(maint) & (maint > 0), maint, np.nan), w, color="white", edgecolor=[col(a) for a in arms], linewidth=0.6, hatch="////", label="maintenance overhead")
    ax.bar(x + w, recur, w, color=[col(a) for a in arms], edgecolor="black", linewidth=0.35, label="re-admission")
    for i, a in enumerate(arms):
        sk = g.loc[a, "skipped"]
        if np.isfinite(sk) and sk > 0.02: ax.annotate(f"{100 * sk:.0f}%\nskipped", (i, max(intro[i], recur[i])), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=st["font"] - 1.5, color=INK2)
    ax.set_yscale("log"); vals = np.concatenate([intro, recur, maint[np.isfinite(maint) & (maint > 0)]]); log_ticks(ax, np.nanmin(vals) / 1.3, np.nanmax(vals) * 2.4, "y"); ax.set_ylim(np.nanmin(vals) / 1.3, np.nanmax(vals) * 2.4)
    ax.set_ylabel("energy per chunk (mJ, log)"); ax.set_xticks(x); ax.set_xticklabels([short(a) for a in arms], rotation=30, ha="right", rotation_mode="anchor")
    ax.grid(True, axis="y"); ax.grid(False, axis="x"); ax.margins(x=0.03)
    finish(fig, axes, figs, "fig_chunk_cost", ncol=3, where="upper")


def fig_stream(rows, ds, figs, st):
    """Accuracy on the classes seen so far and cumulative learner energy along one stream (mean over seeds)."""
    rs = [r for r in rows if r["dataset"] == ds]
    if not rs: return
    arms = [a for a in LINE_ARMS if any(r["arm"] == a for r in rs)]; n_intro = max(len([v for v in r["visits"] if v["kind"] == "intro"]) for r in rs)
    fig, axes = new_fig(st["full"], 2.05, 1, 2); accs = []
    for k, a in enumerate(arms):
        rr = [r for r in rs if r["arm"] == a]; acc = 100 * np.array([r["acc_curve"] for r in rr]).mean(0); en = np.array([r["energy_curve"] for r in rr]).mean(0) / 1000
        t = np.arange(1, len(acc) + 1); accs.append(acc); line(axes[0, 0], t, acc, a, n_marks=7, offset=k); line(axes[0, 1], t, en, a, n_marks=7, offset=k, label=False)
    for ax in axes[0]:
        ax.axvspan(0.5, n_intro + 0.5, color="#efeee9", zorder=0, lw=0); ax.set_xlabel("stream chunk"); ax.margins(x=0.03); ax.grid(True, axis="y"); ax.grid(False, axis="x")
        ax.text(n_intro / 2 + 0.5, 1.0, "new classes", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=st["font"] - 1, color=INK2)
        ax.text((n_intro + 1 + len(accs[0])) / 2, 1.0, "recurrences", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=st["font"] - 1, color=INK2)
    ylim_from(axes[0, 0], np.concatenate(accs), pad=0.1); axes[0, 0].set_ylabel("accuracy on the classes seen (%)")
    axes[0, 1].set_yscale("log"); axes[0, 1].set_ylabel("cumulative learner energy (mJ, log)"); lo_, hi_ = axes[0, 1].get_ylim(); log_ticks(axes[0, 1], lo_, hi_, "y")
    axes[0, 0].set_title(f"{DSNAME[ds]} · {DSTYPE.get(ds, '')}", loc="left", fontsize=st["font"], pad=12); axes[0, 1].set_title(" ", pad=12)
    finish(fig, axes, figs, f"fig_stream_{ds}", ncol=min(6, len(arms)))


def fig_gate(run, sc, dss, figs, st, wide=None):
    """The gate's operating points: for each dataset, the three competence thresholds (one hue, light -> dark) against the same
    seeds of ungated REWIRE: recurrences skipped, lifetime energy saved, and the accuracy it costs."""
    ref = sc[(sc.arm == "rewire")]; panels = []
    for ds in dss:
        p = f"{run}/{ds}/sweep_gate/sweep_gate_summary.csv"
        if not os.path.isfile(p): continue
        g = pd.read_csv(p); g = g[g.arm == "rewire_gated"].sort_values("theta_hi")
        if g.empty: continue
        r = ref[ref.dataset == ds]; r = r[r.seed.isin(sorted(r.seed.unique())[:int(g.n.max())])] if "seed" in r else r
        if r.empty: continue
        panels.append((ds, g, 100 * r.final_acc.mean(), r.lifetime_energy_uJ.mean()))
    if not panels: return
    thetas = sorted(set(float(t) for _, g, _, _ in panels for t in g.theta_hi)); shade = {t: THETA_SHADES[min(i, 2)] for i, t in enumerate(thetas)}
    keys = ("saved", "dacc") if wide is False or (wide is None and st["col"] < 4) else ("skipped", "saved", "dacc")   # column width: two panels
    fig, axes = new_fig(st["col"] if len(keys) == 2 else st["full"], 2.25 if len(keys) == 2 else 1.95, 1, len(keys)); x = np.arange(len(panels)); names = [DSNAME[ds] for ds, *_ in panels]
    series = {"skipped": lambda g, a0, e0: 100 * g.skipped.values, "saved": lambda g, a0, e0: 100 * (1 - g.lifetime_energy_uJ.values / e0), "dacc": lambda g, a0, e0: 100 * g.final_acc.values - a0}
    ylabels = {"skipped": "recurrences skipped (%)", "saved": "lifetime energy saved (%)", "dacc": "accuracy minus ungated (pts)"}
    for ax, key in zip(axes[0], keys):
        for i, (ds, g, a0, e0) in enumerate(panels):
            ys = series[key](g, a0, e0); ax.plot([i] * len(ys), ys, color="#c9c8c1", lw=0.8, zorder=2)
            for t, y in zip(g.theta_hi, ys): ax.plot(i, y, marker="o", ms=4.2, color=shade[float(t)], mec="black", mew=0.3, ls="none", zorder=5, label=f"$\\theta$ = {t:g}")
        ax.axhline(0, color="#898781", lw=0.5, zorder=1); ax.set_xticks(x); ax.set_xticklabels(names, rotation=40 if len(keys) == 2 else 30, ha="right", rotation_mode="anchor"); ax.set_ylabel(ylabels[key])
        ax.grid(True, axis="y"); ax.grid(False, axis="x"); ax.margins(x=0.08, y=0.15)
    default = float(sc.theta_hi.mode().iloc[0]) if "theta_hi" in sc else None
    handles = [plt.Line2D([], [], marker="o", ms=4.2, color=shade[t], mec="black", mew=0.3, ls="none", label=f"$\\theta$ = {t:g}" + (" (default)" if default is not None and abs(t - default) < 1e-9 else "")) for t in thetas]
    finish(fig, axes, figs, "fig_gate" if len(keys) == 2 or wide is None else "fig_gate_wide", ncol=3, handles=handles, labels=[h.get_label() for h in handles])
    if wide is None and len(keys) == 2: fig_gate(run, sc, dss, figs, st, wide=True)                                   # and the three-panel, double-column version


SWEEP_NUM = {}                                                           # filled by fig_sweep: energy growth and accuracy change across the sweep
RUN_DIR = [""]                                                           # the run folder, for the macro generator


def fig_sweep(run, ds, figs, st):
    """Lifetime energy and accuracy as the stream carries more recurrences (the recurrence-density sweep)."""
    rows = []
    for f in glob.glob(f"{run}/{ds}/sweep_recur/n*/rows.json"): rows += json.load(open(f))
    main = f"{run}/{ds}/shift_none/rows.json"
    if os.path.isfile(main): rows += json.load(open(main))
    if not rows: return
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]); arms = [a for a in LINE_ARMS if a in df.arm.unique()]; df = df[df.arm.isin(arms)]
    g = df.groupby(["arm", "n_recur"]).mean(numeric_only=True); fig, axes = new_fig(st["full"], 2.0, 1, 2); ns = sorted(df.n_recur.unique()); accs = []
    for k, a in enumerate(arms):
        xs = sorted(df[df.arm == a].n_recur.unique()); e = [g.loc[(a, x), "lifetime_energy_uJ"] / 1000 for x in xs]; acc = [100 * g.loc[(a, x), "final_acc"] for x in xs]; accs += acc
        line(axes[0, 0], xs, e, a, n_marks=len(xs)); line(axes[0, 1], xs, acc, a, n_marks=len(xs), label=False)
        if len(xs) > 1: SWEEP_NUM[(ds, a)] = dict(n_min=int(xs[0]), n_max=int(xs[-1]), energy_ratio=e[-1] / e[0], acc_change=acc[-1] - acc[0])
    for ax in axes[0]:
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(int(n)) for n in ns]); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("recurrence chunks in the stream")
        ax.margins(x=0.08); ax.grid(True, axis="y"); ax.grid(False, axis="x")
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("lifetime energy (mJ, log)"); lo_, hi_ = axes[0, 0].get_ylim(); log_ticks(axes[0, 0], lo_, hi_, "y")
    axes[0, 1].set_ylabel("accuracy (%)"); ylim_from(axes[0, 1], accs, pad=0.12)
    axes[0, 0].set_title(f"{DSNAME[ds]} · {DSTYPE.get(ds, '')}", loc="left", fontsize=st["font"])
    finish(fig, axes, figs, f"fig_sweep_{ds}", ncol=min(6, len(arms)))


def fig_ablation(sc, dss, figs, st):
    """Each configuration's accuracy relative to REWIRE: the mean over the raw-input datasets, the mean over the ResNet-feature
    datasets, and the range over all of them."""
    ladder = [("rewire_v1", "sketch only, ridge (v1)"), ("abl_sketch_lda", "sketch only, LDA head"), ("abl_dualview_ridge", "dual view, ridge head"),
              ("abl_width128", "sketch 128"), ("abl_width256", "sketch 256"), ("ctrl_relu", "ReLU layer instead of LIF"), ("ctrl_slda_input", "SLDA on the inputs"), ("ranpac", "RanPAC-style (ReLU, ridge)")]
    g = sc.groupby(["dataset", "arm"]).mean(numeric_only=True); ladder = [(a, n) for a, n in ladder if any((ds, a) in g.index for ds in dss)]
    if not ladder or not any((ds, "rewire") in g.index for ds in dss): return
    fig, axes = new_fig(st["col"], 0.27 * len(ladder) + 0.85); ax = axes[0, 0]; raw = [d for d in dss if d in RAW]; feat = [d for d in dss if d not in RAW]
    for i, (a, nm) in enumerate(ladder):
        y = len(ladder) - 1 - i; d = {ds: 100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, "rewire"), "final_acc"]) for ds in dss if (ds, a) in g.index and (ds, "rewire") in g.index}
        if not d: continue
        ax.plot([min(d.values()), max(d.values())], [y, y], color="#c9c8c1", lw=1.4, zorder=2, solid_capstyle="round")
        if raw and any(ds in d for ds in raw): ax.plot(np.mean([d[ds] for ds in raw if ds in d]), y, marker="o", ms=5, color=col(a), mec="black", mew=0.35, ls="none", zorder=6)
        if feat and any(ds in d for ds in feat): ax.plot(np.mean([d[ds] for ds in feat if ds in d]), y, marker="o", ms=5, color="white", mec=col(a), mew=1.0, ls="none", zorder=5)
    ax.axvline(0, color=INK, lw=0.7, zorder=3); ax.set_yticks(range(len(ladder))); ax.set_yticklabels([nm for _, nm in ladder][::-1]); ax.set_xlabel("accuracy relative to REWIRE (points)")
    ax.grid(True, axis="x"); ax.grid(False, axis="y"); ax.margins(y=0.08, x=0.1); ax.text(0, len(ladder) - 0.35, "REWIRE", ha="center", va="bottom", fontsize=st["font"] - 1, color=INK2)
    blue = GROUP_COLOR["proposed"]
    handles = [plt.Line2D([], [], marker="o", ms=5, color=blue, mec="black", mew=0.35, ls="none", label="mean over pixel / event inputs"),
               plt.Line2D([], [], marker="o", ms=5, color="white", mec=blue, mew=1.0, ls="none", label="mean over ResNet features"),
               plt.Line2D([], [], color="#c9c8c1", lw=1.4, label="range over the six datasets")]
    finish(fig, axes, figs, "fig_ablation", ncol=3, handles=handles, labels=[h.get_label() for h in handles])


# ------------------------------------------------------------------------------------------------------------ loading
def load_run(run):
    rows = []
    for ds in DS + ["synthetic"]:
        p = f"{run}/{ds}/shift_none/rows.json"
        if os.path.isfile(p): rows += json.load(open(p))
    if not rows: raise SystemExit(f"no <dataset>/shift_none/rows.json under {run} (run scripts/merge_results.py first)")
    scalar = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    return rows, scalar


def chunk_costs(rows):
    out = []
    for r in rows:
        e = np.asarray(r["energy_curve"], float); inc = np.diff(np.concatenate([[0.0], e])); v = r["visits"]
        kinds = np.array([x["kind"] for x in v]); wrote = np.array([x.get("wrote", True) for x in v], bool); rec = kinds == "recur"
        out.append(dict(dataset=r["dataset"], arm=r["arm"], seed=r["seed"], intro_uJ=float(inc[kinds == "intro"].mean()) if (kinds == "intro").any() else np.nan,
                        recur_uJ=float(inc[rec].mean()) if rec.any() else np.nan, skipped=float((~wrote[rec]).mean()) if rec.any() else np.nan))
    return pd.DataFrame(out)


def retention(rows):
    cur = {}
    for r in rows:
        pre = [x["pre_comp"] for x in r["visits"] if x["kind"] == "recur"]; cur.setdefault((r["dataset"], r["arm"]), []).append(pre)
    return {k: np.nanmean(np.array([c[:min(map(len, cs))] for c in cs]), 0) for k, cs in cur.items() if cs and min(map(len, cs)) > 0}


# ------------------------------------------------------------------------------------------------------------ tables
def fmt(m, s=None, best=None, second=None, nd=1):
    t = f"{m:.{nd}f}" + (f"{{\\tiny$\\pm${s:.{nd}f}}}" if s is not None and np.isfinite(s) else "")
    if best: t = f"\\textbf{{{t}}}"
    elif second: t = f"\\underline{{{t}}}"
    return t


def mark_best(vals, higher=True):
    v = np.array([x if np.isfinite(x) else (-np.inf if higher else np.inf) for x in vals], float); idx = np.argsort(-v if higher else v)
    return (idx[0] if np.isfinite(vals[idx[0]]) else None), (idx[1] if len(idx) > 1 and np.isfinite(vals[idx[1]]) else None)


def tab_main(sc, cc, arms, dss, out):
    """Datasets x {ACC, BWT} for the main arms, then the cost rows (means over datasets)."""
    arms = [a for a in arms if a in sc.arm.unique()]; lines = ["\\begin{tabular}{ll" + "r" * len(arms) + "}", "\\toprule",
             "Dataset & & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{short(a)}}}" for a in arms) + " \\\\", "\\midrule"]
    g = sc.groupby(["dataset", "arm"]); m = g.mean(numeric_only=True); s = g.std(numeric_only=True)
    for i, ds in enumerate(dss):
        if i > 0 and ds not in RAW and dss[i - 1] in RAW: lines.append("\\midrule")
        for key, nm, hib in (("final_acc", "ACC", True), ("bwt", "BWT", True)):
            vals = [100 * m.loc[(ds, a), key] if (ds, a) in m.index else np.nan for a in arms]; sds = [100 * s.loc[(ds, a), key] if (ds, a) in s.index else np.nan for a in arms]
            b, b2 = mark_best(vals, hib)
            lines.append((DSNAME[ds] if key == "final_acc" else "") + f" & {nm} & " + " & ".join(fmt(v, sd, j == b, j == b2) for j, (v, sd) in enumerate(zip(vals, sds))) + " \\\\")
    lines.append("\\midrule")
    costs = cc.groupby("arm").mean(numeric_only=True); life = sc.groupby("arm").lifetime_energy_uJ.mean() / 1000; mem = sc.groupby("arm").standing_bytes.mean() / 1024
    for nm, ser, nd in (("Lifetime mJ", life, 0), ("Re-admission mJ", costs.recur_uJ / 1000, 1), ("Stored KB", mem, 0)):
        vals = [ser.get(a, np.nan) for a in arms]; b, b2 = mark_best(vals, False)
        lines.append(("Cost (mean)" if nm == "Lifetime mJ" else "") + f" & {nm} & " + " & ".join(fmt(v, None, j == b, j == b2, nd) for j, v in enumerate(vals)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/main_body.tex", "w").write("\n".join(lines) + "\n")


def tab_block(sc, arms, dss, out, name, key="final_acc", scale=100, nd=1, higher=True, extra=None):
    """Arms x datasets for one metric (mean +- sd), optional extra columns (dict name -> series over arms)."""
    arms = [a for a in arms if a in sc.arm.unique()]; g = sc.groupby(["dataset", "arm"]); m = g.mean(numeric_only=True); s = g.std(numeric_only=True)
    extra = extra or {}; lines = ["\\begin{tabular}{l" + "r" * (len(dss) + len(extra)) + "}", "\\toprule",
                                  "Method & " + " & ".join(DSNAME[d] for d in dss) + ("".join(f" & {k}" for k in extra)) + " \\\\", "\\midrule"]
    cols = {ds: [scale * m.loc[(ds, a), key] if (ds, a) in m.index else np.nan for a in arms] for ds in dss}
    best = {ds: mark_best(cols[ds], higher) for ds in dss}
    for j, a in enumerate(arms):
        cells = [fmt(cols[ds][j], scale * s.loc[(ds, a), key] if (ds, a) in s.index else np.nan, j == best[ds][0], j == best[ds][1], nd) for ds in dss]
        cells += [fmt(float(v.get(a, np.nan)), None, nd=1) for v in extra.values()]
        lines.append(f"{label(a)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/{name}_body.tex", "w").write("\n".join(lines) + "\n")


def tab_cost(sc, cc, arms, dss, out):
    """Admission, maintenance overhead (energy per chunk above fine-tuning) and re-admission per method, mean over datasets, plus
    the skipped share of recurrences."""
    arms = [a for a in arms if a in sc.arm.unique()]; g = cc.groupby(["dataset", "arm"]).mean(numeric_only=True)
    naive = {ds: g.loc[(ds, "naive"), "intro_uJ"] if (ds, "naive") in g.index else np.nan for ds in dss}
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method & Adm. & Maint. & Re-adm. & Skipped \\\\", "\\midrule"]
    for a in arms:
        intro = np.nanmean([g.loc[(ds, a), "intro_uJ"] for ds in dss if (ds, a) in g.index]) / 1000
        recur = np.nanmean([g.loc[(ds, a), "recur_uJ"] for ds in dss if (ds, a) in g.index]) / 1000
        maint = (0.0 if GROUP.get(a) != "baseline" or a == "naive" else np.nanmean([g.loc[(ds, a), "intro_uJ"] - naive[ds] for ds in dss if (ds, a) in g.index]) / 1000)
        sk = np.nanmean([g.loc[(ds, a), "skipped"] for ds in dss if (ds, a) in g.index])
        lines.append(f"{label(a)} & {intro:.1f} & {'0 (by constr.)' if GROUP.get(a) != 'baseline' else ('--' if a == 'naive' else f'{maint:.1f}')} & {recur:.1f} & {100 * sk:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/cost_body.tex", "w").write("\n".join(lines) + "\n")


def tab_gate(sc, cc, dss, out):
    arms = [a for a in ("rewire", "rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128") if a in sc.arm.unique()]
    g = sc.groupby(["dataset", "arm"]).mean(numeric_only=True); c = cc.groupby(["dataset", "arm"]).mean(numeric_only=True)
    lines = ["\\begin{tabular}{ll" + "r" * len(arms) + "}", "\\toprule", "Dataset & & " + " & ".join(short(a) for a in arms) + " \\\\", "\\midrule"]
    for ds in dss:
        for key, nm, src, sc_ in (("final_acc", "ACC", g, 100), ("recur_uJ", "mJ / re-adm.", c, 1e-3), ("skipped", "skipped", c, 100)):
            vals = [sc_ * src.loc[(ds, a), key] if (ds, a) in src.index else np.nan for a in arms]
            cells = [(f"{v:.0f}\\%" if key == "skipped" else f"{v:.1f}") if np.isfinite(v) else "--" for v in vals]
            lines.append((DSNAME[ds] if key == "final_acc" else "") + f" & {nm} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/gate_body.tex", "w").write("\n".join(lines) + "\n")


def tab_full(sc, cc, dss, out):
    g = sc.groupby(["dataset", "arm"]).mean(numeric_only=True); s = sc.groupby(["dataset", "arm"]).std(numeric_only=True); c = cc.groupby(["dataset", "arm"]).mean(numeric_only=True)
    for ds in dss:
        arms = order([a for a in sc[sc.dataset == ds].arm.unique()])
        lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule", "Method & ACC & BWT & AUROC$_{\\rm stream}$ & AUROC$_{\\rm test}$ & Lifetime mJ & Re-adm. mJ & Stored KB \\\\", "\\midrule"]
        accs = [100 * g.loc[(ds, a), "final_acc"] for a in arms]; b, b2 = mark_best(accs, True)
        for j, a in enumerate(arms):
            r = g.loc[(ds, a)]; sd = s.loc[(ds, a)] if (ds, a) in s.index else None; k = (ds, a)
            lines.append(f"{label(a)} & {fmt(100 * r.final_acc, 100 * sd.final_acc if sd is not None else None, j == b, j == b2)} & {100 * r.bwt:.1f} & {r.recog_auroc_fam:.2f} & {r.fam_auroc_test:.2f} & "
                         f"{r.lifetime_energy_uJ / 1000:.0f} & {c.loc[k, 'recur_uJ'] / 1000 if k in c.index else np.nan:.1f} & {r.standing_bytes / 1024:.0f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/full_{ds}_body.tex", "w").write("\n".join(lines) + "\n")


def tab_fidelity(run, out):
    p = f"{run}/fidelity/fidelity_gate.csv"
    if not os.path.isfile(p): return
    f = pd.read_csv(p); lines = ["\\begin{tabular}{llrrl}", "\\toprule", "Protocol & Method & ACC & BWT & Verdict \\\\", "\\midrule"]
    for _, r in f.iterrows():
        thr = (-0.05, 0.90) if r.protocol == "split" else (-0.10, 0.80)
        verdict = "reference" if r.arm == "naive" else ("pass" if (r.bwt >= thr[0] and r.acc >= thr[1]) else "fail")
        lines.append(f"{'Split-MNIST' if r.protocol == 'split' else 'Permuted MNIST'} & {label(r.arm)} & {100 * r.acc:.1f}{{\\tiny$\\pm${100 * r.acc_se:.1f}}} & {100 * r.bwt:.1f}{{\\tiny$\\pm${100 * r.bwt_se:.1f}}} & {verdict} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/fidelity_body.tex", "w").write("\n".join(lines) + "\n")


def tab_paired(run, out):
    p = f"{run}/pooled_final_acc.csv"; q = f"{run}/paired_final_acc.csv"
    if not os.path.isfile(p): return
    t = pd.read_csv(p); lines = ["\\begin{tabular}{lrrrrr}", "\\toprule", "REWIRE minus & mean diff. (pts) & 95\\% CI & wins & $p$ (Holm) & pairs \\\\", "\\midrule"]
    for _, r in t.sort_values("mean_diff", ascending=False).iterrows():
        lines.append(f"{label(r.arm)} & {100 * r.mean_diff:+.1f} & [{100 * r.ci_lo:+.1f}, {100 * r.ci_hi:+.1f}] & {100 * r.win_rate:.0f}\\% & {r.p_holm:.3f}{'*' if r.p_holm < 0.05 else ''} & {int(r.n_pairs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/paired_body.tex", "w").write("\n".join(lines) + "\n")
    if os.path.isfile(q):
        d = pd.read_csv(q); lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Dataset & REWIRE minus & mean diff. (pts) & 95\\% CI & wins \\\\", "\\midrule"]
        for ds in [x for x in DS if x in d.dataset.unique()]:
            for _, r in d[d.dataset == ds].sort_values("mean_diff", ascending=False).iterrows():
                lines.append(f"{DSNAME[ds]} & {label(r.arm)} & {100 * r.mean_diff:+.1f} & [{100 * r.ci_lo:+.1f}, {100 * r.ci_hi:+.1f}] & -- \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]; open(f"{out}/paired_by_dataset_body.tex", "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------------------ macros
def camel(s):
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s)); words = s.split()
    return "".join(w[0].upper() + w[1:] for w in words) if words else "X"


NUMWORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
def texname(*parts):
    name = "".join(camel(p) for p in parts)
    return re.sub(r"\d", lambda m: NUMWORDS[m.group(0)], name)


def macros(sc, cc, dss, out):
    g = sc.groupby(["dataset", "arm"]).mean(numeric_only=True); s = sc.groupby(["dataset", "arm"]).std(numeric_only=True); c = cc.groupby(["dataset", "arm"]).mean(numeric_only=True)
    num = {}; lines = ["% generated by make_plots.py -- every number the paper quotes; do not edit by hand"]
    def put(name, val, nd=1):
        if val is None or (isinstance(val, float) and not np.isfinite(val)): return
        txt = f"{val:.{nd}f}"; txt = txt[1:] if txt.startswith("-") and float(txt) == 0 else txt          # no "-0.0"
        num[name] = round(float(val), 3); lines.append(f"\\newcommand{{\\{name}}}{{{txt}}}")
    for ds in dss:
        for a in sc[sc.dataset == ds].arm.unique():
            r = g.loc[(ds, a)]; k = (ds, a)
            put(texname("Acc", a, ds), 100 * r.final_acc); put(texname("AccSd", a, ds), 100 * s.loc[k, "final_acc"] if k in s.index else np.nan)
            put(texname("Bwt", a, ds), 100 * r.bwt); put(texname("Energy", a, ds), r.lifetime_energy_uJ / 1000, 0); put(texname("Readm", a, ds), c.loc[k, "recur_uJ"] / 1000 if k in c.index else np.nan)
            put(texname("Auroc", a, ds), r.recog_auroc_fam, 2); put(texname("AurocTest", a, ds), r.fam_auroc_test, 2); put(texname("Stored", a, ds), r.standing_bytes / 1024, 0)
            put(texname("Skipped", a, ds), 100 * c.loc[k, "skipped"] if k in c.index else np.nan, 0)
        if ("rewire" in sc[sc.dataset == ds].arm.unique()):
            for b in ("hlop", "replay", "replay_iso", "ctrl_slda_input", "ewc", "rewire_v1"):
                if (ds, b) in g.index:
                    put(texname("Gain", b, ds), 100 * (g.loc[(ds, "rewire"), "final_acc"] - g.loc[(ds, b), "final_acc"]))
                    put(texname("EnergyRatio", b, ds), g.loc[(ds, b), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"])
                    if (ds, b) in c.index and (ds, "rewire") in c.index: put(texname("ReadmRatio", b, ds), c.loc[(ds, b), "recur_uJ"] / c.loc[(ds, "rewire"), "recur_uJ"])
    # means over datasets
    for a in sc.arm.unique():
        sub = sc[sc.arm == a]; put(texname("Acc", a, "Mean"), 100 * sub.final_acc.mean()); put(texname("Energy", a, "Mean"), sub.lifetime_energy_uJ.mean() / 1000, 0)
        if a in cc.arm.unique(): put(texname("Readm", a, "Mean"), cc[cc.arm == a].recur_uJ.mean() / 1000)
    if "rewire" in sc.arm.unique():
        for b in ("hlop", "replay", "replay_iso", "ctrl_slda_input", "ewc", "rewire_v1"):
            if b in sc.arm.unique():
                ratios = [g.loc[(ds, b), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"] for ds in dss if (ds, b) in g.index and (ds, "rewire") in g.index]
                rr = [c.loc[(ds, b), "recur_uJ"] / c.loc[(ds, "rewire"), "recur_uJ"] for ds in dss if (ds, b) in c.index and (ds, "rewire") in c.index]
                if ratios: put(texname("EnergyRatio", b, "Min"), min(ratios)); put(texname("EnergyRatio", b, "Max"), max(ratios))
                if rr: put(texname("ReadmRatio", b, "Min"), min(rr)); put(texname("ReadmRatio", b, "Max"), max(rr))
    for a in ("rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128"):           # gate variants relative to REWIRE
        if a in sc.arm.unique() and "rewire" in sc.arm.unique():
            ds_ok = [ds for ds in dss if (ds, a) in g.index and (ds, "rewire") in g.index]
            for ds in ds_ok:
                put(texname("Delta", a, ds), 100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, "rewire"), "final_acc"]))
                put(texname("Saved", a, ds), 100 * (1 - g.loc[(ds, a), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"]), 0)
            if ds_ok:
                put(texname("Delta", a, "Mean"), float(np.mean([100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, "rewire"), "final_acc"]) for ds in ds_ok])))
                put(texname("Saved", a, "Mean"), float(np.mean([100 * (1 - g.loc[(ds, a), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"]) for ds in ds_ok])), 0)
    for a, parent in (("hlop_gated", "hlop"), ("replay_gated", "replay"), ("rewire_v1_gated", "rewire_v1")):          # the gate rule on other learners
        ds_ok = [ds for ds in dss if (ds, a) in g.index and (ds, parent) in g.index]
        for ds in ds_ok:
            put(texname("Delta", a, ds), 100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, parent), "final_acc"]))
            put(texname("Saved", a, ds), 100 * (1 - g.loc[(ds, a), "lifetime_energy_uJ"] / g.loc[(ds, parent), "lifetime_energy_uJ"]), 0)
        if ds_ok:
            put(texname("Delta", a, "Mean"), float(np.mean([100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, parent), "final_acc"]) for ds in ds_ok])))
            put(texname("Saved", a, "Mean"), float(np.mean([100 * (1 - g.loc[(ds, a), "lifetime_energy_uJ"] / g.loc[(ds, parent), "lifetime_energy_uJ"]) for ds in ds_ok])), 0)
    for ds in dss:                                                                                                  # the ablation ladder, in points
        for a, b, nm in (("rewire", "ctrl_relu", "LifVsRelu"), ("rewire", "abl_sketch_lda", "InputView"), ("rewire", "abl_dualview_ridge", "LdaHead"), ("abl_sketch_lda", "rewire_v1", "LdaVsRidge"),
                         ("rewire", "abl_width128", "SketchFull"), ("rewire", "abl_select_errors", "WriteAll"), ("abl_v1_select_errors", "rewire_v1", "VOneSelectErrors"), ("abl_select_errors", "rewire", "SelectErrors")):
            if (ds, a) in g.index and (ds, b) in g.index: put(texname("Ladder", nm, ds), 100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, b), "final_acc"]))
    for ds in dss:                                                                                                  # SLDA-on-the-inputs energy relative to REWIRE
        if (ds, "ctrl_slda_input") in g.index and (ds, "rewire") in g.index: put(texname("EnergyRatioSldaOverRewire", ds), g.loc[(ds, "ctrl_slda_input"), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"], 2)
    pooled = os.path.join(RUN_DIR[0], "pooled_final_acc.csv")                                                            # the pooled paired tests
    if os.path.isfile(pooled):
        t = pd.read_csv(pooled); sig = int((t.p_holm < 0.05).sum()); put("PooledSignificant", float(sig), 0); put("PooledArms", float(len(t)), 0)
        for _, r in t.iterrows():
            put(texname("PooledDiff", r.arm), 100 * r.mean_diff); put(texname("PooledWins", r.arm), 100 * r.win_rate, 0)
            num[texname("PooledP", r.arm)] = round(float(r.p_holm), 4); lines.append(f"\\newcommand{{\\{texname('PooledP', r.arm)}}}{{{'<0.001' if r.p_holm < 0.0005 else '=' + format(r.p_holm, '.3f')}}}")   # for "$p\\PooledP...$"
        put("PooledPairs", float(t.n_pairs.max()), 0)
    put("NSeeds", float(sc.groupby(["dataset", "arm"]).size().max()), 0); put("NRuns", float(len(sc)), 0); put("NArms", float(sc.arm.nunique()), 0)
    open(f"{out}/macros.tex", "w").write("\n".join(lines) + "\n"); json.dump(num, open(f"{out}/numbers.json", "w"), indent=1); return num

# ------------------------------------------------------------------------------------------------------------ hypotheses
def summary(sc, cc, dss, num, out):
    g = sc.groupby(["dataset", "arm"]).mean(numeric_only=True); c = cc.groupby(["dataset", "arm"]).mean(numeric_only=True); L = ["# What the numbers support", ""]
    L += ["| hypothesis | " + " | ".join(DSNAME[d] for d in dss) + " |", "|---|" + "---|" * len(dss)]
    grads = [a for a in ("hlop", "hlop_adaptive", "replay", "replay_iso", "ewc") if a in sc.arm.unique()]
    def row(name, f): L.append(f"| {name} | " + " | ".join(f(ds) for ds in dss) + " |")
    def h1(ds):
        if (ds, "rewire") not in g.index or not grads: return "n/a"
        best = max(g.loc[(ds, a), "final_acc"] for a in grads if (ds, a) in g.index); cheap = min(g.loc[(ds, a), "lifetime_energy_uJ"] for a in grads if (ds, a) in g.index)
        d = 100 * (g.loc[(ds, "rewire"), "final_acc"] - best); e = g.loc[(ds, "rewire"), "lifetime_energy_uJ"] / cheap
        return f"{'pass' if d >= -2 and e <= 0.5 else 'fail'} ({d:+.1f} pts, {e:.2f}x energy)"
    def h2(ds):
        if (ds, "rewire") not in c.index or not grads: return "n/a"
        cheap = min(c.loc[(ds, a), "recur_uJ"] for a in grads if (ds, a) in c.index); r = c.loc[(ds, "rewire"), "recur_uJ"] / cheap
        return f"{'pass' if r <= 0.5 else 'fail'} ({r:.2f}x of the cheapest gradient arm)"
    def h3(ds):
        if (ds, "rewire") not in g.index: return "n/a"
        return f"BWT {100 * g.loc[(ds, 'rewire'), 'bwt']:+.1f}" + (f" vs HLOP {100 * g.loc[(ds, 'hlop'), 'bwt']:+.1f}" if (ds, "hlop") in g.index else "") + (f", rehearsal {100 * g.loc[(ds, 'replay'), 'bwt']:+.1f}" if (ds, "replay") in g.index else "")
    def h4(ds):
        if (ds, "rewire") not in g.index: return "n/a"
        v = g.loc[(ds, "rewire"), "recog_auroc_fam"]; return f"{'pass' if v >= 0.9 else 'fail'} (AUROC {v:.2f})"
    def h5(ds):
        parts = []
        for a, b, nm in (("rewire", "ctrl_relu", "LIF vs ReLU"), ("rewire", "abl_sketch_lda", "+input view"), ("rewire", "abl_dualview_ridge", "+LDA head"), ("abl_sketch_lda", "rewire_v1", "LDA vs ridge")):
            if (ds, a) in g.index and (ds, b) in g.index: parts.append(f"{nm} {100 * (g.loc[(ds, a), 'final_acc'] - g.loc[(ds, b), 'final_acc']):+.1f}")
        return "; ".join(parts) or "n/a"
    def h6(ds):
        best = None
        for a in ("rewire_gated", "rewire_gated_maha", "rewire_gated_t095", "rewire_gated_b128"):
            if (ds, a) in g.index and (ds, "rewire") in g.index:
                d = 100 * (g.loc[(ds, a), "final_acc"] - g.loc[(ds, "rewire"), "final_acc"]); e = 1 - g.loc[(ds, a), "lifetime_energy_uJ"] / g.loc[(ds, "rewire"), "lifetime_energy_uJ"]
                if d >= -2 and (best is None or e > best[1]): best = (a, e, d)
        return "n/a" if best is None else f"{'pass' if best[1] >= 0.4 else 'fail'} ({short(best[0])}: {100 * best[1]:.0f}% energy saved at {best[2]:+.1f} pts)"
    row("H1 within 2 pts of the best gradient arm at <= 50% of the cheapest one's lifetime energy", h1)
    row("H2 re-admission at <= 50% of the cheapest gradient arm", h2); row("H3 retention (BWT; 0 = nothing lost)", h3)
    row("H4 stream recognition AUROC >= 0.90", h4); row("H5 ablation ladder (accuracy points)", h5); row("H6 a gate setting saves >= 40% energy at <= 2 pts", h6)
    L += ["", "## Headline numbers", ""]
    for b in ("hlop", "replay", "replay_iso", "ctrl_slda_input", "rewire_v1"):
        ks = [k for k in num if k.startswith(texname("Gain", b, ""))]
        if ks: L.append(f"- REWIRE minus {label(b)} (accuracy points): " + ", ".join(f"{k.replace(texname('Gain', b, ''), '')} {num[k]:+.1f}" for k in ks))
    for b in ("hlop", "replay"):
        if texname("EnergyRatio", b, "Min") in num: L.append(f"- lifetime energy of {label(b)} / REWIRE: {num[texname('EnergyRatio', b, 'Min')]:.1f}x to {num[texname('EnergyRatio', b, 'Max')]:.1f}x; re-admission {num.get(texname('ReadmRatio', b, 'Min'), float('nan')):.1f}x to {num.get(texname('ReadmRatio', b, 'Max'), float('nan')):.1f}x")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--out", default=None); ap.add_argument("--style", default="ieee", choices=list(STYLE))
    ap.add_argument("--gate", default="rewire_gated", help="the gate variant shown in the main table and the figures")
    ap.add_argument("--main-arms", nargs="*", default=None, help="the columns of the main table (default: the two proposed variants, the controls, the baselines)")
    ap.add_argument("--fig-arms", nargs="*", default=None, help="the methods drawn in the figures (default: a fixed set of eight for the Pareto panels, six for the line charts)")
    ap.add_argument("--stream-dataset", default="shd"); ap.add_argument("--sweep-dataset", default="cifar100"); ap.add_argument("--zip", action="store_true")
    ap.add_argument("--retention-datasets", nargs="*", default=["cifar100", "shd"], help="the datasets of the column-width retention figure (fig_retention_all has every dataset)")
    a = ap.parse_args(); run = a.run.rstrip("/"); out = a.out or f"assets_{time.strftime('%Y-%m-%d_%H%M')}"; st = STYLE[a.style]; style(st["font"]); RUN_DIR[0] = run
    tabs, figs = f"{out}/tables", f"{out}/figs"; os.makedirs(tabs, exist_ok=True); os.makedirs(figs, exist_ok=True)
    rows, sc = load_run(run); cc = chunk_costs(rows); dss = [d for d in DS + ["synthetic"] if d in sc.dataset.unique()]
    if a.gate != "rewire_gated":
        for lst in (PARETO_ARMS, LINE_ARMS): lst[:] = [a.gate if x == "rewire_gated" else x for x in lst]
    if a.fig_arms: PARETO_ARMS[:] = list(a.fig_arms); LINE_ARMS[:] = list(a.fig_arms); RET_ARMS[:] = [x for x in a.fig_arms if x != a.gate]
    main_arms = a.main_arms or [x for x in ("rewire", a.gate, "ctrl_slda_input", "ranpac", "hlop", "replay", "replay_iso", "ewc", "naive") if x in sc.arm.unique()]
    abl_all = [x for x in ("rewire_v1", "abl_sketch_lda", "abl_dualview_ridge", "rewire", "abl_width128", "abl_width256", "abl_select_errors", "abl_select_random", "abl_v1_select_errors", "abl_v1_select_random", "abl_v1_woodbury", "ctrl_relu", "ctrl_slda_input", "ridge_input", "ranpac", "hd_memory") if x in sc.arm.unique()]
    abl = [x for x in abl_all if x not in ("abl_v1_select_errors", "abl_v1_select_random", "abl_v1_woodbury", "ridge_input", "hd_memory")]   # the paper's table; the rest in ablation_full
    mj = {"mJ": sc.groupby("arm").lifetime_energy_uJ.mean() / 1000}
    tab_main(sc, cc, main_arms, dss, tabs); tab_block(sc, abl, dss, tabs, "ablation", extra=mj); tab_block(sc, abl_all, dss, tabs, "ablation_full", extra=mj)
    tab_cost(sc, cc, main_arms, dss, tabs); tab_gate(sc, cc, dss, tabs); tab_block(sc, main_arms, dss, tabs, "recognition", key="recog_auroc_fam", scale=1, nd=2)
    tab_full(sc, cc, dss, tabs); tab_fidelity(run, tabs); tab_paired(run, tabs); num = macros(sc, cc, dss, out)
    ret_ds = [d for d in a.retention_datasets if d in dss] or dss[:2]
    fig_pareto(sc, dss, figs, st); fig_retention(rows, ret_ds, figs, st, "fig_retention", st["col"]); fig_retention(rows, dss, figs, st, "fig_retention_all")
    fig_chunk_cost(cc, figs, st); fig_gate(run, sc, dss, figs, st); fig_ablation(sc, dss, figs, st)
    for ds in dss: fig_stream(rows, ds, figs, st); fig_sweep(run, ds, figs, st)
    for kind, ds in (("stream", a.stream_dataset), ("sweep", a.sweep_dataset)):                 # the paper's copies, under a fixed name
        for ext in ("pdf", "png", "eps"):
            if os.path.isfile(f"{figs}/fig_{kind}_{ds}.{ext}"): shutil.copy(f"{figs}/fig_{kind}_{ds}.{ext}", f"{figs}/fig_{kind}.{ext}")
    if SWEEP_NUM:                                                                                  # sweep macros, appended once the figures exist
        lines = []
        for (ds, arm), v in sorted(SWEEP_NUM.items()):
            for key, val, nd in (("SweepEnergyGrowth", v["energy_ratio"], 1), ("SweepAccChange", v["acc_change"], 1)):
                name = texname(key, arm, ds); num[name] = round(float(val), 3); lines.append(f"\\newcommand{{\\{name}}}{{{val:.{nd}f}}}")
            for key, val in (("SweepNMin", v["n_min"]), ("SweepNMax", v["n_max"])):
                name = texname(key, ds); num[name] = val; lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
        seen, uniq = set(), []
        for l in lines:
            if l not in seen: seen.add(l); uniq.append(l)
        open(f"{out}/macros.tex", "a").write("\n".join(uniq) + "\n"); json.dump(num, open(f"{out}/numbers.json", "w"), indent=1)
    summary(sc, cc, dss, num, out)
    shutil.copy(f"{run}/MANIFEST.json", out) if os.path.isfile(f"{run}/MANIFEST.json") else None
    print(f"\nassets in {out}/ ({len(os.listdir(tabs))} tables, {len(glob.glob(figs + '/*.pdf'))} figures, {len(num)} macros)")
    if a.zip: z = shutil.make_archive(out, "zip", root_dir=os.path.dirname(os.path.abspath(out)), base_dir=os.path.basename(out)); print("archive:", z)


if __name__ == "__main__": main()
