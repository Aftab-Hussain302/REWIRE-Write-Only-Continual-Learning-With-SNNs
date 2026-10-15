"""Figures and summary tables written next to every rows.json. One visual language everywhere: colour = group of the arm
(proposed / ablation / control / gradient baseline), marker = the arm, filled markers for the primary arms and open ones for
variants, REWIRE always drawn last and on top. Every figure is saved as PNG (300 dpi), PDF and EPS."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from ..models.arms import ARMS, GROUP, LABEL

# ---- colour by group (a colour-vision-safe set: blue, red, green, violet, grey), marker by arm
GROUP_COLOR = {"proposed": "#2a78d6", "ablation": "#2a78d6", "control": "#008300", "baseline": "#e34948", "fidelity": "#777570"}
COLOR = {"replay": "#4a3aa7", "replay_iso": "#4a3aa7", "replay_gated": "#4a3aa7", "ewc": "#4a3aa7", "naive": "#777570",
         "hlop": "#e34948", "hlop_adaptive": "#e34948", "hlop_gated": "#e34948"}
MARK = {"rewire": "*", "rewire_gated": "P", "rewire_gated_maha": "p", "rewire_gated_t095": "h", "rewire_gated_b128": "H",
        "rewire_v1": "o", "rewire_v1_gated": "D", "abl_sketch_lda": "v", "abl_dualview_ridge": "^", "abl_width128": "<", "abl_width256": ">",
        "abl_select_errors": "x", "abl_select_random": "+", "abl_v1_select_errors": "x", "abl_v1_select_random": "+", "abl_v1_woodbury": "1",
        "ctrl_slda_input": "s", "ctrl_relu": "D", "ranpac": "o", "ridge_input": "v", "hd_memory": "^",
        "hlop": "s", "hlop_adaptive": "D", "hlop_gated": "o", "replay": "s", "replay_iso": "D", "replay_gated": "o", "ewc": "^", "naive": "X", "dsd_snn": "v"}
FILLED = {"rewire", "rewire_gated", "hlop", "replay", "ewc", "naive", "ctrl_slda_input", "rewire_v1"}   # primary arms; the rest are open markers
ORDER = list(ARMS)                                                                                       # the paper's order


def _col(a): return COLOR.get(a, GROUP_COLOR.get(GROUP.get(a, ""), "#777570"))
def _mark(a): return MARK.get(a, "o")
def _label(a): return LABEL.get(a, a)
def _open(a): return a not in FILLED
def _order(arms): return [a for a in ORDER if a in arms] + sorted(a for a in arms if a not in ORDER)


def _style():
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": True,
                         "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.4, "axes.axisbelow": True,
                         "lines.linewidth": 1.3, "errorbar.capsize": 2})


def _save(fig, out, name):
    import logging; logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{name}.png", dpi=300); fig.savefig(f"{out}/{name}.pdf")
    try: fig.savefig(f"{out}/{name}.eps", format="eps")
    except Exception: pass
    plt.close(fig)


def point(ax, x, y, a, xerr=None, yerr=None, size=None):
    """One arm as a point: filled for primary arms, open (white face, coloured edge) for variants; REWIRE largest and on top."""
    op = _open(a); ms = size or (10 if a == "rewire" else 6.5 if op else 5.5)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=_mark(a), color=_col(a), ms=ms, mfc="white" if op else _col(a), mec=_col(a) if op else "black",
                mew=1.0 if op else 0.4, elinewidth=0.6, capsize=1.5, label=_label(a), zorder=8 if a == "rewire" else (4 if op else 6))


def log_ticks(ax, lo, hi, axis="x"):
    """Readable ticks on a log axis: 1-2-5 steps, plain numbers (k for thousands), no minor labels."""
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.01); a = ax.xaxis if axis == "x" else ax.yaxis
    for mult in ((1,), (1, 3), (1, 2, 5)):
        t = [m * 10 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1) for m in mult if lo <= m * 10 ** e <= hi]
        if len(t) >= 2: break
    if len(t) > 6: t = t[::2]
    a.set_major_locator(FixedLocator(t)); a.set_major_formatter(FixedFormatter([f"{v/1000:g}k" if v >= 1000 else f"{v:g}" for v in t])); a.set_minor_formatter(NullFormatter())


def legend_below(fig, ax, ncol=4, pad=0.16):
    """One legend for the whole figure, below every panel; the constrained layout reserves the room for it (an axes-anchored legend
    would make the layout engine shrink the panels instead). Duplicate labels collapsed; `pad` is kept for the call sites."""
    seen = {}
    for a in fig.axes:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    if not seen: return
    try: fig.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.0)
    except ValueError: fig.legend(seen.values(), seen.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=ncol, frameon=False)   # matplotlib < 3.7


def chunk_costs(rows):
    """Per-chunk energy (uJ) by kind from the cumulative energy curve and the visit log: admission (intro chunks) and re-admission
    (recurrence chunks), and the share of recurrences whose write was skipped."""
    out = []
    for r in rows:
        e = np.asarray(r["energy_curve"], float); inc = np.diff(np.concatenate([[0.0], e]))
        v = r["visits"]; kinds = np.array([x["kind"] for x in v]); wrote = np.array([x.get("wrote", True) for x in v], bool)
        rec = kinds == "recur"
        out.append(dict(arm=r["arm"], seed=r["seed"], dataset=r["dataset"], intro_uJ=float(inc[kinds == "intro"].mean()) if (kinds == "intro").any() else np.nan,
                        recur_uJ=float(inc[rec].mean()) if rec.any() else np.nan, skipped=float((~wrote[rec]).mean()) if rec.any() else np.nan))
    return pd.DataFrame(out)


def retention(rows):
    """Accuracy on the arriving class at each recurrence BEFORE anything is written (visit log field pre_comp): what the learner
    still knows when a class comes back. Returned per arm as the mean over seeds at each recurrence index."""
    cur = {}
    for r in rows:
        pre = [x["pre_comp"] for x in r["visits"] if x["kind"] == "recur"]
        cur.setdefault(r["arm"], []).append(pre)
    return {a: np.nanmean(np.array([c[:min(map(len, cs))] for c in cs]), 0) for a, cs in cur.items() if cs and min(map(len, cs)) > 0}


METRICS = [  # key, title, higher-is-better, log scale
    ("final_acc", "Final accuracy on all stream classes", True, False),
    ("bwt", "Backward transfer (0 = no change since a class was introduced)", True, False),
    ("forgetting", "Forgetting (best minus final accuracy, per class)", False, False),
    ("recog_auroc_fam", "Recognition AUROC on the stream (recurrence vs first occurrence)", True, False),
    ("fam_auroc_test", "Familiarity AUROC on the test set (seen vs never-seen classes)", True, False),
    ("S_AUR", "Recovery savings S_AUR (seen vs never-seen recovery curves)", True, False),
    ("RIF", "RIF: accuracy change of the context classes after a write", True, False),
    ("lifetime_energy_uJ", "Lifetime energy over the stream (uJ, 45 nm operation model)", False, True),
    ("energy_per_acc_point", "Energy per accuracy point (uJ per 1%)", False, True),
    ("standing_bytes", "Standing memory between chunks (bytes)", False, True),
    ("train_time_s", "Wall clock for the stream (s)", False, True),
]


def plot_all(rows, cfg):
    _style(); out = cfg["out"]; os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows); arms = _order([a for a in df.arm.unique()]); tag = f" [{cfg.get('dataset', '')}]"
    scalar = df.drop(columns=[c for c in ["acc_curve", "energy_curve", "steps_curve", "visits", "per_class_curve", "stream", "holdout", "censored_detail",
                                          "state_counts", "rec_curve_seen", "rec_curve_novel", "config", "versions", "class_order"] if c in df])
    scalar.to_csv(f"{out}/metrics.csv", index=False)
    g = scalar.groupby("arm"); mean = g.mean(numeric_only=True).reindex(arms); se = (g.std(numeric_only=True) / np.sqrt(g.size().values[:, None])).reindex(arms).fillna(0)
    mean.to_csv(f"{out}/summary_mean.csv"); se.to_csv(f"{out}/summary_se.csv")
    cc = chunk_costs(rows); ccm = cc.groupby("arm").mean(numeric_only=True).reindex(arms); cc.to_csv(f"{out}/chunk_costs.csv", index=False)

    # ---- one bar figure per metric
    for key, title, hib, logy in METRICS:
        if key not in mean or mean[key].isna().all(): continue
        fig, ax = plt.subplots(figsize=(0.42 * len(arms) + 2.2, 3.2)); x = np.arange(len(arms))
        ax.bar(x, mean[key], yerr=se[key], color=[_col(a) for a in arms], alpha=0.9, edgecolor="black", linewidth=0.4, width=0.72)
        ax.set_xticks(x); ax.set_xticklabels([_label(a) for a in arms], rotation=35, ha="right", rotation_mode="anchor"); ax.set_title(title + tag)
        ax.set_ylabel(("higher" if hib else "lower") + " is better"); ax.axhline(0, color="black", lw=0.6)
        if key in ("recog_auroc_fam", "fam_auroc_test"): ax.axhline(0.9, ls="--", color="gray", lw=0.8); ax.set_ylim(0.3, 1.02)
        if logy and (mean[key] > 0).all(): ax.set_yscale("log")
        _save(fig, out, f"metric_{key}")

    # ---- accuracy over the stream
    n_intro = cfg["n_stream_classes"] - 1
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvspan(0.5, n_intro + 0.5, color="#f1f0ec", zorder=0); ax.text(1, 0.03, "admission", fontsize=7, color="#52514e"); ax.text(n_intro + 1, 0.03, "re-admission (recurrences)", fontsize=7, color="#52514e")
    for a in arms:
        curves = np.array([r["acc_curve"] for r in rows if r["arm"] == a]); m, s = curves.mean(0), curves.std(0) / np.sqrt(len(curves)); t = np.arange(1, len(m) + 1)
        ax.plot(t, m, color=_col(a), marker=_mark(a), ms=3.2 if a != "rewire" else 6, mfc="white" if _open(a) else _col(a), lw=1.6 if a == "rewire" else 1.0, alpha=1.0 if not _open(a) else 0.8,
                label=_label(a), zorder=8 if a == "rewire" else 4); ax.fill_between(t, m - s, m + s, color=_col(a), alpha=0.08)
    ax.set_xlabel("stream chunk"); ax.set_ylabel("accuracy on classes seen so far"); ax.set_ylim(0, 1.02); ax.set_title("Accuracy along the stream" + tag)
    legend_below(fig, ax, ncol=4, pad=0.17); _save(fig, out, "acc_over_stream")

    # ---- Pareto: final accuracy vs lifetime energy
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for a in arms: point(ax, mean.loc[a, "lifetime_energy_uJ"], mean.loc[a, "final_acc"], a, xerr=se.loc[a, "lifetime_energy_uJ"], yerr=se.loc[a, "final_acc"])
    pts = mean[["lifetime_energy_uJ", "final_acc"]].values; order = np.argsort(pts[:, 0]); front, best = [], -1
    for i in order:
        if pts[i, 1] > best: front.append(i); best = pts[i, 1]
    ax.plot(pts[front, 0], pts[front, 1], color="#52514e", lw=0.7, ls="--", zorder=2)
    ax.annotate("REWIRE", (mean.loc["rewire", "lifetime_energy_uJ"], mean.loc["rewire", "final_acc"]), textcoords="offset points", xytext=(7, -9), fontsize=7, color=_col("rewire"), fontweight="bold") if "rewire" in arms else None
    ax.set_xscale("log"); log_ticks(ax, mean.lifetime_energy_uJ.min() / 1.3, mean.lifetime_energy_uJ.max() * 1.3)
    ax.set_xlabel("lifetime energy of the learner (uJ, log)"); ax.set_ylabel("final accuracy"); ax.set_title("Accuracy against lifetime energy (up-left is better)" + tag)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left"); _save(fig, out, "pareto_acc_vs_energy")

    # ---- Pareto: final accuracy vs standing memory
    fig, ax = plt.subplots(figsize=(5.4, 4.0)); mem = mean.standing_bytes.replace(0, np.nan)
    for a in arms:
        if np.isfinite(mem.loc[a]): point(ax, mem.loc[a] / 1024, mean.loc[a, "final_acc"], a, yerr=se.loc[a, "final_acc"])
    ax.set_xscale("log"); ax.set_xlabel("standing memory between chunks (KB, log)"); ax.set_ylabel("final accuracy"); ax.set_title("Accuracy against standing memory" + tag)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left"); _save(fig, out, "pareto_acc_vs_memory")

    # ---- the cost of a chunk: admission vs re-admission, per arm
    fig, ax = plt.subplots(figsize=(0.5 * len(arms) + 2.0, 3.4)); x = np.arange(len(arms)); w = 0.38
    ax.bar(x - w / 2, ccm.intro_uJ / 1000, w, color=[_col(a) for a in arms], alpha=0.45, edgecolor="black", linewidth=0.4, label="admission chunk")
    ax.bar(x + w / 2, ccm.recur_uJ / 1000, w, color=[_col(a) for a in arms], alpha=0.95, edgecolor="black", linewidth=0.4, label="re-admission chunk")
    for i, a in enumerate(arms):
        if np.isfinite(ccm.skipped.loc[a]) and ccm.skipped.loc[a] > 0: ax.text(i + w / 2, ccm.recur_uJ.loc[a] / 1000 * 1.12, f"{100 * ccm.skipped.loc[a]:.0f}%\nskipped", ha="center", fontsize=5.5, va="bottom", color="#52514e")
    ax.set_yscale("log"); ax.set_ylabel("energy per chunk (mJ, log)"); ax.set_xticks(x); ax.set_xticklabels([_label(a) for a in arms], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title("What a chunk costs: admission (light) and re-admission (dark)" + tag); ax.legend(loc="upper right"); _save(fig, out, "chunk_cost")

    # ---- retention at re-admission: accuracy on the returning class before anything is written
    ret = retention(rows)
    if ret:
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        for a in arms:
            if a in ret: ax.plot(np.arange(1, len(ret[a]) + 1), ret[a], color=_col(a), marker=_mark(a), ms=3.2 if a != "rewire" else 6, mfc="white" if _open(a) else _col(a), lw=1.6 if a == "rewire" else 1.0, alpha=1.0 if not _open(a) else 0.8, label=_label(a), zorder=8 if a == "rewire" else 4)
        ax.set_xlabel("recurrence index along the stream"); ax.set_ylabel("accuracy on the returning class\nbefore its write"); ax.set_ylim(0, 1.02)
        ax.set_title("Retention at re-admission (flat = nothing to relearn)" + tag); legend_below(fig, ax, ncol=4); _save(fig, out, "retention_at_recurrence")

    # ---- cumulative energy along the stream
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for a in arms:
        curves = np.array([r["energy_curve"] for r in rows if r["arm"] == a]); m = curves.mean(0) / 1000
        ax.plot(np.arange(1, len(m) + 1), m, color=_col(a), marker=_mark(a), ms=2.8, mfc="white" if _open(a) else _col(a), lw=1.4 if a == "rewire" else 0.9, alpha=1.0 if not _open(a) else 0.8, label=_label(a), zorder=8 if a == "rewire" else 4)
    ax.axvline(n_intro + 0.5, color="#52514e", ls=":", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("stream chunk"); ax.set_ylabel("cumulative energy (mJ, log)"); ax.set_title("Cumulative lifetime energy" + tag)
    legend_below(fig, ax, ncol=4); _save(fig, out, "energy_over_stream")

    # ---- recovery curves: forgotten seen classes vs never-seen classes
    bud = [int(b) for b in rows[0]["rec_curve_seen"].keys()] if "rec_curve_seen" in rows[0] else []
    if bud:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for a in arms:
            rs = [r for r in rows if r["arm"] == a]
            ms_ = [np.mean([r["rec_curve_seen"][str(b)] for r in rs]) for b in bud]; mn = [np.mean([r["rec_curve_novel"][str(b)] for r in rs]) for b in bud]
            ax.plot(bud, ms_, color=_col(a), marker=_mark(a), ms=4, mfc="white" if _open(a) else _col(a), lw=1.3, label=_label(a)); ax.plot(bud, mn, color=_col(a), marker=_mark(a), ms=3, lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel("few-shot recovery updates"); ax.set_ylabel("balanced accuracy of the target class"); ax.set_ylim(0.3, 1.02)
        ax.set_title("Recovery: seen classes (solid) vs never-seen classes (dotted)" + tag); ax.legend(fontsize=5.5, ncol=2, loc="lower right"); _save(fig, out, "recovery_curves")

    # ---- leaderboard with the pre-registered bars
    keys = [k for k in ("final_acc", "bwt", "forgetting", "S_AUR", "RIF", "recog_auroc_fam", "fam_auroc_test", "lifetime_energy_uJ", "energy_per_acc_point", "standing_bytes", "censored") if k in mean]
    lb = mean[keys].copy(); lb["recur_uJ"] = ccm.recur_uJ; lb["intro_uJ"] = ccm.intro_uJ; lb["skipped"] = ccm.skipped
    ref = lb.loc[[a for a in ("hlop", "replay", "replay_iso", "ewc") if a in lb.index]]     # the best gradient-based maintenance arm
    if len(ref):
        lb["acc_vs_best_gradient"] = lb.final_acc - ref.final_acc.max(); lb["energy_vs_cheapest_gradient"] = lb.lifetime_energy_uJ / ref.lifetime_energy_uJ.min()
        lb["recur_vs_cheapest_gradient"] = lb.recur_uJ / ref.recur_uJ.min()
        lb["H1_cost"] = (lb.acc_vs_best_gradient >= -0.02) & (lb.energy_vs_cheapest_gradient <= 0.5)
        lb["H2_readmission"] = lb.recur_vs_cheapest_gradient <= 0.5
    lb["H4_recognition"] = lb.recog_auroc_fam >= 0.90
    lb = lb.sort_values("energy_per_acc_point"); lb.to_csv(f"{out}/leaderboard.csv")
    print("\nLEADERBOARD (mean over seeds). H1: within 2 points of the best gradient arm at <= 50% of the cheapest one's lifetime energy;"
          " H2: re-admission at <= 50% of the cheapest gradient arm; H4: recognition AUROC >= 0.90")
    print(lb.round(3).to_string())
    return df


def plot_sweep(rows, cfg):
    """Recurrence density: lifetime energy (log) and final accuracy as the number of recurrence chunks grows."""
    _style(); out = cfg["out"]; os.makedirs(out, exist_ok=True); df = pd.DataFrame(rows); arms = _order(df.arm.unique())
    g = df.groupby(["arm", "n_recur"]); m = g.mean(numeric_only=True); s = (g.std(numeric_only=True) / np.sqrt(g.size().values[:, None])).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for key, ax, ttl, scale in (("lifetime_energy_uJ", axes[0], "Lifetime energy vs recurrence density", 1e-3), ("final_acc", axes[1], "Final accuracy vs recurrence density", 1.0)):
        for a in arms:
            xs = sorted(df[df.arm == a].n_recur.unique())
            ax.errorbar(xs, [m.loc[(a, x), key] * scale for x in xs], yerr=[s.loc[(a, x), key] * scale for x in xs], color=_col(a), marker=_mark(a), ms=4.5,
                        mfc="white" if _open(a) else _col(a), lw=1.4 if a == "rewire" else 1.0, label=_label(a), zorder=8 if a == "rewire" else 4)
        ax.set_xlabel("recurrence chunks in the stream"); ax.set_title(ttl); ax.set_xticks(sorted(df.n_recur.unique()))
    axes[0].set_yscale("log"); axes[0].set_ylabel("lifetime energy (mJ, log)"); axes[1].set_ylabel("final accuracy")
    legend_below(fig, axes[1], ncol=5); _save(fig, out, "sweep_recurrence_density"); m.to_csv(f"{out}/sweep_summary.csv")


def plot_gate_sweep(gate_rows, main_rows, out, ds):
    """The recognition gate as an operating curve: accuracy, lifetime energy and skipped share against the competence threshold."""
    _style(); os.makedirs(out, exist_ok=True); df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in gate_rows])
    if df.empty: return
    df["skipped"] = [float(np.mean([not v.get("wrote", True) for v in r.get("visits", []) if v.get("kind") == "recur"] or [np.nan])) for r in gate_rows]
    g = df.groupby(["arm", "theta_hi"]).agg(final_acc=("final_acc", "mean"), acc_sd=("final_acc", "std"), lifetime_energy_uJ=("lifetime_energy_uJ", "mean"),
                                           bwt=("bwt", "mean"), skipped=("skipped", "mean"), n=("seed", "count")).reset_index(); g.to_csv(f"{out}/sweep_gate_summary.csv", index=False)
    ref = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in main_rows]); rm = ref.groupby("arm").mean(numeric_only=True) if not ref.empty else None
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for a, ga in g.groupby("arm"):
        ga = ga.sort_values("theta_hi"); base = {"rewire_gated": "rewire", "rewire_gated_maha": "rewire", "hlop_gated": "hlop", "replay_gated": "replay", "rewire_v1_gated": "rewire_v1"}.get(a)
        kw = dict(color=_col(a), marker=_mark(a), mfc="white" if _open(a) else _col(a), ms=4.5)
        axes[0].errorbar(ga.theta_hi, ga.final_acc, yerr=ga.acc_sd.fillna(0), label=_label(a), **kw)
        axes[1].plot(ga.theta_hi, ga.lifetime_energy_uJ / 1000, **kw); axes[2].plot(ga.theta_hi, 100 * ga.skipped, **kw)
        if rm is not None and base in rm.index:
            axes[0].axhline(rm.loc[base, "final_acc"], color=_col(base), ls="--", lw=0.8, label=f"{_label(base)}, no gate"); axes[1].axhline(rm.loc[base, "lifetime_energy_uJ"] / 1000, color=_col(base), ls="--", lw=0.8)
    for ax, yl in zip(axes, ("final accuracy", "lifetime energy (mJ)", "recurrences skipped (%)")): ax.set_xlabel("competence threshold $\\theta$"); ax.set_ylabel(yl)
    axes[0].set_title(f"Gate operating curve [{ds}]"); legend_below(fig, axes[0], ncol=4); _save(fig, out, "gate_sweep")


def plot_campaign(rows, root, datasets, shifts=("none",)):
    """Cross-dataset figures and the campaign leaderboard (csv + tex): Pareto small multiples and retention small multiples."""
    _style(); df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    if df.empty: return
    arms = _order(df.arm.unique()); d0 = df[df.shift_mode == "none"]; dss = [d for d in datasets if d in d0.dataset.unique()]
    ncol = min(3, len(dss)); nrow = int(np.ceil(len(dss) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    for i, ds in enumerate(dss):
        ax = axes[i // ncol, i % ncol]; m = d0[d0.dataset == ds].groupby("arm").mean(numeric_only=True); s = d0[d0.dataset == ds].groupby("arm").sem(numeric_only=True)
        for a in arms:
            if a in m.index: point(ax, m.loc[a, "lifetime_energy_uJ"], m.loc[a, "final_acc"], a, xerr=s.loc[a, "lifetime_energy_uJ"], yerr=s.loc[a, "final_acc"])
        ax.set_xscale("log"); log_ticks(ax, m.lifetime_energy_uJ.min() / 1.3, m.lifetime_energy_uJ.max() * 1.3); ax.set_title(ds)
        if i % ncol == 0: ax.set_ylabel("final accuracy")
        if i // ncol == nrow - 1: ax.set_xlabel("lifetime energy (uJ, log)")
    for j in range(len(dss), nrow * ncol): axes[j // ncol, j % ncol].axis("off")
    legend_below(fig, axes[0, 0], ncol=5); _save(fig, root, "campaign_pareto_by_dataset")
    want = ["final_acc", "bwt", "forgetting", "S_AUR", "AUR_seen", "AUR_novel", "RIF", "collapsed", "recog_auroc_fam", "fam_auroc_test", "lifetime_energy_uJ",
            "lifetime_steps", "lifetime_spikes", "energy_per_acc_point", "standing_bytes", "resident_bytes", "params", "censored", "train_time_s"]
    keys = [k for k in want if k in df]
    lb = df.groupby(["dataset", "shift_mode", "arm"])[keys].mean(); lb["n_seeds"] = df.groupby(["dataset", "shift_mode", "arm"]).size()
    lb.to_csv(f"{root}/campaign_leaderboard.csv")
    with open(f"{root}/campaign_leaderboard.tex", "w") as f: f.write(lb.round(3).to_latex())
    print("\nCAMPAIGN LEADERBOARD (mean over seeds)"); print(lb.round(3).to_string())
