"""Statistics an A*-venue reviewer expects: paired tests across seeds, bootstrap confidence intervals, effect sizes.

Every arm sees the identical stream for a given seed (same class roles, same chunk contents, same order), so seeds are PAIRED and
a paired test is the correct one. With 10 seeds the exact Wilcoxon signed-rank test is used; the bootstrap CI is reported on the
paired differences, not on the two means separately."""
import itertools, json, os, numpy as np, pandas as pd


def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank p-value (no SciPy dependency). Zeros dropped, average ranks for ties."""
    d = np.asarray([x for x in d if x != 0], float); n = len(d)
    if n == 0: return 1.0, 0.0
    order = np.argsort(np.abs(d)); ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    a = np.abs(d)
    for v in np.unique(a):                                   # average ranks for ties
        m = a == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    W = ranks[d > 0].sum()
    if n > 20:                                               # normal approximation
        mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mu) / max(sd, 1e-9); from math import erfc
        return float(erfc(abs(z) / np.sqrt(2))), float(W)
    cnt = 0; tot = 0                                         # exact enumeration over sign assignments
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s); tot += 1
        if w >= max(W, n * (n + 1) / 2 - W) or w <= min(W, n * (n + 1) / 2 - W): cnt += 1
    return float(cnt / tot), float(W)


def boot_ci(d, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_table(rows, metric="final_acc", reference="rewire", out=None):
    """Paired comparison of `reference` against every other arm, per dataset and shift mode."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]); recs = []
    for (ds, sh), g in df.groupby(["dataset", "shift_mode"]):
        piv = g.pivot_table(index="seed", columns="arm", values=metric)
        if reference not in piv: continue
        for arm in piv.columns:
            if arm == reference: continue
            pair = piv[[reference, arm]].dropna()
            if len(pair) < 3: continue
            d = (pair[reference] - pair[arm]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
            recs.append(dict(dataset=ds, shift_mode=sh, metric=metric, reference=reference, arm=arm, n_seeds=len(pair),
                             mean_ref=float(pair[reference].mean()), mean_arm=float(pair[arm].mean()), mean_diff=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, cohen_dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p_wilcoxon=p))
    t = pd.DataFrame(recs)
    if out is not None and len(t):
        os.makedirs(out, exist_ok=True); t.to_csv(f"{out}/paired_{metric}.csv", index=False)
    return t


def pooled_table(rows, metric="final_acc", reference="rewire", out=None):
    """Paired test on the POOLED dataset x seed pairs. With three seeds a per-dataset exact Wilcoxon cannot go below p=0.25, so the
    per-dataset tables are descriptive and this pooled test is the one with power: each (dataset, seed) is one matched pair, which
    gives n = n_datasets x n_seeds."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]); recs = []
    for sh, g in df.groupby("shift_mode"):
        piv = g.pivot_table(index=["dataset", "seed"], columns="arm", values=metric)
        if reference not in piv: continue
        for arm in piv.columns:
            if arm == reference: continue
            pair = piv[[reference, arm]].dropna()
            if len(pair) < 5: continue
            d = (pair[reference] - pair[arm]).values; p, _ = wilcoxon_exact(d); lo, hi = boot_ci(d)
            recs.append(dict(shift_mode=sh, metric=metric, reference=reference, arm=arm, n_pairs=len(pair),
                             mean_ref=float(pair[reference].mean()), mean_arm=float(pair[arm].mean()), mean_diff=float(d.mean()),
                             ci_lo=lo, ci_hi=hi, cohen_dz=float(d.mean() / (d.std(ddof=1) + 1e-12)), p_wilcoxon=p,
                             win_rate=float((d > 0).mean())))
    t = pd.DataFrame(recs)
    if len(t):
        t["p_holm"] = np.nan                       # per group by INDEX: np.concatenate over groupby misaligns the rows
        for _, idx in t.groupby("shift_mode").groups.items(): t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_wilcoxon"].values)
    if out is not None and len(t):
        os.makedirs(out, exist_ok=True); t.to_csv(f"{out}/pooled_{metric}.csv", index=False)
    return t


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj
