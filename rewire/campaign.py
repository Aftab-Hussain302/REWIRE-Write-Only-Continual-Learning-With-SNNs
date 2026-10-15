"""One dataset's campaign: the main grid (every arm x every seed), the recurrence-density sweep and the gate sweep, each in its own
folder with a rows.json (resumable), plus the cross-dataset leaderboard, paired tests and figures. Used by run_campaign.py (one
dataset per call) and by scripts/merge_results.py (reports over a finished run)."""
import glob, json, os, time, numpy as np, pandas as pd
from .configure import DEFAULT, apply_dataset_defaults
from .dataloader import load_dataset
from .engine import run_grid
from .utils.plotting import plot_all, plot_sweep, plot_gate_sweep, plot_campaign

REFERENCE = "rewire"                                                                  # the paired tests compare every arm with the proposed method
SWEEP_ARMS = ["rewire", "rewire_gated", "rewire_v1", "ctrl_slda_input", "ranpac", "ridge_input", "hd_memory", "naive", "ewc", "replay", "replay_iso", "hlop", "hlop_adaptive"]
SWEEP_ARMS_BIG = ["rewire", "rewire_gated", "rewire_v1", "ctrl_slda_input", "ranpac", "ridge_input", "hd_memory", "naive", "replay", "replay_iso", "hlop"]   # 80-class streams
GATED = ("rewire_gated", "rewire_gated_maha", "rewire_v1_gated", "hlop_gated", "replay_gated")                                           # the gate sweep


def _grid(cfg, X, y, out, log, **over):
    c = dict(cfg); c.update(over); c["_overrides"] = {**c.get("_overrides", {}), **over}; c["out"] = out; os.makedirs(out, exist_ok=True)   # the grid's own settings win over --set
    rows = run_grid(c, log, X, y); return rows, c


def sweep_arms(cfg): return [a for a in (SWEEP_ARMS_BIG if cfg["n_stream_classes"] >= 40 else SWEEP_ARMS) if a in cfg["arms"]]


def run_campaign(cfg, datasets=("mnist", "cifar10", "cifar100", "nmnist", "shd", "tinyimagenet"), shifts=("none",), recur_sweep=True, gate_sweep=True, log=print):
    root = cfg["out"]; os.makedirs(root, exist_ok=True); t0 = time.time(); main_rows = []; sweep_rows = []; gate_rows = []
    for ds in datasets:
        dc = apply_dataset_defaults(cfg, ds)
        X, y = load_dataset(ds, dc["data"], dc["data_root"], dc)
        for sh in shifts:
            rows, c = _grid(dc, X, y, f"{root}/{ds}/shift_{sh}", log, dataset=ds, shift_mode=sh); plot_all(rows, c); main_rows += rows
        if recur_sweep:
            arms_sw = sweep_arms(dc)
            for n in dc["sweep_recur"]:
                if n == dc["n_recur"] and "none" in shifts: rows = [r for r in main_rows if r["dataset"] == ds and r["shift_mode"] == "none" and r["arm"] in arms_sw]
                else: rows, c = _grid(dc, X, y, f"{root}/{ds}/sweep_recur/n{n}", log, dataset=ds, shift_mode="none", n_recur=n, arms=arms_sw)
                sweep_rows += rows
            c = dict(dc); c["out"] = f"{root}/{ds}/sweep_recur"; plot_sweep([r for r in sweep_rows if r["dataset"] == ds], c)
        if gate_sweep:
            arms_g = [a for a in dc["arms"] if a in GATED]
            for th in cfg["sweep_theta"]:
                if th == dc["theta_hi"] and "none" in shifts: rows = [r for r in main_rows if r["dataset"] == ds and r["shift_mode"] == "none" and r["arm"] in arms_g]
                elif arms_g: rows, c = _grid(dc, X, y, f"{root}/{ds}/sweep_gate/t{th}", log, dataset=ds, shift_mode="none", arms=arms_g, seeds=dc["seeds"][:3], theta_hi=th)
                else: rows = []
                gate_rows += rows
            plot_gate_sweep([r for r in gate_rows if r["dataset"] == ds], [r for r in main_rows if r["dataset"] == ds and r["shift_mode"] == "none"], f"{root}/{ds}/sweep_gate", ds)
    if not cfg.get("parallel_worker", False): reports(root, main_rows, log)
    log(f"[campaign] finished in {(time.time() - t0) / 60:.1f} min; results in {root}/")
    return main_rows, sweep_rows, gate_rows


def reports(root, rows, log=print):
    """Cross-dataset figures, the campaign leaderboard and the paired tests against the proposed method."""
    datasets = sorted({r["dataset"] for r in rows}); shifts = sorted({r["shift_mode"] for r in rows})
    plot_campaign(rows, root, datasets, shifts)
    from .metrics import paired_table, pooled_table
    for m in ("final_acc", "lifetime_energy_uJ", "bwt", "S_AUR", "energy_per_acc_point"):
        paired_table(rows, m, REFERENCE, root)                      # per dataset (descriptive with five seeds)
        t = pooled_table(rows, m, REFERENCE, root)                  # pooled over dataset x seed pairs: the powered test
        if len(t): log(f"[campaign] pooled paired test vs {REFERENCE} on {m}: {int((t.p_holm < 0.05).sum())}/{len(t)} arms differ at p<0.05 (Holm)")


def rebuild_leaderboard(root):
    """Rebuild the campaign leaderboard, figures and paired tests from the saved rows.json files, without re-running anything."""
    rows = []
    for f in glob.glob(f"{root}/*/shift_*/rows.json"): rows += json.load(open(f))
    if not rows: raise FileNotFoundError(f"no rows.json under {root}")
    reports(root, rows)
    print(f"rebuilt {root}/campaign_leaderboard.csv from {len(rows)} runs ({len({r['dataset'] for r in rows})} datasets)")
    return rows
