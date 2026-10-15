#!/usr/bin/env python3
"""Join the per-unit outputs of a run (<run>/parts/...) into the usual result tree and produce every figure, table and report:

    python3 scripts/merge_results.py results/results_2026-10-09_0830          # (run_all.py calls this itself at the end)

Writes, inside the run folder:
  <dataset>/shift_none/rows.json + figures + leaderboard.csv        the main grid (every arm x every seed)
  <dataset>/sweep_recur/n<N>/rows.json, sweep_recurrence_density.*  the recurrence-density sweep
  <dataset>/sweep_gate/t<theta>/rows.json, gate_sweep.*             the gate's operating curve
  fidelity/fidelity_gate.csv                                        the baselines on their native protocols
  campaign_leaderboard.csv/.tex, campaign_pareto_by_dataset.*, paired_*.csv, pooled_*.csv
  summary_all_datasets.csv, summary_<dataset>.md                     the per-dataset tables (mean over seeds)
  <run>.zip                                                          everything above, ready to download
Nothing is recomputed from the learners; only the saved rows are read, so this can be re-run at any time."""
import glob, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rewire.configure import config_for, DATASET_NAME                 # noqa: E402
from rewire.campaign import reports                                   # noqa: E402
from rewire.utils.plotting import plot_all, plot_sweep, plot_gate_sweep, chunk_costs, _label   # noqa: E402


def _rows(paths):
    rows, seen = [], set()
    for p in sorted(paths):
        for r in json.load(open(p)):
            k = (r["dataset"], r.get("n_recur"), r.get("theta_hi"), r["seed"], r["arm"])
            if k not in seen: seen.add(k); rows.append(r)
    return rows


def _write(rows, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rows.json"), "w") as f: json.dump(rows, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def summary_table(rows):
    """One row per arm: the numbers the paper's tables are built from (mean over seeds)."""
    cc = chunk_costs(rows).groupby("arm").mean(numeric_only=True)
    t = pd.DataFrame([dict(arm=r["arm"], seed=r["seed"], ACC=100 * r["final_acc"], BWT=100 * r["bwt"], forgetting=100 * r["forgetting"], AUROC_stream=r["recog_auroc_fam"],
                           AUROC_test=r["fam_auroc_test"], S_AUR=r["S_AUR"], RIF=r["RIF"], mJ=r["lifetime_energy_uJ"] / 1000, mJ_front_end=r.get("frontend_energy_uJ", 0) / 1000,
                           stored_KB=r["standing_bytes"] / 1024, resident_KB=r.get("resident_bytes", np.nan) / 1024, d=r.get("feature_dim"), time_s=r.get("train_time_s", np.nan)) for r in rows])
    g = t.groupby("arm"); m = g.mean(numeric_only=True).drop(columns="seed"); s = g.std(numeric_only=True).drop(columns="seed")
    m["ACC_sd"], m["BWT_sd"], m["n_seeds"] = s["ACC"], s["BWT"], g.size()
    m["mJ_admission"] = cc["intro_uJ"] / 1000; m["mJ_readmission"] = cc["recur_uJ"] / 1000; m["skipped"] = cc["skipped"]
    m.insert(0, "label", [_label(a) for a in m.index]); return m.sort_values("ACC", ascending=False)


def merge(run, zip_it=False):
    run = Path(run); parts = run / "parts"; t0 = time.time(); main_rows = []; datasets = sorted(p.name for p in parts.iterdir() if p.is_dir()) if parts.exists() else []
    frames = []
    for ds in datasets:
        cfg = config_for(ds)
        mains = _rows(glob.glob(str(parts / ds / "main" / "s*" / "rows.json")))
        if mains:
            out = run / ds / "shift_none"; _write(mains, out); c = dict(cfg, out=str(out), shift_mode="none"); plot_all(mains, c); main_rows += mains
            tab = summary_table(mains); tab.to_csv(out / "summary_table.csv"); frames.append(tab.assign(dataset=ds))
            with open(run / f"summary_{ds}.md", "w") as f: f.write(f"# {DATASET_NAME.get(ds, ds)} ({len(mains)} runs)\n\n" + tab.round(3).to_markdown() + "\n")
        sweep = []
        for d in sorted(glob.glob(str(parts / ds / "recur" / "n*"))):
            n = int(Path(d).name[1:]); rs = _rows(glob.glob(f"{d}/s*/rows.json"))
            if rs: _write(rs, run / ds / "sweep_recur" / f"n{n}"); sweep += rs
        if sweep:
            from rewire.campaign import sweep_arms
            sweep += [r for r in mains if r["arm"] in sweep_arms(cfg)]
            plot_sweep(sweep, dict(cfg, out=str(run / ds / "sweep_recur")))
        gate = []
        for d in sorted(glob.glob(str(parts / ds / "gate" / "t*"))):
            rs = _rows(glob.glob(f"{d}/s*/rows.json"))
            if rs: _write(rs, run / ds / "sweep_gate" / Path(d).name); gate += rs
        if gate:
            from rewire.campaign import GATED
            gate += [r for r in mains if r["arm"] in GATED and r["seed"] in sorted({x["seed"] for x in gate})]
            plot_gate_sweep(gate, mains, str(run / ds / "sweep_gate"), ds)
        fid = sorted(glob.glob(str(parts / ds / "fidelity" / "s*" / "fidelity_gate.csv")))
        if fid:
            f = pd.concat([pd.read_csv(p).assign(seed=int(Path(p).parent.name[1:])) for p in fid]); os.makedirs(run / "fidelity", exist_ok=True)
            f.to_csv(run / "fidelity" / "fidelity_gate_per_seed.csv", index=False)
            g = f.groupby(["protocol", "arm"]).agg(acc=("acc", "mean"), acc_se=("acc", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0), bwt=("bwt", "mean"),
                                                   bwt_se=("bwt", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0), n=("seed", "count")).reset_index()
            g.to_csv(run / "fidelity" / "fidelity_gate.csv", index=False)
    if frames:
        allt = pd.concat(frames); allt.to_csv(run / "summary_all_datasets.csv")
        piv = allt.reset_index().pivot_table(index="arm", columns="dataset", values="ACC"); piv.round(1).to_csv(run / "acc_by_dataset.csv")
        print("\nACC by dataset (mean over seeds)\n" + piv.round(1).to_string())
    if main_rows: reports(str(run), main_rows)
    (run / "MERGED").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"merged {len(main_rows)} main-grid runs over {len(datasets)} datasets in {time.time() - t0:.0f}s -> {run}")
    if zip_it:
        z = shutil.make_archive(str(run), "zip", root_dir=run.parent, base_dir=run.name); print(f"archive: {z} ({os.path.getsize(z) / 1e6:.0f} MB)")
    return run


if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    merge(sys.argv[1], zip_it="--zip" in sys.argv)
