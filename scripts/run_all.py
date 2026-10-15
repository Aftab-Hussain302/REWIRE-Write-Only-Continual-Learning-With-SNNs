#!/usr/bin/env python3
"""The whole campaign on one machine, in parallel, resumable, into a new time-stamped folder.

    python3 scripts/run_all.py --data_root ~/data                      # everything: 6 datasets x 5 seeds x every arm + both sweeps + fidelity
    python3 scripts/run_all.py --data_root ~/data --smoke              # a 10-minute end-to-end check of every dataset and stage
    python3 scripts/run_all.py --data_root ~/data --resume results/results_2026-10-09_0830   # continue an interrupted run

Each unit of work (dataset, seed, stage) runs in its own process through `python -m rewire.worker` and writes its own folder under
<run>/parts/, so nothing is shared between processes and a finished unit is never repeated. The longest units start first. When the
queue is empty the parts are merged into the usual tree (<run>/<dataset>/shift_none/..., sweep_recur/, sweep_gate/), every figure
and table is produced, ENVIRONMENT.md and MANIFEST.json record the machine and the code, and <run>.zip is written for download."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rewire.configure import config_for, PAPER_DATASETS, PRESET_OF            # noqa: E402
from rewire.campaign import sweep_arms, GATED                                 # noqa: E402

WEIGHT = dict(tinyimagenet=10.0, cifar100=9.0, shd=5.0, nmnist=2.5, cifar10=1.5, mnist=1.5)   # rough relative cost of one seed of the main grid


def plan(datasets, seeds, over):
    """The list of units: (priority, dataset, seed, stage, extra args), from the same configuration the workers will use."""
    units = []
    for ds in datasets:
        cfg = config_for(ds, dict(over, seeds=seeds)); w = WEIGHT.get(ds, 3.0)
        for s in seeds:
            units.append((w * 1.0, ds, s, "main", []))
            n_sw = len(sweep_arms(cfg)) / max(1, len(cfg["arms"]))
            for n in cfg["sweep_recur"]:
                if n != cfg["n_recur"]: units.append((w * n_sw * (cfg["n_stream_classes"] + n) / (cfg["n_stream_classes"] + cfg["n_recur"]), ds, s, "recur", ["--n-recur", str(n)]))
            if s in seeds[:3]:
                n_g = len([a for a in cfg["arms"] if a in GATED]) / max(1, len(cfg["arms"]))
                for th in cfg["sweep_theta"]:
                    if th != cfg["theta_hi"]: units.append((w * n_g * 2.5, ds, s, "gate", ["--theta", str(th)]))
        if ds == "mnist":
            for s in seeds[:3]: units.append((0.8, ds, s, "fidelity", []))
    units.sort(key=lambda u: -u[0]); return units


def unit_dir(run, ds, seed, stage, extra):
    tag = {"main": "main", "fidelity": "fidelity", "recur": "recur/n" + (extra[1] if extra else ""), "gate": "gate/t" + (extra[1] if extra else "")}[stage]
    return run / "parts" / ds / tag / f"s{seed}"


def environment_md(run, args):
    import platform, torch
    lines = ["# Environment of this run", "", "```"]
    try: lines.append(subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip())
    except Exception: lines.append("gpu: none")
    lines += [f"python {platform.python_version()}  torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version() if torch.cuda.is_available() else 'n/a'}",
              f"cpu: {os.cpu_count()} cores  ({platform.processor() or platform.machine()})", f"workers: {args.workers}  threads per worker: 1", f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", "```", ""]
    try: lines += ["<details><summary>pip freeze</summary>", "", "```", subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).strip(), "```", "</details>"]
    except Exception: pass
    (run / "ENVIRONMENT.md").write_text("\n".join(lines) + "\n")


def manifest(run, args, units):
    files = sorted(p for p in ROOT.rglob("*.py") if "results" not in p.parts and ".venv" not in p.parts)
    h = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code_hash = hashlib.sha256("".join(h[k] for k in sorted(h)).encode()).hexdigest()
    (run / "MANIFEST.json").write_text(json.dumps(dict(command=" ".join(sys.argv), args=vars(args), code_hash=code_hash, files=h, units=len(units),
                                                      datasets=args.datasets, seeds=args.seeds, started=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True); ap.add_argument("--out", default="results", help="parent folder; the run goes into <out>/results_<date>_<time>")
    ap.add_argument("--resume", default=None, help="an existing run folder to continue"); ap.add_argument("--workers", type=int, default=0, help="0 = cores - 1 (max 8)")
    ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true", help="short streams, seed 0, four arms: an end-to-end check of every dataset and stage (~10 min on a GPU)")
    ap.add_argument("--no-sweeps", action="store_true"); ap.add_argument("--no-fidelity", action="store_true"); ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides passed to every unit (JSON values)")
    args = ap.parse_args()
    args.workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if args.smoke: args.seeds = [0]
    run = Path(args.resume) if args.resume else Path(args.out) / f"{'smoke' if args.smoke else 'results'}_{time.strftime('%Y-%m-%d_%H%M')}"
    run.mkdir(parents=True, exist_ok=True); (run / "logs").mkdir(exist_ok=True); (run / "parts").mkdir(exist_ok=True)
    extra = list(args.set)
    if args.smoke: extra += ["n_recur=6", "full_steps=10", "n_train_per_class=150", "n_test_per_class=40", "recovery_budgets=[0,1,2]", "sweep_recur=[4]", "sweep_theta=[0.85,0.95]",
                             'arms=["rewire","rewire_gated","rewire_v1","ctrl_slda_input","hlop","replay","naive"]']
    over = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in extra}
    units = plan(args.datasets, args.seeds, over)
    if args.no_sweeps: units = [u for u in units if u[3] in ("main", "fidelity")]
    if args.no_fidelity: units = [u for u in units if u[3] != "fidelity"]
    environment_md(run, args); manifest(run, args, units)
    print(f"run folder: {run}\n{len(units)} units, {args.workers} workers, datasets {args.datasets}, seeds {args.seeds}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    todo = [u for u in units if not (unit_dir(run, *u[1:]) / "DONE").exists()]; done = len(units) - len(todo)
    print(f"{done} units already finished, {len(todo)} to run", flush=True)
    active, status, t0, durations, failed = {}, {}, time.time(), [], []
    def write_status():
        rem = len(todo) + len(active); avg = (sum(durations) / len(durations)) if durations else None
        json.dump(dict(finished=done, running=[f"{u[1]} s{u[2]} {u[3]} {' '.join(u[4])}" for (_, u, _, _) in active.values()], remaining=rem, failed=failed,
                       elapsed_min=round((time.time() - t0) / 60, 1), eta_min=(round(rem * avg / args.workers / 60, 1) if avg else None), updated=time.strftime("%H:%M:%S")),
                  open(run / "status.json", "w"), indent=1)
    while todo or active:
        for pid, (p, u, ts, log) in list(active.items()):
            code = p.poll()
            if code is None: continue
            log.close(); del active[pid]; dt = time.time() - ts
            if code == 0: done += 1; durations.append(dt / max(0.2, u[0])); print(f"finished {u[1]} s{u[2]} {u[3]} {' '.join(u[4])} in {dt/60:.1f} min ({done}/{len(units)})", flush=True)
            else: failed.append(f"{u[1]} s{u[2]} {u[3]} {' '.join(u[4])} (exit {code})"); print(f"FAILED {u[1]} s{u[2]} {u[3]} (exit {code}); see logs/", flush=True)
            write_status()
        while todo and len(active) < args.workers:
            u = todo.pop(0); d = unit_dir(run, *u[1:]); d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "rewire.worker", "--dataset", u[1], "--seed", str(u[2]), "--stage", u[3], "--out", str(d), "--data_root", args.data_root, *u[4]]
            if extra: cmd += ["--set", *extra]
            log = open(run / "logs" / f"{u[1]}_s{u[2]}_{u[3]}{('_' + u[4][1]) if u[4] else ''}.log", "a"); log.write("\n" + " ".join(cmd) + "\n")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)); active[p.pid] = (p, u, time.time(), log)
            print(f"started  {u[1]} s{u[2]} {u[3]} {' '.join(u[4])}  ({len(active)} running, {len(todo)} queued)", flush=True); write_status()
        time.sleep(5)
    write_status()
    if failed:                                                                        # one retry of the failures, sequentially
        print(f"retrying {len(failed)} failed units", flush=True); retry = [u for u in units if f"{u[1]} s{u[2]} {u[3]} {' '.join(u[4])}" in " ".join(failed)]
        failed.clear()
        for u in retry:
            d = unit_dir(run, *u[1:]); cmd = [sys.executable, "-m", "rewire.worker", "--dataset", u[1], "--seed", str(u[2]), "--stage", u[3], "--out", str(d), "--data_root", args.data_root, *u[4]] + (["--set", *extra] if extra else [])
            with open(run / "logs" / f"{u[1]}_s{u[2]}_{u[3]}_retry.log", "a") as log: code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            if code != 0: failed.append(f"{u[1]} s{u[2]} {u[3]} {' '.join(u[4])} (exit {code})")
        write_status()
    print(f"all units finished in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}", flush=True)
    if not args.no_merge:
        import importlib.util
        spec = importlib.util.spec_from_file_location("merge_results", ROOT / "scripts" / "merge_results.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.merge(run, zip_it=True)


if __name__ == "__main__": main()
