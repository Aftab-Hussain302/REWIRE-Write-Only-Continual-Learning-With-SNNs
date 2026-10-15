"""One unit of the campaign in one process: (dataset, seed, stage), writing its own rows.json so that several units run side by
side without sharing a file. scripts/run_all.py schedules these; scripts/merge_results.py joins their outputs into the usual tree.

    python3 -m rewire.worker --dataset cifar100 --seed 2 --stage main   --out results/X/parts/cifar100/main/s2 --data_root ~/data
    python3 -m rewire.worker --dataset mnist    --seed 0 --stage recur  --n-recur 64 --out ...
    python3 -m rewire.worker --dataset shd      --seed 1 --stage gate   --theta 0.95 --out ...
    python3 -m rewire.worker --dataset mnist    --seed 0 --stage fidelity --out ...

Stages: main (every arm of the paper), recur (the recurrence-density sweep arms at --n-recur), gate (the gated arms at --theta),
fidelity (the baselines on their native protocols; MNIST only)."""
import argparse, json, os, sys, time
from .configure import config_for
from .campaign import sweep_arms, GATED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", required=True, choices=("main", "recur", "gate", "fidelity"))
    ap.add_argument("--out", required=True); ap.add_argument("--data_root", default="data"); ap.add_argument("--n-recur", type=int); ap.add_argument("--theta", type=float)
    ap.add_argument("--arms", nargs="*"); ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    a = ap.parse_args()
    over = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in a.set}; over["seeds"] = [a.seed]
    if a.arms: over["arms"] = a.arms
    cfg = config_for(a.dataset, over); cfg["data_root"], cfg["out"], cfg["shift_mode"] = a.data_root, a.out, "none"
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    if a.stage == "fidelity":
        from .fidelity import run_fidelity
        run_fidelity(cfg, seeds=(a.seed,), epochs=3, out=a.out)
    else:
        from .dataloader import load_dataset
        from .engine import run_grid
        def fix(**kw):                                                     # stage settings win over any --set override (run_grid re-applies them)
            cfg.update(kw); cfg.setdefault("_overrides", {}).update(kw)
        if a.stage == "recur":
            if a.n_recur is None: raise SystemExit("--n-recur is required for the recur stage")
            fix(n_recur=a.n_recur, arms=[x for x in sweep_arms(cfg) if x in cfg["arms"]])
        elif a.stage == "gate":
            if a.theta is None: raise SystemExit("--theta is required for the gate stage")
            fix(theta_hi=a.theta, arms=[x for x in cfg["arms"] if x in GATED])
        X, y = load_dataset(a.dataset, cfg["data"], a.data_root, cfg)
        rows = run_grid(cfg, print, X, y)
        print(f"[worker] {a.dataset} seed {a.seed} {a.stage}: {len(rows)} rows")
    with open(os.path.join(a.out, "DONE"), "w") as f: f.write(f"{time.time() - t0:.0f}s\n")


if __name__ == "__main__": main()
