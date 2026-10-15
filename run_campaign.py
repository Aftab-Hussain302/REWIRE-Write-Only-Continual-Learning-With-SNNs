"""Run the campaign for one or more datasets in one process (the launcher scripts/run_all.py splits the same work over several).

    python3 run_campaign.py --datasets mnist --data_root ~/data --out results/dev            # every arm, five seeds, both sweeps
    python3 run_campaign.py --datasets shd --seeds 0 --arms rewire hlop --no-sweeps --out results/dev
    python3 run_campaign.py --quick --datasets cifar10 --data_root ~/data                      # a short smoke stream

Results: <out>/<dataset>/shift_none/   <out>/<dataset>/sweep_recur/n<N>/   <out>/<dataset>/sweep_gate/t<theta>/   <out>/campaign_*"""
import argparse, json
from rewire.configure import config_for, PRESET_OF
from rewire.campaign import run_campaign

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(PRESET_OF)); ap.add_argument("--seeds", type=int, nargs="*"); ap.add_argument("--arms", nargs="*")
ap.add_argument("--data_root", default="data"); ap.add_argument("--out", default="results/dev")
ap.add_argument("--quick", action="store_true", help="short streams, one seed: a smoke run"); ap.add_argument("--no-sweeps", action="store_true"); ap.add_argument("--no-gate-sweep", action="store_true")
ap.add_argument("--fidelity", action="store_true", help="also run the baselines' fidelity gate on their native protocols (MNIST)")
ap.add_argument("--set", nargs="*", default=[], help="key=value overrides applied last (JSON values)")
ap.add_argument("--parallel-worker", action="store_true", help="suppress the cross-dataset reports (one process handles one dataset)")
a = ap.parse_args()
over = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in a.set}
if a.quick: over.update(seeds=[0], n_recur=8, full_steps=20, budget=25, n_train_per_class=150, n_test_per_class=40, sweep_recur=[4, 8], sweep_theta=[0.85])
if a.seeds is not None: over["seeds"] = a.seeds
if a.arms: over["arms"] = a.arms
for ds in a.datasets:
    cfg = config_for(ds, over); cfg["data_root"], cfg["out"], cfg["parallel_worker"] = a.data_root, a.out, a.parallel_worker
    if a.fidelity and ds == "mnist":
        from rewire.fidelity import run_fidelity
        run_fidelity(cfg, seeds=cfg["seeds"][:3], epochs=1 if a.quick else 3, out=a.out)
    run_campaign(cfg, datasets=[ds], shifts=["none"], recur_sweep=not a.no_sweeps, gate_sweep=not (a.no_sweeps or a.no_gate_sweep))
