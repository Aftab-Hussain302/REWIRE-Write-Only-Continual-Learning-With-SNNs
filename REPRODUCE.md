# Reproducing the paper

Everything below is the exact pipeline that produced the reported numbers; nothing else was used.

## 1. Environment (once)
```
bash setup_env.sh && . .venv/bin/activate
```

## 2. Data
Put the six datasets under one folder as described in `DATASETS.md` (e.g. `~/data`), or let the next command download what is
missing from the official sources (slower, but nothing to do by hand):
```
python3 scripts/prepare_data.py --data_root ~/data
```
This loads every dataset once, reports what it found, downloads a dataset that is absent, and builds the frozen ResNet-18
feature caches for CIFAR-10, CIFAR-100 and Tiny-ImageNet (a few minutes on a GPU). It ends with `ALL DATASETS READY`.

## 3. Checks (about five minutes, CPU is enough)
```
python3 -m rewire.preflight          # the mathematical properties the method relies on, the protocol tables
python3 selftest.py                  # every arm of the paper end to end on a short synthetic stream, the fidelity gate, every figure
python3 -m pytest tests -q           # Propositions 1 and 2 as tests, LDA vs NumPy, recursive vs direct solve, the gate
```

## 4. A ten-minute smoke run of the whole pipeline on the real data
```
python3 scripts/run_all.py --data_root ~/data --smoke
```
Short streams, one seed, seven arms, every dataset and every stage (main grid, recurrence sweep, gate sweep, merge, figures, zip).
The result lands in `results/smoke_<date>_<time>/`.

## 5. The campaign (what the paper reports)
```
nohup python3 scripts/run_all.py --data_root ~/data > run_all.log 2>&1 &
tail -f run_all.log            # or: cat results/results_<date>_<time>/status.json
```
Six datasets x five seeds x 29 arms, the recurrence-density sweep (8, 16, 32 and, on the 80-class streams, 64 recurrence chunks),
the gate sweep (theta in 0.75, 0.85, 0.95, seeds 0-2) and the baselines' fidelity gate. Units of work (dataset, seed, stage) run in
parallel (`--workers`, default = cores - 1) and resume: a finished unit is never repeated, and an interrupted run continues with
`--resume results/results_<date>_<time>`. Expect 4-5 hours on an RTX A6000 with 6 CPU cores (the gradient baselines are CPU-bound).

When the queue is empty the run folder holds
```
<dataset>/shift_none/rows.json, figures, leaderboard.csv, summary_table.csv     the main grid
<dataset>/sweep_recur/, <dataset>/sweep_gate/                                   the two sweeps with their figures
fidelity/fidelity_gate.csv                                                       the baselines on their native protocols
campaign_leaderboard.csv, campaign_pareto_by_dataset.*, paired_*.csv, pooled_*.csv, summary_all_datasets.csv, summary_<dataset>.md
ENVIRONMENT.md, MANIFEST.json, logs/, parts/                                     the machine, the code hashes, the per-unit outputs
```
and `results/results_<date>_<time>.zip` is the archive to download. `python3 scripts/merge_results.py <run folder> --zip` rebuilds
every figure, table and test from the saved rows without re-running anything.

## 6. One dataset in one process (for inspection)
```
python3 run_campaign.py --datasets shd --seeds 0 --arms rewire rewire_gated hlop replay --no-sweeps --data_root ~/data --out results/dev
```

Every row of every `rows.json` carries the full configuration of its run, the stream schedule, the per-chunk accuracy and energy
curves and the visit log, and all numbers in the paper are computed from those rows.

## 7. The paper's tables and figures from the released records (no training)
```
python3 scripts/unpack_records.py results/campaign_2026-10-09_1120
python3 plotting/make_plots.py --run results/campaign_2026-10-09_1120 --out plotting/assets_rebuilt
```
`results/campaign_2026-10-09_1120/` is the run reported in the paper (rows gzipped, `parts/` and the regenerable per-metric
figure files left out). `plotting/assets/` is the copy used for the submission.

## 8. A free GPU (Colab, Kaggle)
`notebooks/REWIRE_quickstart_colab.ipynb` runs steps 3 and 6 on MNIST in about fifteen minutes; `DATASETS.md` says which
datasets fit a free runtime (Tiny-ImageNet needs more RAM).
