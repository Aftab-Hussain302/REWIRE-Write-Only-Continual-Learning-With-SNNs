"""Fidelity gate for the maintenance competitors on their NATIVE protocol: task-incremental Split-MNIST (5 tasks x 2 classes,
multi-head via output masking), identical backbone and optimizer. Reports average accuracy and BWT for naive, HLOP and DSD-SNN.
Gate: a faithful protection mechanism must hold BWT >= -0.05 with average accuracy >= 0.90 on this protocol (the published
numbers, HLOP PMNIST 95.2% / BWT -1.3%, DSD-SNN MNIST TIL 97.3%, use much larger networks and cannot be matched by a 128-unit net;
the gate tests that the MECHANISM works, i.e. that it removes forgetting where naive does not)."""
import json, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
from .configure import DEFAULT
from .dataloader import load_dataset
from .modules import DEV
from .models import build


def _one(name, cfg, seed, protocol, X, y, Xt, yt, tr, te, epochs, log):
    """protocol 'split' = task-IL Split-MNIST (5 tasks x 2 classes, masked head); 'pmnist' = domain-IL Permuted MNIST (5 permutations, 10-way head)."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed + 17)
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]] if protocol == "split" else [list(range(10))] * 5
    perms = [None] + [torch.tensor(rng.permutation(Xt[0].numel()), device=DEV) for _ in range(4)] if protocol == "pmnist" else [None] * 5
    pm = lambda x, p: x if p is None else x.flatten(1)[:, p].reshape(x.shape)
    arm = build(name, cfg, seed, Xt[tr[0][:64]]); base_loss = arm.loss; A = np.zeros((5, 5))
    for i, tk in enumerate(tasks):
        mask = torch.full((10,), -1e9, device=DEV); mask[tk] = 0; arm.loss = (lambda out, yb, mask=mask: base_loss(out + mask, yb))
        Xi = pm(Xt[np.concatenate([tr[c] for c in tk])], perms[i]); yi = yt[np.concatenate([tr[c] for c in tk])]
        steps = epochs * len(yi) // cfg["batch"]; arm.before_visit(tk, True); arm.train_steps(Xi, yi, steps); arm.after_visit(Xi, yi, tk)
        for j, tj in enumerate(tasks[: i + 1]):
            mj = torch.full((10,), -1e9, device=DEV); mj[tj] = 0
            Xe = pm(Xt[np.concatenate([te[c] for c in tj])], perms[j]); ye = yt[np.concatenate([te[c] for c in tj])]
            with torch.no_grad(): A[i, j] = ((arm.predict(Xe, count=False) + mj).argmax(1) == ye).float().mean().item()
    bwt = np.mean([A[-1, j] - A[j, j] for j in range(4)])
    log(f"  fidelity {protocol:6s} {name:12s} seed {seed}: acc {A[-1].mean():.3f} bwt {bwt:+.3f} diag {np.round(np.diag(A), 2).tolist()}")
    return A[-1].mean(), bwt


def run_fidelity(cfg=None, arms=("naive", "hlop", "dsd_snn", "ewc"), seeds=(0, 1, 2), epochs=3, n_per_class=300, log=print, protocols=("split", "pmnist"), out=None):
    cfg = dict(DEFAULT if cfg is None else cfg); X, y = load_dataset(cfg["dataset"], cfg["data"], cfg["data_root"])
    Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV); report = {}
    n_avail = min(int((y == c).sum()) for c in range(10)); n_te = min(100, n_avail // 3); n_per_class = min(n_per_class, n_avail - n_te)
    for protocol in protocols:
        for name in arms:
            accs, bwts = [], []
            for seed in seeds:
                rng = np.random.RandomState(seed); tr, te = {}, {}
                for c in range(10):
                    idx = np.where(y == c)[0]; rng.shuffle(idx); tr[c] = idx[:n_per_class]; te[c] = idx[n_per_class:n_per_class + n_te]
                a_, b_ = _one(name, cfg, seed, protocol, X, y, Xt, yt, tr, te, epochs, log); accs.append(a_); bwts.append(b_)
            report[(protocol, name)] = dict(acc=float(np.mean(accs)), acc_se=float(np.std(accs) / np.sqrt(len(accs))), bwt=float(np.mean(bwts)), bwt_se=float(np.std(bwts) / np.sqrt(len(bwts))))
    log("\nFIDELITY GATE. split = task-IL Split-MNIST (DSD-SNN's native protocol), pmnist = domain-IL Permuted MNIST (HLOP's native protocol).")
    log("PASS = protection mechanism removes the forgetting naive shows: bwt >= -0.05 and acc >= 0.90 (split) / bwt >= -0.10 and acc >= 0.80 (pmnist)")
    for (protocol, name), r in report.items():
        thr = (-0.05, 0.90) if protocol == "split" else (-0.10, 0.80)
        verdict = "n/a (reference)" if name == "naive" else ("PASS" if (r["bwt"] >= thr[0] and r["acc"] >= thr[1]) else "FAIL")
        log(f"  {protocol:6s} {name:12s} acc {r['acc']:.3f}+/-{r['acc_se']:.3f} bwt {r['bwt']:+.3f}+/-{r['bwt_se']:.3f}  {verdict}")
    out = out or cfg.get("out", "results")
    os.makedirs(out, exist_ok=True)
    rows = [dict(protocol=p, arm=n, **v) for (p, n), v in report.items()]
    pd.DataFrame(rows).to_csv(f"{out}/fidelity_gate.csv", index=False)
    with open(f"{out}/fidelity_gate.json", "w") as fh: json.dump(rows, fh, indent=1)
    log(f"  (saved {out}/fidelity_gate.csv)")
    return report


if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--data", default="auto"); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); ap.add_argument("--data_root", default="data"); ap.add_argument("--out", default="results"); ap.add_argument("--hidden", type=int, default=None, help="backbone width (use the campaign preset width)"); a = ap.parse_args()
    c = dict(DEFAULT); c["data"] = a.data; c["data_root"] = a.data_root; c["out"] = a.out
    if a.hidden: c["hidden"] = a.hidden
    run_fidelity(c, seeds=a.seeds, out=a.out)
