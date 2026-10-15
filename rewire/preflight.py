import os
"""Fast scientific guards, run before launching expensive jobs. No dataset downloads."""
import tempfile
import numpy as np
import torch
from .configure import DEFAULT, PRESETS, PRESET_OF, config_for
from .dataloader.loaders import make_stream
from .models.arms import RidgeMemory, build
from .modules import DEV


def check(condition, name):
    if not condition: raise AssertionError(name)
    print("[PASS]", name, flush=True)


def main():
    check(torch.cuda.is_available() or os.environ.get("REWIRE_PREFLIGHT_CPU") == "1", "CUDA available on training host (set REWIRE_PREFLIGHT_CPU=1 to check the mathematics on CPU)")
    check(DEFAULT["hidden"] == DEFAULT["rw_hidden"] == 1024, "gradient baselines and REWIRE's expansion share the width 1024")
    check(len(DEFAULT["seeds"]) >= 5, "at least five seeds")
    check(set(PRESET_OF) == {"mnist", "cifar10", "cifar100", "nmnist", "shd", "tinyimagenet"} and all(PRESET_OF[d] in PRESETS for d in PRESET_OF), "six datasets with a preset each")
    torch.manual_seed(0)
    h = torch.randn(19, 8, device=DEV); y = torch.randint(0, 3, (19,), device=DEV)
    A = h.T @ h + torch.eye(8, device=DEV)
    C = torch.zeros(8, 3, device=DEV); C.index_add_(1, y, h.T)
    expected = torch.linalg.solve(A, C)
    for perm in (torch.arange(19, device=DEV), torch.randperm(19, device=DEV)):
        m = RidgeMemory(8, 3, 1.0)
        for ids in perm.split(4): m.write(h[ids], y[ids])
        m.solve(); check(torch.allclose(m.W, expected, atol=3e-6), "incremental batch ridge equivalence and order independence")
    m = RidgeMemory(1, 2, 1.0)
    old = torch.ones(1, 1, device=DEV)
    m.write(old, torch.tensor([0], device=DEV)); m.solve()
    before = float((m.scores(old)[0, 0] - 1) ** 2)
    m.write(old, torch.tensor([1], device=DEV)); m.solve()
    after = float((m.scores(old)[0, 0] - 1) ** 2)
    check(after > before, "old class can worsen despite additive statistics")
    yy = torch.randint(0, 3, (15,), device=DEV); hh = torch.randn(15, 8, device=DEV)
    one, two = RidgeMemory(8, 3), RidgeMemory(8, 3)
    one.write(hh, yy); one.solve()
    two.write(hh[:7], yy[:7]); two.write(hh[7:], yy[7:]); two.solve()
    check(one.solves == 1 and torch.allclose(one.G, two.G, atol=3e-6)
          and torch.allclose(one.C, two.C, atol=3e-6)
          and torch.allclose(one.W, two.W, atol=3e-6), "one solve matches two-stage sufficient statistics")
    cfg = dict(DEFAULT, hidden=32, rw_hidden=32, rw_sketch=16, T=2,
               n_stream_classes=8, n_classes=10, n_train_per_class=35, n_test_per_class=10,
               chunk_target=12, chunk_context=6, n_recur=16)
    calib = torch.randn(12, 1, 28, 28, device=DEV)
    rw = build("rewire", cfg, 0, calib)
    check(rw.opt is None and rw.net.H == 32, "the deployed learner has no optimizer")
    check(rw.mem.head == "lda" and rw.skip and rw.d == 16 + min(28 * 28, cfg["rw_skip_max"]), "REWIRE = dual view [sketch ; input view] with the LDA readout")
    check(build("rewire_v1", cfg, 0, calib).mem.head == "ridge", "REWIRE-v1 keeps the ridge readout")
    check(rw.resident_bytes() > rw.standing_bytes(), "resident memory includes frozen tensors")
    gated = build("rewire_gated", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)
    labels0 = torch.zeros(len(calib), dtype=torch.long, device=DEV)
    empty_x, empty_y = calib[:0], labels0[:0]
    first = gated.visit(calib, labels0, empty_x, empty_y, 0, 0, True)
    second = gated.visit(calib, labels0, empty_x, empty_y, 0, 1, False)
    check(first["wrote"] and gated.mem.solves == 1 and second["wrote"] is False
          and second["write_cost_uJ"] == 0.0 and gated.opt is None, "skipped recurrence performs no solve/write/gradient update")
    for ds in ("mnist", "cifar100", "shd"):
        dc = config_for(ds)
        check(dc["n_stream_classes"] == {"mnist": 8, "cifar100": 80, "shd": 16}[ds], f"{ds} stream class count")
        check(all(isinstance(n, int) and n >= 0 for n in dc["sweep_recur"]), f"{ds} recurrence sweep configured")
    check(config_for("shd")["T"] == 20 and config_for("mnist")["T"] == 6, "SHD binned into 20 frames, frames into 6")
    rng = np.random.RandomState(0)
    X = rng.randn(600, 2).astype("float32"); labels = np.repeat(np.arange(10), 60)
    cfg.update(n_recur=16, min_recurrences_per_class=2)
    st = make_stream(X, labels, 0, cfg)
    counts = {c: 0 for c in st["stream_classes"]}
    for v in st["chunks"]:
        if v["kind"] == "recur": counts[v["cls"]] += 1
    check(min(counts.values()) >= 2, "balanced recurrence schedule")
    from .results_io import load, ORDER
    from .models.arms import PAPER_ARMS
    check(set(config_for("mnist")["arms"]) == set(PAPER_ARMS) and set(PAPER_ARMS) <= set(ORDER), "every scheduled arm is in the registry and the report order")
    with tempfile.TemporaryDirectory() as out:
        from pathlib import Path
        import json
        f = Path(out) / "mnist" / "shift_none"; f.mkdir(parents=True)
        (f / "rows.json").write_text(json.dumps([{"arm": "rewire", "dataset": "mnist", "shift_mode": "none", "seed": 0, "collapsed": True, "final_acc": 0.1}]))
        check(bool(load(out).iloc[0].collapsed), "collapsed run loaded and retained")
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__": main()
