"""The properties the paper relies on, as executable checks (CPU, seconds):  python3 -m pytest tests -q
  Proposition 1  order invariance: the memory after any ordering of the writes equals batch fitting on the union
  Proposition 2  count-free discriminant: duplicating a class's samples leaves the LDA means and biases unchanged (ridge changes)
  the LDA solve matches a NumPy reference; the recursive (Woodbury) ridge solve matches the direct one; a skipped recurrence costs no write."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewire.models.arms import RidgeMemory, build
from rewire.configure import DEFAULT
from rewire.modules import DEV

torch.manual_seed(0)
d, K, n = 12, 4, 60
H = torch.randn(n, d, device=DEV); Y = torch.randint(0, K, (n,), device=DEV)


def _fit(head, order, chunks=4, **kw):
    m = RidgeMemory(d, K, 1.0, head=head, **kw)
    for ids in order.split(n // chunks): m.write(H[ids], Y[ids])
    m.solve(); return m


def test_order_invariance_ridge_and_lda():
    for head in ("ridge", "lda"):
        a = _fit(head, torch.arange(n, device=DEV)); b = _fit(head, torch.randperm(n, device=DEV)); c = _fit(head, torch.randperm(n, device=DEV), chunks=1)
        for m in (b, c):
            assert torch.allclose(a.W, m.W, atol=1e-5) and torch.allclose(a.b, m.b, atol=1e-5), head


def test_count_free_discriminant():
    """Writing class 0 three times over (as a recurring class is) shifts the ridge readout towards class 0: its one-hot targets act
    as a prior. The LDA head's class-dependent terms are the class means, which do not change; the counts enter only the pooled
    covariance, so the share of samples the readout assigns to class 0 moves far less."""
    dup = torch.cat([torch.arange(n, device=DEV), torch.where(Y == 0)[0], torch.where(Y == 0)[0]])
    share = {}
    for head in ("ridge", "lda"):
        a = _fit(head, torch.arange(n, device=DEV), chunks=1); b = _fit(head, dup, chunks=1)
        if head == "lda": assert torch.allclose(a.C / a.n.clamp(min=1)[None], b.C / b.n.clamp(min=1)[None], atol=1e-5), "class means are count-free"
        pa, pb = a.scores(H).argmax(1), b.scores(H).argmax(1); share[head] = float((pb == 0).float().mean() - (pa == 0).float().mean())
    assert share["ridge"] > 0, "the ridge readout drifts towards the over-written class"
    assert abs(share["lda"]) < share["ridge"], f"the LDA readout must drift less than ridge ({share})"


def test_lda_matches_numpy():
    m = _fit("lda", torch.arange(n, device=DEV), chunks=1, lda_eps=0.1)
    Hn, Yn = H.cpu().numpy().astype(np.float64), Y.cpu().numpy()
    mu = np.stack([Hn[Yn == k].mean(0) if (Yn == k).any() else np.zeros(d) for k in range(K)], 1)
    Sw = sum(((Hn[Yn == k] - mu[:, k]).T @ (Hn[Yn == k] - mu[:, k])) for k in range(K) if (Yn == k).any())
    Sigma = Sw / (n - len(set(Yn.tolist()))); A = Sigma + 0.1 * np.trace(Sigma) / d * np.eye(d)
    W = np.linalg.solve(A, mu); b = -0.5 * (mu * W).sum(0)
    assert np.allclose(m.W.cpu().numpy(), W, atol=1e-4) and np.allclose(m.b.cpu().numpy(), b, atol=1e-4)


def test_woodbury_matches_direct():
    a = _fit("ridge", torch.arange(n, device=DEV)); w = _fit("ridge", torch.arange(n, device=DEV), solver="woodbury", refresh_every=0)
    assert torch.allclose(a.W, w.W, atol=1e-4)


def test_write_only_and_gate():
    cfg = dict(DEFAULT, hidden=32, rw_hidden=32, rw_sketch=16, T=2, n_classes=10, n_stream_classes=8, n_holdout=2, n_train_per_class=35, n_test_per_class=10,
               chunk_target=12, chunk_context=6, n_recur=16, stem="conv", conv_stem=False)
    calib = torch.rand(12, 1, 28, 28, device=DEV); y0 = torch.zeros(12, dtype=torch.long, device=DEV); e = calib[:0]; ey = y0[:0]
    rw = build("rewire", cfg, 0, calib)
    assert rw.opt is None and rw.mem.head == "lda" and rw.skip, "REWIRE: no optimizer, LDA readout, dual view"
    first = rw.visit(calib, y0, e, ey, 0, 0, True); assert first["wrote"] and rw.mem.solves == 1 and first["steps"] == 0
    g = build("rewire_gated", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)       # a gate that always recognises
    g.visit(calib, y0, e, ey, 0, 0, True); second = g.visit(calib, y0, e, ey, 0, 1, False)
    assert second["wrote"] is False and second["write_cost_uJ"] == 0.0 and g.mem.solves == 1, "a recognised recurrence is probed, not written"
    b = build("rewire_gated_b128", {**cfg, "theta_hi": 0.0, "rw_gate_delta": 100.0}, 0, calib)  # the budgeted gate writes until 128 samples are banked
    b.visit(calib, y0, e, ey, 0, 0, True); third = b.visit(calib, y0, e, ey, 0, 1, False)
    assert third["wrote"] is True, "below the budget the recurrence is written"
