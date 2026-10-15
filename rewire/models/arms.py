"""Every learner of the paper behind one interface (Arm.visit / predict / familiarity / recover_step / standing_bytes).
REWIRE and its variants are write-only (no optimizer); HLOP, rehearsal, EWC and fine-tuning are the gradient baselines, trained
at the same width on the same stream under the same cost model. The registry ARMS at the bottom is the single list of arm
names, groups and labels used by the campaign, the figures and the paper."""
import math, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ..modules import LIFNet, CostMeter, DEV, balanced_acc

def balanced_batch(y, n, gen):
    """Class-balanced minibatch indices (a chunk holds 80 target + 20+20 context samples; unweighted sampling collapses the net)."""
    cls = torch.unique(y); per = max(1, n // len(cls)); out = []
    for c in cls:
        ii = torch.where(y == c)[0]; out.append(ii[torch.randint(0, len(ii), (per,), generator=gen)])
    return torch.cat(out)


class Arm:
    """Common interface. visit(Xt,yt,Xc,yc,k,t) trains on one chunk and returns a log dict; recover_step is the
    arm's native single update used by the matched S_ctrl protocol; predict returns 10 logits/scores."""
    gradient = True

    def __init__(self, name, cfg, seed, calib_X):
        self.name, self.cfg, self.seed = name, cfg, seed; self.K = cfg.get("n_classes", 10)
        self.in_dim = int(calib_X[0, 0].numel()) if calib_X.dim() == 5 else int(calib_X[0].numel())   # per time step for event data
        self.meter = CostMeter(cfg); self.seen = set(); self.net = self.build_net(); self.net.calibrate(calib_X, cfg.get("drive_std", 1.5))
        self.lr = cfg["lr"] * (784.0 / self.in_dim) ** 0.5      # keep the effective step size comparable across input sizes (CIFAR has 4x the pixels)
        self.opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=self.lr) if self.gradient else None

    def build_net(self):
        net = LIFNet(self.cfg, in_dim=self.in_dim).to(DEV)
        if getattr(self, "frozen_substrate", False):
            for p in (net.fc1.weight, net.fc1.bias): p.requires_grad_(False)   # stable substrate: only the readout learns and forgets
        return net

    # ----- inference
    @torch.no_grad()
    def predict(self, X, count=True):
        out = self.net(X)
        if count: self.meter.forward_pass(self.net.stats, len(X), train=False)
        return out

    @torch.no_grad()
    def familiarity(self, X):
        return F.softmax(self.predict(X, count=False), 1).max(1).values

    # ----- training primitives
    def loss(self, out, yb): return F.cross_entropy(out, yb)

    def step(self, Xb, yb, params=None):
        self.opt.zero_grad(); out = self.net(Xb); loss = self.loss(out, yb); loss.backward()
        self.mask_grads()
        if params is not None:
            for p in self.net.parameters():
                if not any(p is q for q in params): p.grad = None
        self.opt.step(); self.meter.forward_pass(self.net.stats, len(Xb), train=True); self.meter.steps += 1
        self.overhead_per_step(len(Xb))

    def mask_grads(self): pass
    def overhead_per_step(self, n): pass

    def train_steps(self, X, y, steps, params=None):
        g = torch.Generator(device="cpu").manual_seed(self.seed * 7919 + len(self.seen) + int(steps))
        for _ in range(steps):
            idx = balanced_batch(y, self.cfg["batch"], g); self.step(X[idx], y[idx], params)

    steps_mult = 1

    def visit(self, Xt, yt, Xc, yc, k, t, is_new):
        X = torch.cat([Xt, Xc]); y = torch.cat([yt, yc]); self.before_visit(k, is_new); n = self.cfg["full_steps"] * self.steps_mult
        cs = k if isinstance(k, list) else [k]; wrote = True; c = float("nan")
        if self.cfg.get("gate_updates", False) and not is_new:            # gate-transfer control: the baseline may skip its update when
            n_ho = max(4, len(yt) // 5)                                    # its own held-out competence is at least theta_hi (the same
            Xev, yev = torch.cat([Xt[:n_ho], Xc[: len(Xc) // 2]]), torch.cat([yt[:n_ho], yc[: len(yc) // 2]])   # rule REWIRE's gate uses)
            with torch.no_grad(): c = float(np.mean([balanced_acc(self.predict(Xev).argmax(1), yev, k2) for k2 in cs]))
            if c >= self.cfg["theta_hi"]: wrote = False; n = 0
        if wrote: self.train_steps(X, y, n); self.after_visit(X, y, k)
        self.seen |= set(cs)
        return dict(steps=n, state=("known" if not wrote else "full"), deficit=float("nan"), competence=c, wrote=wrote)

    def before_visit(self, k, is_new): pass
    def after_visit(self, X, y, k): pass

    def recover_step(self, Xb, yb):
        self.cur_classes = torch.unique(yb); self.step(Xb, yb); self.seen |= set(yb.tolist())

    def standing_bytes(self): return 0


class NaiveForget(Arm):
    """Trains every chunk with the full budget, no protection: the natural-forgetting reference."""



class HLOP(Arm):
    """HLOP baseline on the AUTHORS' OWN Hebbian module (rewire/baselines/hlop_official/hlop_module.py, copied verbatim from
    github.com/pkuxmq/HLOP-SNN, Apache 2.0). The loop mirrors spiking_train_pmnist.py: on the first task the subspace of every
    projected layer is learned by the Hebbian rule (5 iterations per batch, momentum 0.9, lr 0.01) without projection; on later
    tasks weight gradients are computed from the orthogonal residual x - HLOP.projection(x) of the merged old subspace while a
    new subspace block is learned for the current task with the old blocks fixed; after each task the blocks are merged.
    Subspace sizes follow the paper's PMNIST recipe scaled to the widths used here (hlop_k initial, hlop_grow per new task,
    hlop_kmax cap). HLOP is a task- and domain-incremental method; here a "task" is each chunk that introduces a new class, and a
    recurrence chunk trains under projection without growing the subspace. We state this adaptation in the paper."""

    def build_net(self):
        from ..baselines.hlop_official import HLOP as OfficialHLOP
        net = LIFNet(self.cfg, in_dim=self.in_dim).to(DEV); net.proj = True
        self.hl = [OfficialHLOP(self.in_dim, lr=self.cfg["hlop_hebb_lr"], momentum=True).to(DEV),
                   OfficialHLOP(net.H, lr=self.cfg["hlop_hebb_lr"], momentum=True).to(DEV)]
        for h, k in zip(self.hl, self.cfg["hlop_k"]): h.add_subspace(k); h.to(DEV)
        self.n_tasks = 0; self.growing = True; self._open_block = True; return net

    def _sync_projection(self):
        """Expose the merged old subspace (block 0) to the projected linear layers as an orthonormal basis."""
        for name, h in zip(("H1", "H2"), self.hl):
            if h.subspace_num >= 1 and self.n_tasks >= 1:
                W = h.get_weight_value([0]); Q, _ = torch.linalg.qr(W.t()); setattr(self.net, name, Q.t().contiguous())
            else:
                setattr(self.net, name, torch.zeros(0, h.in_features, device=DEV))

    @torch.no_grad()
    def hebb_update(self, X):
        """Hebbian subspace learning on the traces of this batch, exactly as the official forward_with_update."""
        x1 = X.reshape(-1, X.shape[1], self.in_dim).mean(1) if X.dim() == 5 else X.flatten(1)
        x2 = self.net.features(X)
        for h, x in zip(self.hl, (x1, x2)):
            fix = list(range(h.subspace_num - 1)) if (self.n_tasks >= 1 and h.subspace_num > 1) else None
            h.forward_with_update(x, iteration=self.cfg["hlop_hebb_iters"], fix_subspace_id_list=fix)
            self.meter.overhead(self.cfg["hlop_hebb_iters"] * 3 * h.weight.shape[0] * h.in_features * len(x))

    def step(self, Xb, yb, params=None):
        super().step(Xb, yb, params)
        if self._open_block or self.cfg.get("hlop_refresh_on_recur", False): self.hebb_update(Xb)   # original: only while a new block is open

    def before_visit(self, k, is_new):
        self._open_block = bool(is_new) and self.n_tasks >= 1 and self.growing
        if self._open_block:                                       # a new task: open a new subspace block, old blocks fixed
            for h in self.hl:
                kmax = self.cfg["hlop_k"][0 if h is self.hl[0] else 1] + self.cfg["hlop_grow"] * self.cfg["n_stream_classes"]   # room for every class
                if h.index_list[-1] + self.cfg["hlop_grow"] <= kmax: h.add_subspace(self.cfg["hlop_grow"]); h.to(DEV)
        if self.n_tasks == 0: self._open_block = True              # the first task learns its subspace without projection
        self._sync_projection()

    def after_visit(self, X, y, k):
        for h in self.hl:
            if h.subspace_num > 1: h.merge_subspace()
        self.n_tasks += 1; self._sync_projection()

    def overhead_per_step(self, n):
        self.meter.overhead(n * (2 * self.net.H1.shape[0] * self.in_dim + 2 * self.net.H2.shape[0] * self.net.H))

    def standing_bytes(self): return 4 * sum(h.weight.numel() * (2 if h.momentum else 1) for h in self.hl)



class DSDSNN(Arm):
    """Dynamic Structure Development (Han et al., IJCAI 2023), 1-hidden-layer version with a shared head.
    Each visit = a task: grow dsd_grow empty neurons (current), prune current neurons by the importance rule
    P <- gamma*P + exp(-epoch/eta) * (alpha*Norm(S) - rho_p), S = sum of input weights, then freeze survivors.
    Frozen neurons keep their input rows and output columns fixed (parameter isolation)."""

    def build_net(self):
        self.steps_mult = self.cfg.get("dsd_epochs_mult", 1); self.cur_classes = torch.arange(self.cfg.get("n_classes", 10), device=DEV)
        net = LIFNet(self.cfg, h_max=self.cfg["dsd_hmax"], active=self.cfg["dsd_init"], in_dim=self.in_dim).to(DEV)
        self.status = torch.zeros(self.cfg["dsd_hmax"], dtype=torch.long, device=DEV); self.status[: self.cfg["dsd_init"]] = 1
        self.P = torch.full((self.cfg["dsd_hmax"],), self.cfg["dsd_p0"], device=DEV); self.epoch = 0; return net

    def before_visit(self, k, is_new):
        empty = torch.where(self.status == 0)[0][: self.cfg["dsd_grow"]]
        if len(empty):
            with torch.no_grad():
                self.net.fc1.weight[empty] = torch.randn(len(empty), self.in_dim, device=DEV) * (self.net.fc1.weight[self.status == 2].std().item() if (self.status == 2).any() else 1 / math.sqrt(self.in_dim)); self.net.fc1.bias[empty] = 0
                self.net.fc2.weight[:, empty] = 0
            self.status[empty] = 1; self.net.mask[empty] = 1; self.P[empty] = self.cfg["dsd_p0"]
        self.epoch = 0

    def mask_grads(self):
        """Parameter isolation: frozen neurons keep their input rows; the shared head trains only the rows of the
        classes present in the current chunk (the single-head analogue of DSD's per-task output head)."""
        frozen = self.status == 2
        if self.net.fc1.weight.grad is not None: self.net.fc1.weight.grad[frozen] = 0; self.net.fc1.bias.grad[frozen] = 0
        if self.net.fc2.weight.grad is not None:
            rows = torch.ones(self.K, dtype=torch.bool, device=DEV); rows[self.cur_classes] = False
            self.net.fc2.weight.grad[rows] = 0; self.net.fc2.bias.grad[rows] = 0

    def train_steps(self, X, y, steps, params=None):
        self.cur_classes = torch.unique(y); per_epoch = max(1, steps // 4)
        for s in range(steps):
            g = torch.Generator(device="cpu").manual_seed(self.seed * 7919 + self.meter.steps)
            idx = balanced_batch(y, self.cfg["batch"], g); self.step(X[idx], y[idx], params)
            if (s + 1) % per_epoch == 0: self.prune()

    @torch.no_grad()
    def prune(self):
        cfg = self.cfg; cur = self.status == 1
        if cur.sum() <= 2: return
        S = self.net.fc1.weight.abs().sum(1); Sn = (S - S[cur].min()) / (S[cur].max() - S[cur].min() + 1e-9)
        phi = cfg["dsd_alpha"] * Sn - cfg["dsd_rho_p"]; self.epoch += 1
        self.P[cur] = cfg["dsd_gamma"] * self.P[cur] + math.exp(-self.epoch / cfg["dsd_eta"]) * phi[cur]
        kill = cur & (self.P < 0)
        if kill.sum() >= cur.sum() - 2: return
        self.status[kill] = 0; self.net.mask[kill] = 0; self.net.fc2.weight[:, kill] = 0

    def after_visit(self, X, y, k): self.status[self.status == 1] = 2

    def standing_bytes(self): return 4 * int((self.status == 2).sum().item()) * (self.in_dim + 11)


# ----------------------------------------------------------------------------- EWC and replay
class EWC(Arm):
    """Elastic weight consolidation with an accumulating diagonal Fisher after every visit."""

    def build_net(self):
        net = LIFNet(self.cfg, in_dim=self.in_dim).to(DEV); self.F = None; self.star = None; return net

    def loss(self, out, yb):
        l = F.cross_entropy(out, yb)
        if self.F is not None:
            l = l + 0.5 * self.cfg["ewc_lambda"] * sum((f * (p - s) ** 2).sum() for f, p, s in zip(self.F, self.net.parameters(), self.star))
        return l

    def overhead_per_step(self, n):
        if self.F is not None: self.meter.overhead(3 * sum(p.numel() for p in self.net.parameters()))

    def after_visit(self, X, y, k):
        Fn = [torch.zeros_like(p) for p in self.net.parameters()]
        for i in range(0, len(y), self.cfg["batch"]):
            self.opt.zero_grad(); F.cross_entropy(self.net(X[i:i + self.cfg["batch"]]), y[i:i + self.cfg["batch"]]).backward()
            for f, p in zip(Fn, self.net.parameters()): f += p.grad.detach() ** 2 * len(y[i:i + self.cfg["batch"]])
            self.meter.forward_pass(self.net.stats, len(y[i:i + self.cfg["batch"]]), train=True)
        Fn = [f / len(y) for f in Fn]
        self.n_fisher = getattr(self, "n_fisher", 0) + 1
        self.F = Fn if self.F is None else [(a * (self.n_fisher - 1) + b) / self.n_fisher for a, b in zip(self.F, Fn)]   # running mean, not a sum
        self.star = [p.detach().clone() for p in self.net.parameters()]

    def standing_bytes(self): return 8 * sum(p.numel() for p in self.net.parameters())


class Replay(Arm):
    """Reservoir buffer of replay_M samples; every step trains on chunk batch + replay batch."""

    def build_net(self):
        self.buf_X, self.buf_y, self.n_seen_samples = [], [], 0; return LIFNet(self.cfg, in_dim=self.in_dim).to(DEV)

    def step(self, Xb, yb, params=None):
        if self.buf_y:
            bx, by = torch.stack(self.buf_X), torch.tensor(self.buf_y, device=DEV)
            idx = torch.randint(0, len(by), (min(len(by), self.cfg["batch"]),)); Xb = torch.cat([Xb, bx[idx]]); yb = torch.cat([yb, by[idx]])
        super().step(Xb, yb, params)

    def buffer_size(self, X):
        """replay_M samples; with replay_M="per_class" the standard 20 exemplars per stream class; with "iso" the number of raw
        samples that fits in REWIRE's standing memory 4(d^2 + dK + K) bytes, so the two methods hold the same bytes."""
        m = self.cfg["replay_M"]
        if m == "per_class": return int(self.cfg.get("replay_per_class", 20)) * int(self.cfg["n_stream_classes"])
        if m == "iso":                                            # REWIRE's full standing memory: Gram, class sums, and the class means/variances/counts
            d, K = self.cfg["rw_sketch"], self.cfg["n_classes"]; per = self.bytes_per_sample(X[0])
            return max(10, int(4 * (d * d + d * K + K + 2 * K * d + K) // per))
        return int(m)

    def after_visit(self, X, y, k):
        M = self.buffer_size(X); rng = np.random.RandomState(self.seed + self.n_seen_samples)
        for i in range(len(y)):
            self.n_seen_samples += 1
            if len(self.buf_y) < M: self.buf_X.append(X[i]); self.buf_y.append(int(y[i]))
            else:
                j = rng.randint(0, self.n_seen_samples)
                if j < M: self.buf_X[j] = X[i]; self.buf_y[j] = int(y[i])

    def recover_step(self, Xb, yb): self.step(Xb, yb); self.seen |= set(yb.tolist())

    @staticmethod
    def bytes_per_sample(x):
        """Frames: float32 (4 bytes per value). Event tensors (T x C x H x W) are integer counts and are stored as uint8: 1 byte per value."""
        return (1 if x.dim() == 4 else 4) * x.numel() + 8

    def standing_bytes(self):
        return int(sum(self.bytes_per_sample(x) for x in self.buf_X))



# ----------------------------------------------------------------------------- REWIRE (proposed method)
class RidgeMemory:
    """Analytic readout on frozen features, kept as streaming sufficient statistics.

    G = sum_i h_i h_i^T  (d x d),  C = sum_i h_i e_{y_i}^T  (d x K),  W = (G + lambda I)^-1 C.

    Updates are order independent and equivalent to joint ridge regression on all written samples.
    A changed joint optimum may increase loss on an earlier class. One solve is needed per written chunk.

    Two readouts on the SAME statistics (G, C, n):
      head="ridge"    : one-hot ridge regression, W = (G + lambda I)^-1 C (REWIRE-v1). The class counts enter the solution as
                        priors on the scores, which biases it on a repetition stream where recurring classes are written more often.
      head="lda"      : shared within-class covariance + class means (the SLDA form, Hayes & Kanan 2020): Sw = G - sum_k C_k C_k^T / n_k
                        is formed at solve time, so nothing extra is stored; W_k = Lam mu_k, b_k = -1/2 mu_k^T Lam mu_k with
                        Lam = (Sw/dof + eps * tr/d * I)^-1. The class-dependent terms are count-free (the proposed readout).
      solver="woodbury": keep R = (G + lambda I)^-1 and update it with each written batch of n rows at O(n d^2) instead of
                        re-factorising at O(d^3) (the recursive form); R is recomputed from G every `refresh_every` solves. Ridge only."""

    def __init__(self, d, K=10, lam=1.0, head="ridge", lda_eps=0.01, solver="direct", refresh_every=25, keep_inverse=False):
        self.d, self.K, self.lam = d, K, lam
        self.head, self.lda_eps, self.solver, self.refresh_every, self.keep_inverse = head, lda_eps, solver, refresh_every, keep_inverse
        self.G = torch.zeros(d, d, device=DEV); self.C = torch.zeros(d, K, device=DEV); self.n = torch.zeros(K, device=DEV)
        self.W = torch.zeros(d, K, device=DEV); self.b = torch.zeros(K, device=DEV); self.dirty = False; self.solves = 0
        self.R = None; self.Lam = None; self._pending = []

    def write(self, h, y):
        self.G += h.t() @ h; self.C.index_add_(1, y, h.t()); self.n.index_add_(0, y, torch.ones(len(y), device=DEV)); self.dirty = True
        if self.solver == "woodbury": self._pending.append(h.detach())

    def solve(self):
        if not self.dirty: return 0
        self.dirty = False; self.solves += 1
        if self.head == "lda": return self._solve_lda()
        if self.solver == "woodbury": return self._solve_woodbury()
        self._pending = []
        A = self.G + self.lam * torch.eye(self.d, device=DEV)
        try: self.W = torch.linalg.solve(A, self.C)
        except Exception: self.W = torch.linalg.lstsq(A, self.C).solution
        # Conservative MAC-equivalent estimate: LU factorization, K triangular solves, plus formation of A.
        # This is an operation model, not measured GPU joules.
        return int(2 * self.d ** 3 / 3 + 2 * self.d * self.d * self.K + self.d * self.d)

    def _solve_lda(self):
        d, K = self.d, self.K; seen = self.n > 0; ns = self.n.clamp(min=1)
        M = self.C / ns[None]                                   # d x K class means (zero columns for unseen classes)
        Sw = self.G - (self.C / ns[None]) @ self.C.t()          # within-class scatter: G - sum_k C_k C_k^T / n_k
        dof = max(float(self.n.sum() - seen.sum()), 1.0); Sigma = Sw / dof
        tr = float(Sigma.diagonal().mean()); A = Sigma + self.lda_eps * (tr if tr > 0 else 1.0) * torch.eye(d, device=DEV)
        if self.keep_inverse:                                   # the Mahalanobis familiarity needs the inverse itself
            try: self.Lam = torch.linalg.inv(A)
            except Exception: self.Lam = torch.linalg.pinv(A)
            self.W = self.Lam @ M; cost = d ** 3 + d * d * K
        else:                                                   # otherwise one LU solve for W (2 d^3 / 3), no inverse formed
            try: self.W = torch.linalg.solve(A, M)
            except Exception: self.W = torch.linalg.lstsq(A, M).solution
            self.Lam = None; cost = 2 * d ** 3 // 3 + d * d * K
        self.b = -0.5 * (M * self.W).sum(0)
        return int(cost + d * d * K + d * d)                    # + forming Sw (d^2 K) and A (d^2)

    def _solve_woodbury(self):
        """The inverse is kept in float64 and re-symmetrised after every update: in float32 the recursion drifts within a few
        updates at d ~ 1000 (the classical RLS wind-up), which the float64 accumulator removes; the periodic refresh from G stays
        as a safety net. The operation count is unchanged (the model counts operations, not their precision)."""
        d, K = self.d, self.K; cost = 0
        if self.R is None or (self.refresh_every and self.solves % self.refresh_every == 0):
            A = (self.G + self.lam * torch.eye(d, device=DEV)).double(); self.R = torch.linalg.inv(A); cost += d ** 3; self._pending = []
        else:
            for H in self._pending:                              # rank-n update: (A + H^T H)^-1 = R - R H^T (I + H R H^T)^-1 H R
                Hd = H.double(); n = Hd.shape[0]; RH = self.R @ Hd.t(); S = torch.eye(n, device=DEV, dtype=torch.float64) + Hd @ RH
                self.R = self.R - RH @ torch.linalg.solve(S, RH.t()); self.R = 0.5 * (self.R + self.R.t())
                cost += 2 * d * d * n + 2 * n * n * d + n ** 3
            self._pending = []
        self.W = (self.R @ self.C.double()).float(); cost += d * d * K
        return int(cost)

    def scores(self, h):
        s = h @ self.W + self.b; return s.masked_fill((self.n == 0)[None], -1e9)

    def bytes(self): return 4 * (self.d * self.d + self.d * self.K + self.K)

class REWIRE(Arm):
    """REWIRE: a write-only continual learner on a frozen spiking substrate.

    substrate : rw_hidden leaky integrate-and-fire units with fixed random input weights, read out as spike counts over T steps.
                Never updated, so the representation cannot drift; regenerated from the seed, so it costs no standing memory.
    sketch    : a fixed random map of the spike counts to rw_sketch dimensions followed by a ReLU (also regenerated from the seed).
    dual view : with rw_skip_input the memory sees [spiking sketch ; input view]; the input view is the front-end feature itself
                or, for inputs wider than rw_skip_max, a fixed random projection to rw_skip_max dimensions. The spiking block is
                scaled once, label-free, to the input view's variance (sk_scale).
    memory    : RidgeMemory above. Writing a chunk = adding its features to (G, C, n) and one closed-form solve; no gradient
                anywhere. With rw_head="lda" the readout is the count-free discriminant (the proposed configuration).
    gate      : with rw_gated a recurrence is written only if it is NOT already known: known = familiar (cosine or Mahalanobis
                match to the stored class means, against the class's own calibrated reference) AND competent (balanced accuracy on
                the chunk's held-out slice >= theta_hi). rw_gate_budget > 0 additionally requires >= budget written samples of
                the class before a recurrence may be skipped.
    modes     : "memonly" (the analytic memory) and "hd" (write-once hyperdimensional bundling on the same features, a control)."""
    gradient = False   # the base class must not build an optimizer over the frozen substrate

    def __init__(self, name, cfg, seed, calib_X, mode="memonly"):
        self.mode = mode
        super().__init__(name, cfg, seed, calib_X)
        d = self.d; self.mem = RidgeMemory(d, self.K, cfg["rw_lambda"], head=cfg.get("rw_head", "ridge"), lda_eps=cfg.get("rw_lda_eps", 0.01),
                                           solver=cfg.get("rw_solver", "direct"), refresh_every=cfg.get("rw_refresh_every", 25), keep_inverse=(cfg.get("rw_fam", "cos") == "maha"))
        self.opt = None
        self.mu = torch.zeros(self.K, d, device=DEV); self.sq = torch.zeros(self.K, d, device=DEV); self.cnt = torch.zeros(self.K, device=DEV)
        self.mastery = {}; self.log = []; self.fam_ref = {}; self.n_bank = {}   # n_bank: samples written per class (budgeted gate)
        if self.skip:                                              # dual view: one label-free scalar from the calibration batch
            with torch.no_grad():
                hs = self._sketch_view(calib_X); xin = self._input_view(calib_X)
                self.sk_scale = float(xin.std() / (hs.std() + 1e-6)) * float(cfg.get("rw_skip_scale", 1.0))

    def build_net(self):
        cfg = self.cfg; H = cfg["rw_hidden"]
        net = LIFNet(cfg, h_max=H, in_dim=self.in_dim).to(DEV)
        for p in net.parameters(): p.requires_grad_(False)
        self.d_sk = (self.in_dim if cfg.get("rw_raw_ridge", False) else cfg["rw_sketch"])
        g = torch.Generator().manual_seed(4242 + self.seed)
        # sketch: a fixed random map H -> d followed by a ReLU (the random-feature expansion of RanPAC, McDonnell et al. 2023).
        self.S = (torch.randn(H, self.d_sk, generator=g) / H ** 0.5).to(DEV); self.Sb = (torch.randn(self.d_sk, generator=g) * 0.1).to(DEV)
        # dual view (rw_skip_input): identity on the front-end feature, or a fixed random projection to rw_skip_max dims for wider inputs.
        self.skip = bool(cfg.get("rw_skip_input", False)) and not cfg.get("rw_raw_ridge", False); self.P = None; self.sk_scale = 1.0; d_in = 0
        if self.skip:
            dmax = int(cfg.get("rw_skip_max", 512))
            if self.in_dim > dmax:
                gP = torch.Generator().manual_seed(4343 + self.seed); self.P = (torch.randn(self.in_dim, dmax, generator=gP) / self.in_dim ** 0.5).to(DEV); d_in = dmax
            else: d_in = self.in_dim
        self.d = self.d_sk + d_in
        return net

    def _input_view(self, X):
        z = X.reshape(-1, X.shape[1], self.in_dim).mean(1) if X.dim() == 5 else X.flatten(1)
        return z if self.P is None else z @ self.P

    def _sketch_view(self, X):
        """The spiking (or, for the ReLU control, non-spiking) random-feature block alone, before any scaling."""
        if self.cfg.get("rw_no_lif", False):
            z = X.reshape(-1, X.shape[1], self.in_dim).mean(1) if X.dim() == 5 else X.flatten(1)
            if not hasattr(self, "S_in"):
                g = torch.Generator(device="cpu").manual_seed(self.seed + 4242)
                self.S_in = (torch.randn(self.in_dim, self.net.H, generator=g) / self.in_dim ** 0.5).to(DEV); self.S_in_b = (torch.randn(self.net.H, generator=g) * 0.1).to(DEV)
            return F.relu(F.relu(z @ self.S_in + self.S_in_b) @ self.S + self.Sb)
        return F.relu(self.net.features(X) @ self.S + self.Sb)

    # ----- features
    def feats(self, X):
        with torch.no_grad():
            if self.cfg.get("rw_raw_ridge", False):              # the input-only controls: no expansion, no sketch
                return X.reshape(-1, X.shape[1], self.in_dim).mean(1) if X.dim() == 5 else X.flatten(1)
            hs = self._sketch_view(X)
            if self.skip: return torch.cat([hs * self.sk_scale, self._input_view(X)], 1)   # dual view: [spiking sketch ; input]
            return hs

    def _fam(self, h):
        """Per-sample familiarity. Default: max cosine to the stored class means. rw_fam="maha" (needs the LDA head): a bounded
        Mahalanobis score 1 / (1 + D^2/d) under the shared covariance the head already holds, so classifier, gate and novelty score
        use one statistic. Before the first solve (no covariance yet) it falls back to the cosine."""
        if self.cfg.get("rw_fam", "cos") == "maha" and getattr(self.mem, "Lam", None) is not None:
            M = self.mu / self.cnt.clamp(min=1)[:, None]; Lam = self.mem.Lam; hL = h @ Lam
            d2 = (hL * h).sum(1, keepdim=True) - 2 * hL @ M.t() + ((M @ Lam) * M).sum(1)[None]
            s = 1.0 / (1.0 + d2.clamp(min=0) / self.d)
            return s.masked_fill((self.cnt == 0)[None], 0.0).max(1).values
        P = F.normalize(self.mu / self.cnt.clamp(min=1)[:, None], dim=1)
        s = (F.normalize(h, dim=1) @ P.t()).masked_fill((self.cnt == 0)[None], -1.0)
        return s.max(1).values

    def _fam_macs(self, n):
        return n * (self.d * self.d + self.K * self.d) if self.cfg.get("rw_fam", "cos") == "maha" else n * self.K * self.d

    def _hd_scores(self, h):
        """Write-once hyperdimensional memory on the SAME features as the analytic memory: each class keeps a bipolar bundle
        (the sign of the summed sign(h) vectors); recognition is the cosine to the bundle. No solve, no gradients."""
        if not hasattr(self, "hd_bundle"): self.hd_bundle = torch.zeros(self.K, self.d, device=DEV); self.hd_mu = torch.zeros(self.d, device=DEV); self.hd_n = 0
        B = torch.sign(self.hd_bundle); s = F.normalize(torch.sign(h - self.hd_mu), dim=1) @ F.normalize(B, dim=1).t()
        return s.masked_fill((self.cnt == 0)[None], -1e9)

    def _extra_macs(self, n):
        if self.cfg.get("rw_raw_ridge", False): return 0
        m = n * (self.in_dim * self.net.H + self.net.H * self.d_sk if self.cfg.get("rw_no_lif", False) else self.net.H * self.d_sk)
        if self.P is not None: m += n * self.in_dim * self.P.shape[1]      # dual view: projection of a wide input to rw_skip_max dims
        return m

    def _fwd_stats(self):
        """Forward-pass cost: the LIF expansion's measured stats, or nothing for the non-spiking controls."""
        return self.net.stats if not self.cfg.get("rw_no_lif", False) else dict(macs_per_sample=0.0, sops_per_sample=0.0, spikes_per_sample=0.0, active_frac=0.0)

    @torch.no_grad()
    def predict(self, X, count=True):
        h = self.feats(X); out = self._hd_scores(h) if self.mode == "hd" else self.mem.scores(h)
        if count: self.meter.forward_pass(self._fwd_stats(), len(X), train=False); self.meter.overhead(self._extra_macs(len(X)))
        return out

    @torch.no_grad()
    def familiarity(self, X):
        """Recognition is a MATCH to what is stored, not the classifier's own confidence: the analytic readout fits any class it
        has written, so its posterior says nothing about novelty; the match to the stored class means does."""
        return self._fam(self.feats(X))

    # ----- write
    @torch.no_grad()
    def write(self, X, y, h=None, cached_prefix=0):
        if h is None: h = self.feats(X)
        if self.mode == "hd":                                     # bipolar code: sign of the feature minus its running mean
            if not hasattr(self, "hd_bundle"): self.hd_bundle = torch.zeros(self.K, self.d, device=DEV); self.hd_mu = torch.zeros(self.d, device=DEV); self.hd_n = 0
            self.hd_mu = (self.hd_mu * self.hd_n + h.sum(0)) / (self.hd_n + len(h)); self.hd_n += len(h)
            self.hd_bundle.index_add_(0, y, torch.sign(h - self.hd_mu))
        else:
            self.mem.write(h, y)
        self.mu.index_add_(0, y, h); self.sq.index_add_(0, y, h * h); self.cnt.index_add_(0, y, torch.ones(len(y), device=DEV))
        self.meter.forward_pass(self._fwd_stats(), len(X) - cached_prefix, train=False)
        self.meter.overhead(self._extra_macs(len(X) - cached_prefix) + len(X) * self.d * (1 if self.mode == "hd" else self.d + self.K))
        if self.mode != "hd": self.meter.overhead(self.mem.solve())
        self.seen |= set(y.tolist())

    def _bank(self, y):
        """Book-keeping for the budgeted gate: how many samples of each class have been written so far (labels only, no cost)."""
        for c, n in zip(*torch.unique(y, return_counts=True)): self.n_bank[int(c)] = self.n_bank.get(int(c), 0) + int(n)

    # ----- stream interface
    def visit(self, Xt, yt, Xc, yc, k, t, is_new):
        cfg = self.cfg; cs = k if isinstance(k, list) else [k]
        cost_start = self.meter.energy_uJ()
        n_ho = max(4, len(yt) // 5); Xho, yho = Xt[:n_ho], yt[:n_ho]                     # held-out slice: competence only, never written
        Xtr, ytr = torch.cat([Xt[n_ho:], Xc]), torch.cat([yt[n_ho:], yc])
        Xev, yev = torch.cat([Xho, Xc[: len(Xc) // 2]]), torch.cat([yho, yc[: len(yc) // 2]])
        ht = self.feats(Xt)
        fs = self._fam(ht)
        fam = float(fs.mean())
        self.meter.forward_pass(self._fwd_stats(), len(Xt), train=False); self.meter.overhead(self._extra_macs(len(Xt)) + self._fam_macs(len(Xt)))
        gated = cfg.get("rw_gated", False) and not is_new
        if gated:                                                          # familiarity AND labelled competence decide the write
            with torch.no_grad(): c_pre = float(np.mean([balanced_acc(self.predict(Xev, count=True).argmax(1), yev, k2) for k2 in cs]))
            tau = min(self.fam_ref.get(k2, -1.0) for k2 in cs)
            known = (fam >= tau - cfg.get("rw_gate_delta", 0.05)) and (c_pre >= cfg["theta_hi"])
            budget = int(cfg.get("rw_gate_budget", 0))                     # budgeted gate: a recognised, competent recurrence is skipped
            if budget > 0 and min(self.n_bank.get(k2, 0) for k2 in cs) < budget: known = False   # only once >= budget samples are banked
        else: known = False
        wrote = not known
        probe_cost = self.meter.energy_uJ() - cost_start
        cost_before_write = self.meter.energy_uJ()
        n_written = len(ytr) if wrote else 0
        if wrote:
            sel_mode = cfg.get("rw_select", "none")
            if sel_mode in ("errors", "random") and not is_new:
                # Selective write (the count-bias control): score the chunk with the CURRENT readout and write only the samples it gets
                # wrong plus a seeded random fraction rw_select_keep of the ones it gets right; "random" writes a random subset of the
                # same size, the matched control that separates the error signal from simply writing less.
                h_ctx = self.feats(Xc) if len(yc) else ht[:0]
                self.meter.forward_pass(self._fwd_stats(), len(Xc), train=False); self.meter.overhead(self._extra_macs(len(Xc)))
                h_tr = torch.cat([ht[n_ho:], h_ctx])
                pred = self.mem.scores(h_tr).argmax(1); self.meter.overhead(len(h_tr) * self.d * self.K)
                g_sel = torch.Generator(device="cpu").manual_seed(self.seed * 7919 + t + 17)
                keep = (torch.rand(len(ytr), generator=g_sel) < cfg.get("rw_select_keep", 0.25)).to(DEV)
                sel = (pred != ytr) | keep
                if sel_mode == "random":
                    k_sel = int(sel.sum()); perm = torch.randperm(len(ytr), generator=g_sel)[:k_sel].to(DEV)
                    sel = torch.zeros_like(sel); sel[perm] = True
                n_written = int(sel.sum())
                if n_written > 0: self.write(Xtr[sel], ytr[sel], h=h_tr[sel], cached_prefix=n_written); self._bank(ytr[sel])
            else:
                self.write(Xtr, ytr, h=torch.cat([ht[n_ho:], self.feats(Xc)]) if len(yc) else ht[n_ho:], cached_prefix=len(yt) - n_ho); self._bank(ytr)
        write_cost = self.meter.energy_uJ() - cost_before_write
        with torch.no_grad(): c = c_pre if known else float(np.mean([balanced_acc(self.predict(Xev).argmax(1), yev, k2) for k2 in cs]))
        # The familiarity reference is calibrated on chunk-internal samples only, never on the test set.
        f_ho = self._fam(ht[:n_ho])
        for k2 in cs: self.fam_ref[k2] = float(f_ho.mean())
        self.meter.overhead(self._fam_macs(len(Xho)))
        m = max(self.mastery.get(k2, 0.0) for k2 in cs) if not is_new else 1.0
        d = max(0.0, m - c) if not is_new else 1.0
        with torch.no_grad(): m_new = c if not wrote else float(np.mean([balanced_acc(self.predict(Xev).argmax(1), yev, k2) for k2 in cs]))
        for k2 in cs: self.mastery[k2] = max(m_new, 0.5)
        state = "write"
        if gated: state = "known" if known else ("shifted" if fam < tau - cfg.get("rw_gate_delta", 0.05) else "partial")
        rec = dict(steps=0, state=state, deficit=d, competence=c, competence_pre=c_pre if gated else None,
                   competence_threshold=cfg["theta_hi"], familiarity_threshold=(tau - cfg.get("rw_gate_delta", 0.05)) if gated else None,
                   mastery=m_new, fam=fam, wrote=wrote, write_samples=n_written,
                   probe_cost_uJ=probe_cost, write_cost_uJ=write_cost,
                   write_reason="introduction" if is_new else ("mastered" if known else "recurrence_needs_write"))
        self.log.append(dict(t=t, cls=(cs[0] if len(cs) == 1 else -1), **rec)); return rec

    def recover_step(self, Xb, yb):
        self.write(Xb, yb); self.meter.steps += 1

    def standing_bytes(self):
        """What persists between chunks: (G, C, n) plus the class means, squared sums and counts for the familiarity test, plus 4
        bytes for the seed that regenerates the substrate and the sketch."""
        if self.mode == "hd": return 4 + self.K * self.d // 8 + 4 * self.K * (self.d + 1)      # seed + bit-packed bundles + class means/counts
        return self.mem.bytes() + 4 * (self.mu.numel() + self.sq.numel() + self.K) + 4

    def resident_bytes(self):
        """Everything held in memory at deployment, including the regenerable frozen tensors."""
        tensors = [self.mem.G, self.mem.C, self.mem.W, self.mem.n, self.mu, self.sq, self.cnt,
                   self.S, self.Sb, *self.net.parameters(), self.net.mask]
        if hasattr(self, "S_in"): tensors += [self.S_in, self.S_in_b]
        if self.P is not None: tensors += [self.P]
        for extra in ("R", "Lam"):
            if getattr(self.mem, extra, None) is not None: tensors += [getattr(self.mem, extra)]
        return sum(t.numel() * t.element_size() for t in tensors)


# ----------------------------------------------------------------------------- the arms of the paper
REWIRE_OPTS = dict(rw_skip_input=True, rw_head="lda")   # the proposed configuration: dual-view sketch + count-free (LDA) readout


def _rewire_cfg(cfg, **over):
    """REWIRE's configuration: v1's expansion and sketch width (rw_sketch; rw2_sketch overrides it), the dual view and the LDA head
    with shrinkage rw2_eps, plus any per-arm overrides."""
    c2 = {**cfg, **REWIRE_OPTS, "rw_sketch": int(cfg.get("rw2_sketch") or cfg["rw_sketch"]), "rw_lda_eps": float(cfg.get("rw2_eps", 0.1))}
    c2.update(over); return c2


def _mem(name, c2, seed, calib_X, mode="memonly"): return REWIRE(name, c2, seed, calib_X, mode)


# name -> (group, label, builder). Groups: proposed | ablation | control | baseline | fidelity. The label is the one used in
# every figure and table; the paper's assets tool reads this table so names never drift between code and paper.
ARMS = {
    # ---- the proposed method and its gate variants
    "rewire":            ("proposed", "REWIRE (ours)",                        lambda n, c, s, x: _mem(n, _rewire_cfg(c), s, x)),
    "rewire_gated":      ("proposed", "REWIRE-gated (ours)",                  lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_gated=True), s, x)),
    "rewire_gated_maha": ("proposed", "REWIRE-gated, Mahalanobis (ours)",     lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_gated=True, rw_fam="maha"), s, x)),
    "rewire_gated_t095": ("proposed", "REWIRE-gated, $\\theta$=0.95 (ours)",  lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_gated=True, theta_hi=0.95), s, x)),
    "rewire_gated_b128": ("proposed", "REWIRE-gated, budget 128 (ours)",      lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_gated=True, rw_gate_budget=128), s, x)),
    # ---- ablations: one factor removed or changed
    "rewire_v1":         ("ablation", "REWIRE-v1: sketch only, ridge",        lambda n, c, s, x: _mem(n, c, s, x)),
    "rewire_v1_gated":   ("ablation", "REWIRE-v1 gated",                      lambda n, c, s, x: _mem(n, {**c, "rw_gated": True}, s, x)),
    "abl_sketch_lda":    ("ablation", "Sketch only, LDA readout",             lambda n, c, s, x: _mem(n, {**c, "rw_head": "lda", "rw_lda_eps": float(c.get("rw2_eps", 0.1))}, s, x)),
    "abl_dualview_ridge":("ablation", "Dual view, ridge readout",             lambda n, c, s, x: _mem(n, {**c, "rw_skip_input": True}, s, x)),
    "abl_width128":      ("ablation", "REWIRE, sketch 128",                   lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_sketch=128), s, x)),
    "abl_width256":      ("ablation", "REWIRE, sketch 256",                   lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_sketch=256), s, x)),
    "abl_select_errors": ("ablation", "REWIRE, selective write (errors)",     lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_select="errors"), s, x)),
    "abl_select_random": ("ablation", "REWIRE, selective write (random)",     lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_select="random"), s, x)),
    "abl_v1_select_errors": ("ablation", "REWIRE-v1, selective write (errors)", lambda n, c, s, x: _mem(n, {**c, "rw_select": "errors"}, s, x)),
    "abl_v1_select_random": ("ablation", "REWIRE-v1, selective write (random)", lambda n, c, s, x: _mem(n, {**c, "rw_select": "random"}, s, x)),
    "abl_v1_woodbury":   ("ablation", "REWIRE-v1, recursive solve",           lambda n, c, s, x: _mem(n, {**c, "rw_solver": "woodbury"}, s, x)),
    # ---- controls: the same memory without the spiking layer
    "ctrl_slda_input":   ("control", "SLDA on the inputs",                    lambda n, c, s, x: _mem(n, {**c, "rw_no_lif": True, "rw_raw_ridge": True, "rw_head": "lda", "rw_lda_eps": float(c.get("rw2_eps", 0.1))}, s, x)),
    "ctrl_relu":         ("control", "ReLU layer instead of LIF",             lambda n, c, s, x: _mem(n, _rewire_cfg(c, rw_no_lif=True), s, x)),
    "ranpac":            ("control", "RanPAC-style (ReLU, ridge)",            lambda n, c, s, x: _mem(n, {**c, "rw_no_lif": True}, s, x)),
    "ridge_input":       ("control", "Ridge on the inputs",                   lambda n, c, s, x: _mem(n, {**c, "rw_no_lif": True, "rw_raw_ridge": True}, s, x)),
    "hd_memory":         ("control", "HD bundling",                           lambda n, c, s, x: _mem(n, c, s, x, "hd")),
    # ---- gradient baselines (same width, same stream, same cost model)
    "hlop":              ("baseline", "HLOP",                                 lambda n, c, s, x: HLOP(n, c, s, x)),
    "hlop_adaptive":     ("baseline", "HLOP-R (subspace refreshed)",          lambda n, c, s, x: HLOP(n, {**c, "hlop_refresh_on_recur": True}, s, x)),
    "hlop_gated":        ("baseline", "HLOP, competence-gated",               lambda n, c, s, x: HLOP(n, {**c, "gate_updates": True}, s, x)),
    "replay":            ("baseline", "Rehearsal (20/class)",                 lambda n, c, s, x: Replay(n, {**c, "replay_M": "per_class"}, s, x)),
    "replay_iso":        ("baseline", "Rehearsal (iso-memory)",               lambda n, c, s, x: Replay(n, {**c, "replay_M": "iso"}, s, x)),
    "replay_gated":      ("baseline", "Rehearsal, competence-gated",          lambda n, c, s, x: Replay(n, {**c, "replay_M": "per_class", "gate_updates": True}, s, x)),
    "ewc":               ("baseline", "EWC",                                  lambda n, c, s, x: EWC(n, c, s, x)),
    "naive":             ("baseline", "Fine-tuning",                          lambda n, c, s, x: NaiveForget(n, c, s, x)),
    # ---- fidelity gate only (its own protocol; not on the stream)
    "dsd_snn":           ("fidelity", "DSD-SNN",                              lambda n, c, s, x: DSDSNN(n, c, s, x)),
}
PAPER_ARMS = [a for a, (g, _, _) in ARMS.items() if g != "fidelity"]
GROUP = {a: g for a, (g, _, _) in ARMS.items()}
LABEL = {a: l for a, (_, l, _) in ARMS.items()}


def build(name, cfg, seed, calib_X):
    if name not in ARMS: raise ValueError(f"unknown arm '{name}'; known: {', '.join(ARMS)}")
    return ARMS[name][2](name, cfg, seed, calib_X)
