"""LIF neuron with surrogate gradient, the HLOP-style projected linear function, the cost meter and the spiking expansion."""
import math, torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpikeFn(torch.autograd.Function):
    slope = 2.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v); return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors; return g / (1.0 + SpikeFn.slope * v.abs()) ** 2


class ProjLinearFn(torch.autograd.Function):
    """Linear layer whose WEIGHT gradient is computed from the projected presynaptic trace x_hat = x - H^T(Hx).
    This is HLOP's orthogonal projection (Xiao et al., ICLR 2024): forward unchanged, updates confined to the
    complement of the consolidated subspace H (rows orthonormal)."""

    @staticmethod
    def forward(ctx, x, W, b, H):
        ctx.save_for_backward(x, W, H); return x @ W.t() + b

    @staticmethod
    def backward(ctx, g):
        x, W, H = ctx.saved_tensors
        xh = x - (x @ H.t()) @ H if H.shape[0] > 0 else x
        return g @ W, g.t() @ xh, g.sum(0), None


class CostMeter:
    """Lifetime cost in the convention standard for spiking networks (Zhou et al., ICLR 2023; Yao et al., 2022), built on the
    45 nm, 0.9 V energy table of Horowitz (ISSCC 2014):

        E = E_MAC * FLOPs + E_AC * SOPs,     E_MAC = 4.6 pJ (32-bit float MAC),  E_AC = 0.9 pJ (32-bit float accumulate)

    FLOPs are the dense multiply-accumulates of non-spiking layers (input drive, readouts, projections, analytic solves); SOPs are
    accumulate operations triggered by spikes, SOPs(l) = fr(l) * T * FLOPs(l), with fr the measured firing rate and T the number of
    time steps. Training is charged a backward factor over the forward pass. The figure is an analytic operation count at one
    technology node, not a hardware measurement."""

    def __init__(self, cfg):
        self.cfg = cfg; self.reset()

    def reset(self):
        self.macs = 0.0; self.sops = 0.0; self.steps = 0; self.samples = 0; self.spikes = 0.0; self.extra_macs = 0.0

    def forward_pass(self, stats, n, train):
        f = 1.0 + (self.cfg["backward_factor"] if train else 0.0)
        self.macs += stats["macs_per_sample"] * n * f; self.sops += stats["sops_per_sample"] * n * f
        self.spikes += stats["spikes_per_sample"] * n; self.samples += n

    def overhead(self, macs):  # method-specific standing overhead (projection, Hebbian updates, penalties, replay ...)
        self.extra_macs += macs

    def energy_uJ(self):
        return ((self.macs + self.extra_macs) * self.cfg["pj_mac"] + self.sops * self.cfg["pj_sop"]) * 1e-6

    def snapshot(self):
        return dict(macs=self.macs + self.extra_macs, sops=self.sops, steps=self.steps, samples=self.samples, spikes=self.spikes, energy_uJ=self.energy_uJ())


class LIFNet(nn.Module):
    """in_dim -> H hidden LIF (direct current encoding over T steps, rate readout) -> 10 linear logits.
    `mask` (H_max,) marks active hidden neurons (all ones except for DSD growth). `H1`/`H2` are HLOP subspaces."""

    def __init__(self, cfg, h_max=None, active=None, in_dim=784):
        super().__init__()
        self.cfg = cfg; H = h_max or cfg["hidden"]; self.H = H; self.in_dim = in_dim; self.K = cfg.get("n_classes", 10)
        self.fc1 = nn.Linear(in_dim, H); self.fc2 = nn.Linear(H, self.K)
        self.register_buffer("mask", torch.zeros(H)); self.mask[: (active or H)] = 1.0
        self.register_buffer("H1", torch.zeros(0, in_dim)); self.register_buffer("H2", torch.zeros(0, H))
        self.gain = 1.0; self.stats = {}; self.proj = False
        SpikeFn.slope = cfg["surrogate_slope"]

    @torch.no_grad()
    def calibrate(self, x, target_std=1.5):
        """Fold the drive scale into the initial weights (gain stays 1) so the optimizer step size is in native units."""
        z = x.reshape(-1, x.shape[1], self.in_dim)[:, 0] if x.dim() == 5 else x.flatten(1)
        cur = self.fc1(z); g = float(target_std / (cur.std() + 1e-6)); self.fc1.weight.mul_(g); self.fc1.bias.mul_(g); self.gain = 1.0

    def features(self, x):
        """Static input: a constant drive is integrated for T steps (direct current encoding). Event input (N,T,C,H,W): the drive of
        step t is the t-th event frame, so the network sees the real temporal signal."""
        ev = x.dim() == 5
        if ev:
            xs = x.reshape(x.shape[0], x.shape[1], -1)
            cur_t = [(ProjLinearFn.apply(xs[:, t], self.fc1.weight, self.fc1.bias, self.H1) if self.proj else self.fc1(xs[:, t])) * self.gain for t in range(xs.shape[1])]
            cur = cur_t[0]
        else:
            x = x.flatten(1)
            cur = (ProjLinearFn.apply(x, self.fc1.weight, self.fc1.bias, self.H1) if self.proj else self.fc1(x)) * self.gain
        T, beta, thr = self.cfg["T"], self.cfg["beta"], self.cfg["thr"]
        v = torch.zeros_like(cur); tot = 0.0; ever = torch.zeros_like(cur)
        for t in range(T):
            drive = cur_t[min(t, len(cur_t) - 1)] if ev else cur
            v = beta * v + drive; s = SpikeFn.apply(v - thr) * self.mask; v = v - s * thr; tot = tot + s; ever = torch.maximum(ever, s)
        rate = tot / T; n_act = float(self.mask.sum())
        spk = tot.detach().sum(1).mean().item()
        # Input-layer cost. Static input: one dense pass (the constant drive is computed once). Event input, by cfg["event_input_cost"]:
        #   "dense_once" (legacy): one dense pass regardless of T;  "dense_per_frame": one dense pass per frame (T x);
        #   "spikes": the frames are binary events, so each input event costs one accumulate per postsynaptic neuron (SOPs), no MACs.
        macs_in, sops_in = self.in_dim * n_act, 0.0
        if ev:
            mode = self.cfg.get("event_input_cost", "dense_once")
            if mode == "dense_per_frame": macs_in = self.in_dim * n_act * xs.shape[1]
            elif mode == "spikes": macs_in, sops_in = 0.0, (xs != 0).sum((1, 2)).float().mean().item() * n_act
        self.stats = dict(macs_per_sample=macs_in, sops_per_sample=spk * self.K + sops_in, spikes_per_sample=spk,
                          active_frac=ever.detach().sum(1).mean().item() / max(n_act, 1))
        return rate

    def forward(self, x):
        h = self.features(x)
        return ProjLinearFn.apply(h, self.fc2.weight, self.fc2.bias, self.H2) if self.proj else self.fc2(h)

    def n_params(self):
        return int(self.in_dim * self.mask.sum().item() + self.mask.sum().item() + self.K * self.mask.sum().item() + self.K)


def balanced_acc(pred, y, c):
    pos, neg = y == c, y != c
    rec = (pred[pos] == c).float().mean().item() if pos.any() else 0.0
    spec = (pred[neg] != c).float().mean().item() if neg.any() else 1.0
    return 0.5 * (rec + spec)
