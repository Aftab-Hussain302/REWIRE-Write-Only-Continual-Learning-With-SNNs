import copy, hashlib, json, os, time, sys, numpy as np, torch
from .configure import DEFAULT, apply_dataset_defaults
from .dataloader import load_dataset, make_stream, rotate, apply_shift, stem_transform
from .modules import DEV, balanced_acc
from .models.arms import build, GROUP


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def accuracy(arm, X, y, classes):
    out = arm.predict(X, count=False); mask = torch.full((out.shape[1],), -1e9, device=DEV); mask[classes] = 0
    pred = (out + mask).argmax(1); return (pred == y).float().mean().item(), pred


def _pools_ok(pool_y, ctx_y): return len(pool_y) > 0


def recovery_event(arm, c, pool_X, pool_y, ctx_X, ctx_y, te_X, te_y, classes, cfg, rng, ceiling=None):
    if not _pools_ok(pool_y, ctx_y): return 0, True
    """Matched few-shot in-context recovery. The criterion is min(tau, tau_rel x ceiling), where ceiling is the best balanced
    accuracy this arm ever reached on that class during the stream (for a novel class, the mean ceiling over the seen ones).
    Recovery means returning to what the model could once do; an absolute bar censors arms whose ceiling is simply lower and
    made v2's KWIK and DSD savings unreadable. Returns (steps, censored)."""
    k = cfg["shots"]; target = cfg["tau"] if ceiling is None else min(cfg["tau"], cfg["tau_rel"] * ceiling)
    for step in range(cfg["budget"] + 1):
        _, pred = accuracy(arm, te_X, te_y, classes)
        if balanced_acc(pred, te_y, c) >= target: return step, False
        if step == cfg["budget"]: return step, True
        ip = rng.choice(len(pool_y), k, replace=len(pool_y) < k)
        if len(ctx_y):
            ic = rng.choice(len(ctx_y), k, replace=len(ctx_y) < k)
            arm.recover_step(torch.cat([pool_X[ip], ctx_X[ic]]), torch.cat([pool_y[ip], ctx_y[ic]]))
        else: arm.recover_step(pool_X[ip], pool_y[ip])


def recovery_curve(arm, c, pool_X, pool_y, ctx_X, ctx_y, te_X, te_y, classes, cfg, rng):
    """Censoring-free recovery measure: balanced accuracy of class c after 0, 1, 2, 5, 10 few-shot in-context steps.
    Every arm is measured on the same grid, so no event can be lost to a step budget, and the area under the curve gives a single
    number (AUR) that is defined even when an arm never reaches an absolute criterion."""
    k = cfg["shots"]; out = {}; prev = 0
    if not _pools_ok(pool_y, ctx_y):
        _, pred = accuracy(arm, te_X, te_y, classes); return {b: balanced_acc(pred, te_y, c) for b in cfg["recovery_budgets"]}
    for b in cfg["recovery_budgets"]:
        for _ in range(b - prev):
            ip = rng.choice(len(pool_y), k, replace=len(pool_y) < k)
            if len(ctx_y):
                ic = rng.choice(len(ctx_y), k, replace=len(ctx_y) < k)
                arm.recover_step(torch.cat([pool_X[ip], ctx_X[ic]]), torch.cat([pool_y[ip], ctx_y[ic]]))
            else: arm.recover_step(pool_X[ip], pool_y[ip])
        prev = b
        _, pred = accuracy(arm, te_X, te_y, classes); out[b] = balanced_acc(pred, te_y, c)
    return out


def savings(ev_rel, ev_nov, cens_rel, cens_nov):
    """S_ctrl on budget-capped costs (secondary) and on UNCENSORED events only (primary; nan when nothing is uncensored)."""
    En = max(np.mean(ev_nov), 1e-9); S_cap = float(1 - np.mean(ev_rel) / En)
    ur = [e for e, cz in zip(ev_rel, cens_rel) if not cz]; un = [e for e, cz in zip(ev_nov, cens_nov) if not cz]
    S_unc = float(1 - np.mean(ur) / max(np.mean(un), 1e-9)) if ur and un else float("nan")
    return S_cap, S_unc


def run_one(name, seed, X, y, cfg, log=print):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.RandomState(seed)
    st = make_stream(X, y, seed, cfg); Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV)
    S, Hd = st["stream_classes"], st["holdout"]; dom = {c: False for c in range(cfg["n_classes"])}     # which classes have moved to their shifted domain

    def te(cs):
        xs, ys = [], []
        for c in cs:
            x = Xt[st["test"][c]]; xs.append(apply_shift(x, c, seed, cfg) if dom[c] else x); ys.append(yt[st["test"][c]])
        return torch.cat(xs), torch.cat(ys)

    arm = build(name, cfg, seed, Xt[st["chunks"][0]["target"]])
    effective = {k: v for k, v in cfg.items() if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    protocol = json.dumps(effective, sort_keys=True, default=str)
    run_key = f"{cfg['dataset']}:{cfg['shift_mode']}:{seed}:{name}:{protocol}"
    R = dict(arm=name, seed=seed, dataset=cfg["dataset"], shift_mode=cfg["shift_mode"], n_recur=cfg["n_recur"], theta_hi=float(getattr(arm, "cfg", cfg).get("theta_hi", cfg["theta_hi"])),
             group=GROUP.get(name, ""),
             run_id=hashlib.sha256(run_key.encode()).hexdigest()[:16], config=effective,
             versions={"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
             schedule_sha256=hashlib.sha256(str([(ch["cls"], ch["kind"]) for ch in st["chunks"]]).encode()).hexdigest(),
             class_order=[int(z) for z in st["order"]],
             hidden_width=(cfg["rw_hidden"] if getattr(arm, "gradient", True) is False else cfg["hidden"]),
             sketch_width=cfg["rw_sketch"], feature_dim=int(getattr(arm, "d", cfg["rw_sketch"])),
             stream=[([int(c) for c in ch["cls"]] if isinstance(ch["cls"], list) else int(ch["cls"]), ch["kind"]) for ch in st["chunks"]], holdout=[int(h) for h in Hd])
    seen, acc_curve, energy_curve, steps_curve, visits, rif, after_last, best, after_first = [], [], [], [], [], [], {}, {}, {}
    per_class_curve = {c: [] for c in S}; t0 = time.time()
    for ch in st["chunks"]:
        c, kind = ch["cls"], ch["kind"]; cs = c if isinstance(c, list) else [c]; is_new = any(k2 not in seen for k2 in cs)
        Xc, yc = Xt[ch["target"]], yt[ch["target"]]; Xx, yx = Xt[ch["context"]], yt[ch["context"]]
        if ch["shifted"]: Xc = apply_shift(Xc, c, seed, cfg); dom[c] = True
        Xx = torch.cat([apply_shift(Xx[yx == o], o, seed, cfg) if dom[o] else Xx[yx == o] for o in ch["context_classes"]]) if len(yx) else Xx
        yx = torch.cat([yx[yx == o] for o in ch["context_classes"]]) if len(yx) else yx
        pre_fam = arm.familiarity(Xc).mean().item()
        _, pred = accuracy(arm, torch.cat([Xc, Xx]), torch.cat([yc, yx]), list(range(cfg["n_classes"]))); pre_comp = float(np.mean([balanced_acc(pred, torch.cat([yc, yx]), k2) for k2 in cs]))
        ctxc = ch["context_classes"]; teS_X, teS_y = te(S)
        if ctxc:
            _, p0 = accuracy(arm, teS_X, teS_y, S); ba0 = np.mean([balanced_acc(p0, teS_y, a) for a in ctxc])
        info = arm.visit(Xc, yc, Xx, yx, c, ch["t"], is_new)
        for k2 in cs:
            if k2 not in seen: seen.append(k2)
        if ctxc:
            _, p1 = accuracy(arm, teS_X, teS_y, S); rif.append(np.mean([balanced_acc(p1, teS_y, a) for a in ctxc]) - ba0)
        a_seen, _ = accuracy(arm, *te(seen), classes=seen); acc_curve.append(a_seen)
        _, pS = accuracy(arm, teS_X, teS_y, S)
        for k2 in S:
            m = teS_y == k2; ak = (pS[m] == k2).float().mean().item() if k2 in seen else float("nan"); per_class_curve[k2].append(ak)
            if k2 in seen: best[k2] = max(best.get(k2, 0), ak)
        for k2 in cs:
            after_last[k2] = per_class_curve[k2][-1]
            if k2 not in after_first: after_first[k2] = per_class_curve[k2][-1]     # accuracy right after the class's FIRST visit
        if ch["t"] == 0:                                        # learnability guard: an arm at chance right after its first visit is
            with torch.no_grad():                               # broken for this input, not merely forgetting. Flagged in the log.
                R["diag_first_chunk_acc"] = float((arm.predict(Xc, count=False).argmax(1) == yc).float().mean())
        snap = arm.meter.snapshot(); energy_curve.append(snap["energy_uJ"]); steps_curve.append(snap["steps"])
        visits.append(dict(t=int(ch["t"]), cls=(cs[0] if len(cs) == 1 else -1), kind=kind, shifted=bool(ch["shifted"]), pre_fam=float(pre_fam), pre_comp=float(pre_comp),
                           **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items() if not isinstance(v, torch.Tensor)}))
    R["train_time_s"] = time.time() - t0
    teS_X, teS_y = te(S); final_pc = {c: per_class_curve[c][-1] for c in S}
    R["final_acc"] = float(np.mean(list(final_pc.values())))
    R["bwt"] = float(np.mean([final_pc[c] - after_first[c] for c in S]))          # Lopez-Paz & Ranzato: final minus after the first visit
    R["bwt_recur"] = float(np.mean([final_pc[c] - after_last[c] for c in S]))     # final minus after the most recent visit (repetition variant)
    R["forgetting"] = float(np.mean([best[c] - final_pc[c] for c in S]))
    R["acc_curve"], R["energy_curve"], R["steps_curve"], R["visits"] = [float(a) for a in acc_curve], [float(e) for e in energy_curve], [int(s) for s in steps_curve], visits
    R["per_class_curve"] = {str(int(c)): [float(v) for v in cur] for c, cur in per_class_curve.items()}
    R["RIF"] = float(np.mean(rif)) if rif else 0.0
    intro = [v for v in visits if v["kind"] == "intro"]; recur = [v for v in visits if v["kind"] == "recur"]
    R["recog_auroc_fam"] = auroc([v["pre_fam"] for v in recur], [v["pre_fam"] for v in intro]); R["recog_auroc_comp"] = auroc([v["pre_comp"] for v in recur], [v["pre_comp"] for v in intro])
    with torch.no_grad():
        R["fam_auroc_test"] = auroc(arm.familiarity(teS_X).cpu().numpy(), arm.familiarity(te(Hd)[0]).cpu().numpy())
    R["steps_intro"] = float(np.mean([v["steps"] for v in intro])); R["steps_recur"] = float(np.mean([v["steps"] for v in recur]))
    R["S_stream"] = float(1 - R["steps_recur"] / max(R["steps_intro"], 1e-9)); R["frac_recur_lighter"] = float(np.mean([v["steps"] < R["steps_intro"] for v in recur]))
    if name.startswith("controller"):
        R["state_counts"] = {s: int(sum(v["state"] == s for v in recur)) for s in ("known", "partial", "shifted", "unknown")}; R["deficit_recur"] = float(np.nanmean([v["deficit"] for v in recur]))
    learned_param_bytes = sum(p.numel() * p.element_size() for p in arm.net.parameters() if p.requires_grad)
    opt = getattr(arm, "opt", None)
    optimizer_bytes = sum(v.numel() * v.element_size() for stt in opt.state.values() for v in stt.values() if isinstance(v, torch.Tensor)) if opt else 0
    adaptive_bytes = arm.standing_bytes() + learned_param_bytes + optimizer_bytes
    snap = arm.meter.snapshot(); R.update(lifetime_energy_uJ=snap["energy_uJ"], lifetime_steps=snap["steps"], lifetime_samples=snap["samples"], lifetime_spikes=snap["spikes"],
                                          lifetime_macs=snap["macs"], standing_bytes=arm.standing_bytes(), params=arm.net.n_params(),
                                          adaptive_bytes=adaptive_bytes, optimizer_bytes=optimizer_bytes,
                                          resident_bytes=arm.resident_bytes() if hasattr(arm, "resident_bytes") else
                                          arm.standing_bytes() + sum(p.numel() * p.element_size() for p in arm.net.parameters()),
                                          number_of_solves=arm.mem.solves if hasattr(arm, "mem") else 0)
    R["frontend_energy_uJ"] = 0.0
    if cfg.get("stem_macs"):                                   # a frozen front end is computed ONCE per arriving sample and cached;
        n_stream = sum(len(ch["target"]) + len(ch["context"]) for ch in st["chunks"])   # an identical constant for every method,
        R["frontend_energy_uJ"] = cfg["stem_macs"] * n_stream * cfg["pj_mac"] * 1e-6    # reported separately, not inside the
    R["total_energy_uJ"] = R["lifetime_energy_uJ"] + R["frontend_energy_uJ"]            # learning-system energy that is compared
    R["energy_per_acc_point"] = snap["energy_uJ"] / max(100 * R["final_acc"], 1e-9)
    # ----- matched S_ctrl protocol (forgotten seen classes in their CURRENT domain vs never-seen holdouts)
    last = [k2 for ch in st["chunks"][-cfg["recur_gap_min"]:] for k2 in (ch["cls"] if isinstance(ch["cls"], list) else [ch["cls"]])]
    n_ctx = max(4, min(40, min(len(st["pool"][c]) // 2 for c in last)))     # small datasets have short pools; keep a real split
    ctxX = torch.cat([apply_shift(Xt[st["pool"][c][:n_ctx]], c, seed, cfg) if dom[c] else Xt[st["pool"][c][:n_ctx]] for c in last]); ctxy = torch.cat([yt[st["pool"][c][:n_ctx]] for c in last])
    cand = sorted([c for c in S if c not in last], key=lambda c: final_pc[c])[: cfg["n_forgotten_events"]]
    ev = dict(relearn=[], novel=[], shift=[]); cz = dict(relearn=[], novel=[], shift=[]); base = copy.deepcopy(arm)
    ceil_novel = float(np.mean([best[c] for c in S]))
    for c in cand:
        pX = Xt[st["pool"][c][n_ctx:]]; pX = apply_shift(pX, c, seed, cfg) if dom[c] else pX; pY = yt[st["pool"][c][n_ctx:]]
        s, z = recovery_event(copy.deepcopy(base), c, pX, pY, ctxX, ctxy, teS_X, teS_y, S, cfg, rng, best[c]); ev["relearn"].append(s); cz["relearn"].append(z)
        if Xt.dim() == 2:                                       # feature front end: the fixed-rotation shift probe needs images; skipped
            ev["shift"].append(float("nan")); cz["shift"].append(False)
        else:
            teX2 = teS_X.clone(); m = teS_y == c
            if m.any(): teX2[m] = rotate(teX2[m], 20.0)
            s, z = recovery_event(copy.deepcopy(base), c, rotate(pX, 20.0), pY, ctxX, ctxy, teX2, teS_y, S, cfg, rng, best[c]); ev["shift"].append(s); cz["shift"].append(z)
    for c in Hd:
        cls = S + [c]; tX, tY = te(cls)
        s, z = recovery_event(copy.deepcopy(base), c, Xt[st["pool"][c]], yt[st["pool"][c]], ctxX, ctxy, tX, tY, cls, cfg, rng, ceil_novel); ev["novel"].append(s); cz["novel"].append(z)
    R["recovery_ceiling"] = float(np.mean([best[c] for c in cand])); R["ceiling_novel"] = ceil_novel
    R["S_ctrl"], R["S_ctrl_unc"] = savings(ev["relearn"], ev["novel"], cz["relearn"], cz["novel"]); R["S_shift"], R["S_shift_unc"] = savings(ev["shift"], ev["novel"], cz["shift"], cz["novel"])
    # censoring-free recovery curves: seen (forgotten) classes vs never-seen holdouts, same grid for every arm
    cur_rel = [recovery_curve(copy.deepcopy(base), c, (apply_shift(Xt[st["pool"][c][40:]], c, seed, cfg) if dom[c] else Xt[st["pool"][c][40:]]), yt[st["pool"][c][40:]], ctxX, ctxy, teS_X, teS_y, S, cfg, rng) for c in cand]
    cur_nov = []
    for c in Hd:
        cls = S + [c]; tX, tY = te(cls); cur_nov.append(recovery_curve(copy.deepcopy(base), c, Xt[st["pool"][c]], yt[st["pool"][c]], ctxX, ctxy, tX, tY, cls, cfg, rng))
    bud = cfg["recovery_budgets"]
    R["rec_curve_seen"] = {str(b): float(np.mean([d[b] for d in cur_rel])) for b in bud}
    R["rec_curve_novel"] = {str(b): float(np.mean([d[b] for d in cur_nov])) for b in bud}
    def aur(cur):
        values = np.array([np.mean([d[b] for d in cur]) for b in bud])
        return float(np.sum((values[:-1] + values[1:]) * np.diff(bud) / 2) / (bud[-1] - bud[0]))
    R["AUR_seen"], R["AUR_novel"] = aur(cur_rel), aur(cur_nov)
    R["S_AUR"] = float((R["AUR_seen"] - R["AUR_novel"]) / max(1e-9, 1 - R["AUR_novel"]))   # normalised savings in [-inf, 1]
    R["rec_at_0_seen"], R["rec_at_1_seen"] = R["rec_curve_seen"][str(bud[0])], R["rec_curve_seen"][str(bud[1])]
    R.update(E_relearn=float(np.mean(ev["relearn"])), E_novel=float(np.mean(ev["novel"])), E_shift=float(np.mean(ev["shift"])), censored=int(sum(map(sum, cz.values()))),
             censored_detail={k: int(sum(v)) for k, v in cz.items()})
    chance = 1.0 / cfg["n_stream_classes"]; R["collapsed"] = bool(R["final_acc"] <= chance + 0.02 or R.get("diag_first_chunk_acc", 1.0) < 0.35)
    if R["collapsed"]: log(f"  !! {name}/seed{seed}: COLLAPSED (final acc {R['final_acc']:.3f} vs chance {chance:.3f}); included in all results")
    log(f"  {name:17s} s{seed} {cfg['dataset']:7s} {cfg['shift_mode']:6s} | acc {R['final_acc']:.2f} bwt {R['bwt']:+.2f} | recur {R['steps_recur']:.1f}/{R['steps_intro']:.0f} steps | "
        f"S_AUR {R['S_AUR']:+.2f} (S_ctrl {R['S_ctrl_unc']:+.2f}) RIF {R['RIF']:+.3f} | AUROC {R['recog_auroc_fam']:.2f} | {R['lifetime_energy_uJ']:.0f} uJ {R['standing_bytes']/1024:.1f}KB")
    return R


def run_grid(cfg, log=print, X=None, y=None):
    cfg = apply_dataset_defaults(cfg, cfg["dataset"])
    if X is None: X, y = load_dataset(cfg["dataset"], cfg["data"], cfg["data_root"], cfg)
    Xraw = None
    if X.ndim == 4 and cfg.get("stem", "conv") == "resnet18" and cfg["dataset"] in cfg.get("stem_resnet_datasets", ["cifar10", "cifar100"]):
        from .dataloader.loaders import pretrained_features
        Xraw = X; X, macs, stats = pretrained_features(X, cfg, cfg["dataset"], cfg["data_root"]); cfg = dict(cfg); cfg["stem_macs"] = macs; cfg["_feat_stats"] = stats
        keep = ~np.isnan(X[:, 0]); X, y, Xraw = X[keep], y[keep], Xraw[keep]                # the normalisation subset never enters a stream or test set
    elif X.ndim == 4 and cfg.get("conv_stem", True) and (X.shape[-3] > 1 or cfg.get("conv_stem_gray", True)):
        from .dataloader.loaders import as_unit_float
        Z, macs = stem_transform(torch.tensor(as_unit_float(X)), cfg); X = Z.numpy(); cfg = dict(cfg); cfg["stem_macs"] = macs   # uint8 (Tiny-ImageNet) -> [0, 1]
    os.makedirs(cfg["out"], exist_ok=True); t0 = time.time()
    path = f"{cfg['out']}/rows.json"
    rows = json.load(open(path)) if cfg.get("resume", True) and os.path.isfile(path) else []
    match_keys = ("rw_hidden", "rw_sketch", "hidden", "rw_lambda", "full_steps", "n_recur", "chunk_context", "T", "stem")
    for r in rows:
        old = r.get("config")
        if old is None or any(old.get(k) != cfg.get(k) for k in match_keys):
            raise ValueError(f"Incompatible earlier results in {path}. Use a fresh --out; never mix protocols.")
    if len({(r["seed"], r["arm"]) for r in rows}) != len(rows):
        raise ValueError(f"Duplicate seed/method rows in {path}; inspect before resuming")
    done = {(r["seed"], r["arm"]) for r in rows}
    failures = []
    for seed in cfg["seeds"]:
        if Xraw is not None and cfg["shift_mode"] != "none":
            # domain shift acts on IMAGES; with a feature front end the shifted features are extracted once per seed (the transform is
            # seed-specific) and apply_shift maps each feature row to its shifted twin through a lookup table
            from .dataloader.loaders import pretrained_features, shifted_features
            cfg = dict(cfg); cfg["_shift_lut"], cfg["_shifted"] = shifted_features(Xraw, X, seed, cfg, cfg["dataset"], cfg["data_root"])
        for name in cfg["arms"]:
            if (seed, name) in done: continue
            try:
                rows.append(run_one(name, seed, X, y, cfg, log)); done.add((seed, name))
            except Exception as e:
                import traceback; traceback.print_exc(); log(f"  !! {name}/seed{seed} failed: {e!r}"); failures.append((seed, name, repr(e)))
            tmp = path + ".tmp"
            with open(tmp, "w") as f: json.dump(rows, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            os.replace(tmp, path)
    if failures: raise RuntimeError(f"{len(failures)} failed runs in {cfg['out']}: {failures}")
    log(f"done: {len(rows)} runs in {time.time() - t0:.0f}s")
    return rows


def run_all(cfg, log=print):
    rows = run_grid(cfg, log)
    from .utils.plotting import plot_all
    return plot_all(rows, cfg)


def plot_saved(out="results"):
    from .utils.plotting import plot_all
    with open(f"{out}/rows.json") as f: rows = json.load(f)
    cfg = dict(DEFAULT); cfg["out"] = out
    if rows: cfg["dataset"], cfg["shift_mode"] = rows[0].get("dataset", cfg["dataset"]), rows[0].get("shift_mode", "none")
    return plot_all(rows, cfg)
