"""Loader of a finished run's rows.json files into one flat table (used by the preflight and the reporting tools)."""
import glob, json
import pandas as pd
from .models.arms import ARMS

ORDER = list(ARMS)
SCALARS = ("dataset", "shift_mode", "arm", "seed", "n_recur", "theta_hi", "final_acc", "bwt", "bwt_recur", "forgetting", "recog_auroc_fam", "fam_auroc_test",
           "lifetime_energy_uJ", "frontend_energy_uJ", "total_energy_uJ", "standing_bytes", "resident_bytes", "number_of_solves", "hidden_width", "sketch_width",
           "feature_dim", "collapsed", "S_AUR", "AUR_seen", "AUR_novel", "RIF", "energy_per_acc_point", "train_time_s")


def load(res, pattern="*/shift_*/rows.json"):
    rows = []
    for f in sorted(glob.glob(f"{res}/{pattern}")):
        for r in json.load(open(f)): rows.append({k: r.get(k) for k in SCALARS})
    d = pd.DataFrame(rows)
    if len(d): d["collapsed"] = d.collapsed.fillna(False).astype(bool)
    return d
