#!/usr/bin/env python3
"""Check the data folder, load every dataset once through the real loaders, and build the frozen front-end feature caches
sequentially (so that the parallel campaign never computes them twice).

    python3 scripts/prepare_data.py --data_root ~/data              # all six datasets
    python3 scripts/prepare_data.py --data_root ~/data --datasets mnist shd

Expected layout under --data_root (see DATASETS.md):
  MNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte      cifar-10-batches-py/    cifar-100-python/
  nmnist/Train/<digit>/*.bin  (nmnist/Test/... optional)       shd/shd_train.h5        tiny-imagenet-200/{wnids.txt, train/<wnid>/images/*.JPEG}
A dataset that is absent is downloaded automatically (torchvision for MNIST / CIFAR, scripts/download_data.py for N-MNIST, SHD\nand Tiny-ImageNet) when the machine is online; pass --no-download to disable this. Writes <data_root>/DATASETS_READY.json."""
import argparse, glob, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from rewire.configure import config_for, PAPER_DATASETS, DATASET_NAME     # noqa: E402

LAYOUT = {"mnist": ("MNIST/raw/train-images-idx3-ubyte", os.path.isfile), "cifar10": ("cifar-10-batches-py/data_batch_1", os.path.isfile),
          "cifar100": ("cifar-100-python/train", os.path.isfile), "nmnist": ("nmnist/Train/0", lambda p: len(glob.glob(p + "/*.bin")) > 0),
          "shd": ("shd/shd_train.h5", os.path.isfile), "tinyimagenet": ("tiny-imagenet-200/wnids.txt", lambda p: os.path.isfile(p) and len(glob.glob(os.path.dirname(p) + "/train/*/images/*.JPEG")) > 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True); ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS); ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="do not fetch missing datasets (by default they are downloaded, see scripts/download_data.py)")
    a = ap.parse_args(); root = os.path.expanduser(a.data_root); report = {}
    from rewire.dataloader import load_dataset
    from rewire.dataloader.loaders import pretrained_features
    for ds in a.datasets:
        rel, test = LAYOUT[ds]; p = os.path.join(root, rel); ok = bool(test(p)); t0 = time.time()
        if not ok and not a.no_download:                                     # fetch the official archive into the expected layout
            print(f"\n== {DATASET_NAME[ds]}: not found at {p}; downloading it (automatic; a manual download beforehand is faster)", flush=True)
            try:
                from download_data import ensure
                ensure([ds], root, quiet=True); ok = bool(test(p))
            except Exception as e:
                print(f"   download failed: {e!r}", flush=True)
        print(f"\n== {DATASET_NAME[ds]}: {'found' if ok else 'NOT FOUND'} at {p}", flush=True)
        try:
            cfg = config_for(ds); X, y = load_dataset(ds, cfg["data"], root, cfg)
            info = dict(found=ok, samples=int(len(y)), classes=int(len(set(y.tolist()))), shape=list(X.shape[1:]), dtype=str(X.dtype))
            if not a.no_cache and ds in cfg["stem_resnet_datasets"] and cfg.get("stem") == "resnet18":
                F, macs, _ = pretrained_features(X, cfg, ds, root); info["feature_cache"] = f"_features_resnet18_{ds}_{len(X)}.npz"; info["features"] = list(F.shape)
            info["seconds"] = round(time.time() - t0, 1); report[ds] = info; print(f"   ok: {info}", flush=True)
        except Exception as e:
            report[ds] = dict(found=ok, error=repr(e)); print(f"   ERROR: {e!r}", flush=True)
    json.dump(report, open(os.path.join(root, "DATASETS_READY.json"), "w"), indent=1)
    bad = [d for d, r in report.items() if "error" in r]
    print("\n" + ("ALL DATASETS READY" if not bad else f"NOT READY: {bad}") + f"  (report in {root}/DATASETS_READY.json)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__": main()
