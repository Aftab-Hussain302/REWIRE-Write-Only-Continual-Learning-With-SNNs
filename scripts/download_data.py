#!/usr/bin/env python3
"""Download the six datasets of the paper into the layout the loaders expect (see DATASETS.md).

    python3 scripts/download_data.py --data_root ~/data                 # everything that is missing
    python3 scripts/download_data.py --data_root ~/data --datasets shd nmnist

`scripts/prepare_data.py` calls this automatically for every dataset it cannot find, so a manual download is never required;
it only saves time (the archives are 0.2-1.1 GB each and the public servers are not fast).

Sources (public, the official releases):
  MNIST, CIFAR-10, CIFAR-100   torchvision (http://yann.lecun.com/exdb/mnist, https://www.cs.toronto.edu/~kriz/cifar.html)
  N-MNIST                      Orchard et al. 2015, Train.zip of the official release (the Dropbox link published with the dataset)
  SHD                          Cramer et al. 2022, shd_train.h5.gz from https://zenkelab.org/datasets/
  Tiny-ImageNet                the Stanford CS231n release, http://cs231n.stanford.edu/tiny-imagenet-200.zip

Nothing is modified after extraction: the loaders read the official files as they are."""
import argparse, glob, gzip, os, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

URLS = {
    "nmnist": ["https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxia/Train.zip?dl=1"],
    "shd": ["https://zenkelab.org/datasets/shd_train.h5.gz", "https://compneuro.net/datasets/shd_train.h5.gz"],
    "tinyimagenet": ["http://cs231n.stanford.edu/tiny-imagenet-200.zip"],
}
MANUAL = {
    "nmnist": "https://www.garrickorchard.com/datasets/n-mnist  (Train.zip; unzip so that <root>/nmnist/Train/<digit>/*.bin exists)",
    "shd": "https://zenkelab.org/datasets/  (shd_train.h5.gz; gunzip to <root>/shd/shd_train.h5)",
    "tinyimagenet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip  (unzip to <root>/tiny-imagenet-200/)",
}


def present(ds, root):
    r = {"mnist": os.path.isfile(os.path.join(root, "MNIST/raw/train-images-idx3-ubyte")),
         "cifar10": os.path.isfile(os.path.join(root, "cifar-10-batches-py/data_batch_1")),
         "cifar100": os.path.isfile(os.path.join(root, "cifar-100-python/train")),
         "nmnist": len(glob.glob(os.path.join(root, "nmnist/Train/0/*.bin"))) > 0,
         "shd": os.path.isfile(os.path.join(root, "shd/shd_train.h5")),
         "tinyimagenet": os.path.isfile(os.path.join(root, "tiny-imagenet-200/wnids.txt")) and len(glob.glob(os.path.join(root, "tiny-imagenet-200/train/*/images/*.JPEG"))) > 0}
    return r[ds]


def fetch(url, dest):
    """Stream `url` to `dest` with a progress line; a partial file is removed on failure."""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (REWIRE data download)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0; t0 = time.time(); last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk); done += len(chunk)
            if time.time() - last > 2:
                last = time.time(); mb = done / 1e6; rate = mb / max(time.time() - t0, 1e-6)
                print(f"\r   {mb:8.1f} MB" + (f" of {total/1e6:.1f} MB" if total else "") + f"  ({rate:.1f} MB/s)", end="", flush=True)
    print(); os.replace(tmp, dest)


def download(ds, root):
    os.makedirs(root, exist_ok=True)
    if ds in ("mnist", "cifar10", "cifar100"):
        import torchvision
        cls = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[ds]
        cls(root, train=True, download=True); cls(root, train=False, download=True); return
    last = None
    for url in URLS[ds]:
        try:
            name = url.split("/")[-1].split("?")[0]; arc = os.path.join(root, name)
            print(f"   downloading {url}", flush=True); fetch(url, arc)
            if ds == "nmnist":
                out = os.path.join(root, "nmnist"); os.makedirs(out, exist_ok=True)
                with zipfile.ZipFile(arc) as z: z.extractall(out)
                if not glob.glob(os.path.join(out, "Train/0/*.bin")):          # some archives nest a top-level folder
                    inner = glob.glob(os.path.join(out, "*", "Train"))
                    if inner: shutil.move(inner[0], os.path.join(out, "Train"))
            elif ds == "shd":
                out = os.path.join(root, "shd"); os.makedirs(out, exist_ok=True)
                with gzip.open(arc, "rb") as fin, open(os.path.join(out, "shd_train.h5"), "wb") as fout: shutil.copyfileobj(fin, fout, 1 << 24)
            elif ds == "tinyimagenet":
                with zipfile.ZipFile(arc) as z: z.extractall(root)
            os.remove(arc)
            if present(ds, root): return
            last = RuntimeError(f"archive from {url} did not produce the expected layout")
        except Exception as e:                                                  # try the next mirror
            last = e; print(f"   failed: {e!r}", flush=True)
            for p in glob.glob(os.path.join(root, "*.part")): os.remove(p)
    raise RuntimeError(f"could not download {ds}: {last}. Download it by hand from {MANUAL[ds]}")


def ensure(datasets, root, quiet=False):
    """Download every dataset in `datasets` that is not already under `root`. Returns the list that was downloaded."""
    got = []
    for ds in datasets:
        if present(ds, root):
            if not quiet: print(f"== {ds}: found under {root}")
            continue
        print(f"== {ds}: not found under {root}; downloading (this can take a while on the public servers)", flush=True)
        download(ds, root); got.append(ds); print(f"== {ds}: ready", flush=True)
    return got


def main():
    from rewire.configure import PAPER_DATASETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True); ap.add_argument("--datasets", nargs="+", default=PAPER_DATASETS)
    a = ap.parse_args(); ensure(a.datasets, os.path.expanduser(a.data_root))


if __name__ == "__main__": main()
