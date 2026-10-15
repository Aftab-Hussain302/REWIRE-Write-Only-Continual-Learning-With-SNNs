# Datasets

Datasets are not bundled. Put the six under one root (`--data_root`; the campaign used `~/data`) in the layout below, or let
`python3 scripts/prepare_data.py --data_root <root>` fetch whatever is missing: it downloads the official archives
(`scripts/download_data.py`; torchvision for MNIST and CIFAR) and unpacks them into this layout. The automatic path is slower
than a manual download only because the public servers are slow (0.2-1.1 GB per archive); `--no-download` disables it.

| dataset | folder under the data root | source |
|---|---|---|
| MNIST | `MNIST/raw/` (the four idx files, decompressed) | http://yann.lecun.com/exdb/mnist/ ; torchvision downloads it if absent |
| CIFAR-10 | `cifar-10-batches-py/` | `cifar-10-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html (torchvision downloads it if absent) |
| CIFAR-100 | `cifar-100-python/` | `cifar-100-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html (torchvision downloads it if absent) |
| N-MNIST | `nmnist/Train/<digit>/*.bin` (and `nmnist/Test/...`, unused) | Orchard et al. 2015, the official `Train.zip`: https://www.garrickorchard.com/datasets/n-mnist |
| SHD | `shd/shd_train.h5` | Cramer et al. 2022, `shd_train.h5.gz` from https://zenkelab.org/datasets/ (gunzip it) |
| Tiny-ImageNet | `tiny-imagenet-200/wnids.txt`, `tiny-imagenet-200/train/<wnid>/images/*.JPEG` | the official `tiny-imagenet-200.zip`: http://cs231n.stanford.edu/tiny-imagenet-200.zip |

`python3 scripts/prepare_data.py --data_root <root>` loads every dataset once through the real loaders, reports what it found,
and builds the frozen front-end feature caches (below) before the parallel campaign starts. A dataset that cannot be loaded is
reported as an error; nothing is ever substituted for it.

N-MNIST `.bin` files and the SHD `.h5` file are parsed by the loaders directly (no event-data library; `h5py` is needed).
N-MNIST events are binned into T = 6 frames of 2 x 34 x 34; SHD spike trains into T = 20 frames of 700 channels. Only the samples
the protocol needs are materialised (train + test per class, see `rewire/configure.py`).

CIFAR-10, CIFAR-100 and Tiny-ImageNet are presented to every method through the same frozen ImageNet-pretrained ResNet-18
(torchvision `IMAGENET1K_V1`, downloaded on first use). Its 512-d features are extracted once per dataset and cached as
`<root>/_features_resnet18_<dataset>_<N>.npz`; feature standardisation uses a fixed 10 % subset of the images, which is then
removed from every stream and test set. Tiny-ImageNet is read as 64 x 64 uint8 images (1.2 GB for 100k images) and scaled per
batch inside the ResNet pass. Tiny-ImageNet overlaps the front end's pretraining data; the paper says so and anchors the 80-class
claims on CIFAR-100.

## Memory note for free GPU runtimes (Colab, Kaggle)

MNIST, N-MNIST, SHD, CIFAR-10 and CIFAR-100 fit a free runtime one dataset at a time. Tiny-ImageNet keeps 100 000 images of
64 x 64 pixels (1.2 GB) in memory while the frozen ResNet-18 extracts their features, which a free runtime often cannot hold
next to PyTorch; use the smaller streams there, reduce the per-class sample count (`--set n_train_per_class=200
n_test_per_class=50`), or use a high-RAM runtime.
