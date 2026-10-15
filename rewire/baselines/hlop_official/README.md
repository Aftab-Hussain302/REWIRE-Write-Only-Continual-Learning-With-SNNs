# HLOP's Hebbian module (third-party code)

`hlop_module.py` in this folder was **not written by the REWIRE authors**. It is taken, unmodified, from the official code
release of the HLOP paper, [github.com/pkuxmq/HLOP-SNN](https://github.com/pkuxmq/HLOP-SNN) (maintained by Mingqing Xiao).

| Item | Details |
|---|---|
| Source | [`modules/hlop_module.py`](https://github.com/pkuxmq/HLOP-SNN/blob/530b89aa19ca1b8e31bd43cf3af8db14a229da64/modules/hlop_module.py) in [pkuxmq/HLOP-SNN](https://github.com/pkuxmq/HLOP-SNN), commit `530b89a` (20 February 2024) |
| Paper | Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Di He, Zhouchen Lin, "Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks", ICLR 2024 ([OpenReview](https://openreview.net/forum?id=MeB86edZ1P), [arXiv:2402.11984](https://arxiv.org/abs/2402.11984)) |
| Licence | Apache License 2.0; `LICENSE` in this folder is the licence file of that repository, copied unchanged |
| Changes | none: the file is byte-identical to the source (SHA-256 `cc12fa88abc39b0bdb3985066cd9902dc1c73f49a5d1f949342dd47d2ace7faf`) |

All credit for this module belongs to its authors. Only `__init__.py` (one import line), `NOTICE` and this README in this
folder are ours.

**What comes from the authors and what is ours.** The HLOP baseline learns its subspaces with the authors' own Hebbian rule:
it calls `add_subspace`, `forward_with_update`, `merge_subspace` and `get_weight_value` of this module, on the first task
without projection and on every later new-class chunk as a new block with the old blocks fixed, followed by a merge. Everything
else in the baseline is our code. The training loop in `rewire/models/arms.py` (class `HLOP`) follows the task structure of the
authors' `spiking_train_pmnist.py`, but its settings (optimiser, layer widths, subspace sizes, learning rates; see
`rewire/configure.py`) are ours and differ from the authors' setup. The gradient projection uses an orthonormal (QR) basis of
the authors' learned subspace (`_sync_projection` in `rewire/models/arms.py`, `ProjLinearFn` in `rewire/modules/neuron.py`)
instead of the module's own `projection()`, and is applied to the weight gradients. The adaptation of their task- and
domain-incremental method to class-incremental streams with repetition is ours and is described in `BASELINES.md`.

**Checking the copy against the source**, from the repository root:

```bash
# git blob ID, the same on every OS; upstream reports 7e59e7d555d377443baea6d06d90d65ab09ca994
git hash-object rewire/baselines/hlop_official/hlop_module.py
gh api "repos/pkuxmq/HLOP-SNN/contents/modules/hlop_module.py?ref=530b89aa19ca1b8e31bd43cf3af8db14a229da64" --jq .sha

# or SHA-256 of the bytes (Linux, Colab; on macOS use `shasum -a 256`)
curl -sL https://raw.githubusercontent.com/pkuxmq/HLOP-SNN/530b89aa19ca1b8e31bd43cf3af8db14a229da64/modules/hlop_module.py | sha256sum
sha256sum rewire/baselines/hlop_official/hlop_module.py
```

If you use the HLOP baseline, please cite the authors' paper:

```bibtex
@inproceedings{xiao2024hebbian,
  title     = {Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks},
  author    = {Xiao, Mingqing and Meng, Qingyan and Zhang, Zongpeng and He, Di and Lin, Zhouchen},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=MeB86edZ1P}
}
```
