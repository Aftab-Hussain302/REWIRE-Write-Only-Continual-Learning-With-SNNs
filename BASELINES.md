# Baselines, controls and attribution

Every learner runs on the same stream, the same front end, the same 1024-unit width and the same cost model (`rewire/configure.py`).
The arm names below are the ones used in the code, the figures and the paper (`rewire/models/arms.py`, `ARMS`).

* `hlop`, `hlop_adaptive`, `hlop_gated`: Xiao, Meng, Zhang, He, Lin, "Hebbian Learning based Orthogonal Projection for Continual
  Learning of Spiking Neural Networks", ICLR 2024; official code: https://github.com/pkuxmq/HLOP-SNN.
  `rewire/baselines/hlop_official/hlop_module.py` is not our code: it is the authors' Hebbian module, taken unmodified from
  [`modules/hlop_module.py` at commit `530b89a`](https://github.com/pkuxmq/HLOP-SNN/blob/530b89aa19ca1b8e31bd43cf3af8db14a229da64/modules/hlop_module.py),
  Apache License 2.0 (its LICENSE, a NOTICE and a provenance README are in that folder). The baseline uses it for the Hebbian
  subspace learning; the training loop and the gradient projection (`rewire/models/arms.py`, `rewire/modules/neuron.py`)
  are ours. The published method is task- and
  domain-incremental; its use on a class-incremental stream with repetition is our adaptation: subspace capacity for every class;
  `hlop` freezes the consolidated subspace as published; `hlop_adaptive` refreshes it on recurrences; `hlop_gated` skips the update
  of a recurrence when its own held-out competence reaches theta (the competence half of REWIRE's gate, as a control).
* `replay`, `replay_iso`, `replay_gated`: rehearsal with a reservoir buffer (Chaudhry et al., "On Tiny Episodic Memories in
  Continual Learning", 2019): 20 exemplars per class; `replay_iso` holds as many raw samples as fit in REWIRE's standing memory;
  `replay_gated` is the competence-gated control.
* `ewc`: Kirkpatrick et al., PNAS 2017, with a running-mean diagonal Fisher.
* `naive`: fine-tuning without protection, the unprotected reference.
* `dsd_snn`: Han et al., IJCAI 2023, our re-implementation; used only in the fidelity gate (it fails Permuted MNIST there).
* `ctrl_slda_input`: streaming LDA on the inputs (Hayes & Kanan, "Lifelong Machine Learning with Deep Streaming Linear
  Discriminant Analysis", CVPR-W 2020) - REWIRE's readout without the spiking layer.
* `ridge_input`: ridge regression on the inputs held as sufficient statistics (the analytic class-incremental recipe of Zhuang et
  al., ACIL, NeurIPS 2022, on the front-end features).
* `ranpac`: a RanPAC-style control (McDonnell et al., NeurIPS 2023): ReLU random layers at REWIRE's widths with the ridge readout,
  no spiking layer. `ctrl_relu` is the same substitution inside REWIRE's full configuration (dual view + LDA readout).
* `hd_memory`: write-once hyperdimensional bundling (Kanerva, Cognitive Computation 2009) on REWIRE's own features.
* Energy coefficients: Horowitz, ISSCC 2014 (45 nm); the accounting convention of Zhou et al., Spikformer, ICLR 2023.
* Class-incremental learning with repetition: Hemati et al., CoLLAs 2023.
* The frozen spiking expansion with a linear readout follows the liquid-state-machine / reservoir tradition (Maass, Natschlaeger,
  Markram 2002; Jaeger 2001); the dual view (input beside the reservoir state) is standard echo-state-network practice.
