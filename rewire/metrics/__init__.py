"""Metric definitions used throughout, with their sources.

ACC and BWT follow Lopez-Paz & Ranzato (NeurIPS 2017, GEM): with R[i, j] the accuracy on task j after training on task i,
ACC = mean_j R[T, j] and BWT = mean_{j<T} (R[T, j] - R[j, j]). Forgetting follows Chaudhry et al. (ECCV 2018): the drop from the
best accuracy a class ever reached to its final accuracy. Savings is the recovery-curve measure S_AUR defined in the paper,
adapted from Ebbinghaus's method of savings. Energy follows E = E_MAC * FLOPs + E_AC * SOPs with the 45 nm coefficients of
Horowitz (ISSCC 2014) as used by Spikformer (Zhou et al., ICLR 2023)."""
from .stats import wilcoxon_exact, boot_ci, holm, paired_table, pooled_table  # noqa: F401
