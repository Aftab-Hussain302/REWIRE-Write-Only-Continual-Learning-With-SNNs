"""REWIRE: write-only continual learning on a frozen spiking substrate, for streams in which classes recur."""
from .configure import DEFAULT, PRESETS, DATASETS, config_for, apply_dataset_defaults  # noqa: F401
from .models.arms import ARMS, PAPER_ARMS, build  # noqa: F401
