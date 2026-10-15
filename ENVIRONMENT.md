# Environment

`scripts/run_all.py` overwrites this file inside every run folder with the machine it ran on (GPU, driver, CUDA, Python, torch,
`pip freeze`) and writes `MANIFEST.json` with a SHA-256 of every source file, so a result folder identifies the code that produced it.

The campaign reported in the paper ran on one NVIDIA RTX A6000 (48 GB), 6 CPU cores, Ubuntu, with the versions recorded in the
run folder's `ENVIRONMENT.md`; the earlier development pass used one NVIDIA A100 and Google Colab T4 instances.
