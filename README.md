# Reliability Failure of RUL Prediction Intervals under Sensor Degradation

Code, checkpoints and results behind:

> Y. Wang, "Reliability Failure of Remaining-Useful-Life Prediction Intervals
> under Sensor Degradation: A Leakage-Audited Evaluation, Mean-Scale
> Attribution, and Maintenance-Decision Consequences" (IEEE Transactions on
> Reliability, in review).

## Layout

- `paper/` -- `main.tex`, `supplementary.tex`, `refs.bib`, `IEEEtran.cls`,
  the compiled `main.pdf`/`supplementary.pdf`, and `submission/` (cover
  letter and submission checklist).
- `code/` -- every script needed to reproduce the results below, given the
  checkpoints and the NASA C-MAPSS data. File names describe function
  (`run_sweep_noise_lstm.py`, `attribution_interp_lstm.py`,
  `maintenance_stress_test_rul_correction_v2.py`, ...) rather than an
  internal round number. `common.py` / `noise_injection.py` /
  `transformer_common.py` / `sweep_engine.py` / `interval_score.py` are
  shared modules imported throughout.
- `results/canonical_splits.json` -- the engine-level fit/val/calib
  partition (60/20/20 by engine), one per (dataset, seed) -- 5 seeds x 4
  datasets = 20 independent partitions, not one partition shared across
  seeds.
- `results/checkpoints/` -- trained model weights: `lstm/` and
  `transformer/` (5 seeds x 4 datasets each, the checkpoints behind every
  reported number), `lstm_drift_controls/` (used only by the drift-control
  scripts) and `samesplit_control/` (four additional same-partition,
  different-initialisation checkpoints for the seed-variance control in
  the appendix).
- `results/{clean,degradation,attribution,decision,latency,seeds}/` --
  result files grouped by what they evaluate: clean-input benchmarks and
  cost tables; noise/bias/gain/drift sweeps and their crossover points;
  the mean/scale attribution decomposition; maintenance-decision stress
  tests; latency measurements; and seed-variance/ensemble-size controls.
  `results/table_provenance.csv` documents which script and result file
  back every table in `main.tex`.
- `results/superseded/` -- the three pre-audit results referenced in the
  paper's Appendix A (Audit History): the MC-Dropout baseline that omitted
  the aleatory variance term, the ensemble evaluation that gave each
  member its own noise realisation, and the MSE-proxy mean-only
  counterfactual. See `results/superseded/README.md`.
- `LICENSE` -- MIT.

## Reproducing a result

1. Set `RUL_DATA_DIR` to a local copy of the NASA C-MAPSS dataset
   (`train_FD00X.txt` / `test_FD00X.txt` / `RUL_FD00X.txt` for X in
   1,2,3,4), or place it at `<repo>/data/`. C-MAPSS is available from the
   NASA Prognostics Center of Excellence and is not redistributed here.
2. `code/canonical_splits.py` generates `results/canonical_splits.json`.
3. `code/train_lstm.py`, `code/train_lstm_mcdropout.py`,
   `code/train_lstm_split_cp.py`, `code/train_lstm_fd003_nll_and_mechanism.py`,
   `code/train_lstm_fd003_mse_mcdropout.py`, `code/train_transformer_nll.py`,
   `code/train_transformer_msemcd.py`, `code/train_transformer_split_cp.py`
   (re)train the checkpoints under the leakage-audited protocol (already
   provided under `results/checkpoints/`; retraining is not required to
   reproduce downstream results and is not asserted bit-reproducible --
   see below).
4. Every other script under `code/` consumes those checkpoints and
   `canonical_splits.json` to produce one or more files under `results/`;
   each script's own docstring states which paper table/section it feeds
   and which upstream files it depends on. All scripts write into
   `results/generated/` (created on first run, mirroring the working
   layout used during development); a result file's final, released name
   and location under `results/{category}/` is documented in
   `results/table_provenance.csv` and in each script's docstring.
5. Every script that injects randomness requires `PYTHONHASHSEED=0` (it
   will refuse to run otherwise) and derives its seeds from a SHA-256
   digest of stable identifiers (dataset, condition, trial, ...), never
   from Python's salted `hash()`. **Do not change any string passed to
   `stable_seed(...)` calls** -- doing so changes every downstream noise
   realisation.

## Scope of the reproducibility guarantee

Two independent reruns of the noise-dependent pipeline, on the same
machine (Python 3.12.3, PyTorch 2.11.0+cu130/CUDA 13.0, cuDNN 9.1.9,
`PYTHONHASHSEED=0`), reproduced 31 result files bit-for-bit; see the
project's own audit trail for the three reproducibility issues found and
fixed along the way (Python's salted `hash()`, cuDNN's non-deterministic
algorithm selection, and unseeded MC-Dropout sampling). Model training
(checkpoint weights depend on training-time CUDA kernel selection),
per-sample latency timing, and cross-platform reproduction are outside
that guarantee. Six result files (`table2_clean_full.json` and the
pooled-ECE aggregate under `results/clean/`, plus the three drift/D1
controls and the maintenance-decision family under `results/decision/`)
are pure aggregations of other, already-verified result files rather than
new stochastic computations, and are checked for exact (not bit-for-bit
file) agreement instead.

## Environment

- Python 3.12.3, PyTorch 2.11.0+cu130 (CUDA build 13.0), cuDNN 9.1.9
- GPU used for all reported numbers: NVIDIA RTX A5000 (Laptop), driver
  596.47, 16 GB VRAM
- OS: Linux (WSL2/Ubuntu), kernel 6.18

Different PyTorch/cuDNN/GPU combinations are not guaranteed to reproduce
the exact checksums above even with `PYTHONHASHSEED=0`; cuDNN's
deterministic algorithms are deterministic for a given library version and
GPU architecture, not necessarily across versions.

## License

MIT (see `LICENSE`). NASA C-MAPSS is not redistributed here; obtain it from
the NASA Prognostics Center of Excellence.
