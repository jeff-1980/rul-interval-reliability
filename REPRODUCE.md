# Reproducing the noise-dependent result files

This document covers bit-for-bit reproduction of the **32 result files**
listed below: the ones whose generation involves injected randomness
(noise/bias/gain injection, MC-Dropout sampling), plus two pure
aggregations of already-verified files (`clean/table2_clean_full.json` and
`clean/lstm_clean_ece_pooled.json`) whose generating scripts had been lost
and were reconstructed and checked to exact agreement against the existing
files. It does not claim that "all files" in this repository are
reproducible this way -- checkpoints, per-engine coverage, latency
measurements, and the maintenance-decision family are out of scope here,
either because they were never affected by the reproducibility issues
below or because reproducing them exactly would also require pinning
training-time non-determinism (checkpoint weights) that this document does
not attempt to characterise.

## Scope: what is and isn't guaranteed

**Reproducible (verified bit-identical across two independent full
reruns)**: the first 31 files below, for every mechanism except
MC-Dropout's own live T=50 sampling, which is reproducible only if you
re-run the unmodified code with the seed it itself derives via
`stable_seed(...)` at each call site -- there is no way to recover the
specific MC-Dropout sampling used in results generated before this
reproducibility fix (that randomness was never seeded and the seed was
never logged).

**Reproducible to exact (not necessarily bit-for-bit) numerical agreement**:
the last 2 files (`table2_clean_full.json`, `lstm_clean_ece_pooled.json`),
which are deterministic aggregations/re-derivations of other files in this
repository, not new stochastic computations.

**Not covered by this document**: model training (checkpoint weights
depend on training-time CUDA kernel selection and are not asserted
reproducible here), any file not in the list below, and anything under
`results/superseded/`.

## Environment used to produce the checksums below

- Python 3.12.3
- PyTorch 2.11.0+cu130 (CUDA build 13.0), cuDNN 9.1.9 (`torch.backends.cudnn.version()` = 91900)
- GPU: NVIDIA RTX A5000 (Laptop), driver 596.47, 16 GB VRAM
- OS: WSL2/Ubuntu (Linux 6.18 kernel)
- `PYTHONHASHSEED=0` (hard-asserted by `common.require_fixed_hashseed()` at
  the top of every script listed below -- the script refuses to run
  without it)

Different PyTorch/cuDNN/GPU combinations are not guaranteed to reproduce
these exact checksums even with `PYTHONHASHSEED=0`; cuDNN's deterministic
algorithms are deterministic for a given library version and GPU
architecture, not necessarily across versions. What is guaranteed across
environments is that `stable_seed(...)` itself (a `hashlib.sha256` digest)
always returns the same integer for the same inputs.

## How to reproduce

```bash
cd code
export PYTHONHASHSEED=0
export RUL_DATA_DIR=/path/to/cmapss   # or place the data at ../data/
bash run_pipeline.sh myrun
```

This runs, in order, the scripts that generate the 32 files below (see
`run_pipeline.sh` for the exact list and grouping) using the checkpoints
already provided under `results/checkpoints/`. Every script writes into
`results/generated/`; `publish_results.py` then copies each output to its
released name and location under `results/{category}/` (see
`results/table_provenance.csv` for the mapping). Total wall time on the
environment above: approximately 40 minutes.

To verify reproducibility yourself: run the sequence twice
(`run_pipeline.sh runA` then `run_pipeline.sh runB`, snapshotting the 32
released files between runs), then compare MD5s -- this is exactly what
`code/check_md5.py` does (edit its snapshot directory to point at your two
runs' outputs).

## The 32 files and their MD5 (this environment, this codebase revision)

```
46abfd892187522bbb0611b6d1951ec9  results/degradation/lstm_noise_main_arm.json
aaa9b334535806fff1613b7b4c1e41f6  results/degradation/lstm_noise_armc.json
47a4960c08c68ef5fed5905f5bd51b4d  results/degradation/lstm_fd003_noise_main_arm.json
1f778e1d03a712fc2345e425e5f6f9d1  results/degradation/lstm_fd003_noise_armc.json
a59387fab272a1216b1fbcc7d8b023d6  results/degradation/dose_response_feat_oob.json
324f04b32eabfeed801a431de8c19473  results/degradation/lstm_frozen_sigma_decomposition.json
4eb53cbdbaa9139db82eb2db4a9a4ce8  results/degradation/lstm_frozen_sigma_decomposition_relative.json
e02638c5096b626531851d9c92710ea4  results/degradation/lstm_clamp_frac_main_arm.json
ba7681d1f6f31704ed607990930ed4e5  results/degradation/lstm_clamp_frac_armc.json
c84cb6b8520970a737395dfbf891d509  results/degradation/lstm_relative_half_life.json
4f16d9f16a8ea65c1080e8fae8b647fd  results/degradation/transformer_noise_main_arm.json
de8c8f3a923ae3e3a3c81a56de6ec12e  results/degradation/transformer_noise_armc.json
c09eb1aa2538654e2815a2ce6757733f  results/degradation/transformer_half_life.json
0913a145af62ab6e4dc898b167b382b4  results/degradation/transformer_clamp_frac.json
00809bfeb014f455e0a11ce90ee045d2  results/degradation/transformer_frozen_sigma_decomposition.json
18ce095b914a3ed65ffb1eee68cc27cb  results/degradation/degradation_sweep_bias_gain_drift.json
9b9cede1b181b830fb8521082c3b6ce5  results/degradation/degradation_half_life_bias_gain_drift.json
5829b500dfca35de74f58733130b0aeb  results/attribution/frozen_output_decomposition_2x2_nearest_grid.json
2e71fdf04735a8da6f92ded6c6d65bb6  results/attribution/attribution_raw_per_seed_nearest_grid.json
0eef9f64dca63fe99faff67c442223b9  results/attribution/attribution_transformer_exact_interp.json
31304f5ad41bd054d3ce23662903adb3  results/attribution/attribution_lstm_exact_interp.json
9f3bcc885d8c25fbaa5f24cf048f1dd1  results/degradation/threshold_crossover_refinement.json
b68580fc47ea378d03ea973025d8dff4  results/degradation/threshold_crossover_leftside_extension.json
b643b1f4ab4f889459e009c36e8afd9f  results/attribution/attribution_bootstrap_by_order_exact_interp.json
643158befa9afcdbba1cfa92206c8d28  results/degradation/drift_controls.json
d517422bda3246e4ada57683c2eec662  results/decision/maintenance_decision_two_sided.json
049412c1b9b336b38617293a5634936b  results/decision/maintenance_decision_one_sided.json
4bf44f24a82407702c0f6ff08723411a  results/degradation/drift_fixed_endpoint_shuffle_control.json
225229f0245b0d31f3515c2dc953899d  results/decision/maintenance_decision_rul_corrected_v2.json
05c0ecb17fea6dd4397d06d5d57ce9b3  results/decision/maintenance_decision_rul_correction_exact_match_check.json
09b674d0de8dbd787929946dc99d700d  results/clean/table2_clean_full.json
c2fc8c265458cd1581ba59dd985e889c  results/clean/lstm_clean_ece_pooled.json
```

30 of these 32 files were verified bit-identical across two independent
full reruns of the predecessor pipeline on 2026-09-21/22, carried over
unchanged since this release only renamed files and directories, not their
contents. The 2 reconstructed aggregation files were verified after
building this release: `lstm_clean_ece_pooled.json` reproduced bit-for-bit
(fresh checkpoint inference, no cross-file floating-point merge);
`table2_clean_full.json` reproduced to a maximum absolute difference of
7.1e-15 against the pre-existing file (not byte-identical -- floating-point
reduction-order noise from re-aggregating several already-rounded JSON
floats, nine orders of magnitude below the 1e-6 tolerance used to judge
this reconstruction), not a value discrepancy.
