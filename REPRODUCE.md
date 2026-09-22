# Reproducing the result files

This document covers reproduction of the **74 result files** listed below
under this release's protocol (two changes from the predecessor pipeline,
both inference-level, no retraining). 73 of the 74 are `run_pipeline.sh`
outputs; the 74th (`results/diagnostics/A1_perturbation_target_diagnostic.json`)
is a one-off pre-flight diagnostic, run separately, held to the same bar
because it is now cited in the main text (see "How to reproduce" below):

- **Sensor-only perturbation on FD002/FD004.** These two datasets' feature
  columns are 3 operating-condition settings + 15 sensor channels. Every
  degradation experiment (Gaussian noise arms A/B/C, bias, gain, drift and
  its controls, threshold sweeps, attribution, the decision stress test)
  now perturbs only the 15 sensor channels, leaving the 3 setting columns
  untouched -- settings are the commanded operating regime, not sensor
  measurements subject to degradation. FD001/FD003 have no setting columns
  and are unaffected by this change. `f_oob` is likewise computed only over
  the perturbed sensor channels. `results/diagnostics/A1_perturbation_target_diagnostic.json`
  is the pre-registered check that confirmed this change was safe to make
  (see its own section below); it also retains the joint (18-column)
  perturbation's numbers for comparison, so no separate joint-perturbation
  control file is shipped.
- **Test-truth convention.** Every table (clean, degradation, attribution,
  decision, the Table I/II families) now scores against
  `min(official_RUL - 1, 125)`, matching the training/calibration label
  convention's counting origin (the last row of a run-to-failure training
  trajectory has RUL = 0). Previously the test split scored against the
  official RUL value directly. The uncapped-RUL column, where reported, is
  `official_RUL - 1` accordingly.

This list supersedes the predecessor 32-file list entirely -- every file
in that list is either included below (with updated content and checksum)
or was superseded by a file that is.

## Scope: what is and isn't guaranteed

**Reproducible (verified bit-identical across two independent reruns,
2026-09-22)**: 71 of the 74 files below, for every mechanism except
MC-Dropout's own live T=50/T=100 sampling, which is reproducible only if
you re-run the unmodified code with the seed it itself derives via
`stable_seed(...)` at each call site -- there is no way to recover the
specific MC-Dropout sampling used in results generated before this
reproducibility fix (that randomness was never seeded and the seed was
never logged).

**Reproducible except for one non-guaranteed field**: 3 files
(`clean/lstm_deep_ensemble.json`, `clean/lstm_fd003_deep_ensemble.json`,
`clean/transformer_deep_ensemble.json`) differ across the two reruns only
in their `latency_ms_per_sample` field, which is wall-clock timing and has
never been claimed reproducible (see `README.md`'s scope note). No other
field in these files differs.

**Not independently re-verified this round**:
`results/diagnostics/A2_target_alignment_diagnostic.json` (a pre-registered
diagnostic run once, before committing to this round's full rerun, to
motivate the RUL-1 convention change -- see `results/diagnostics/README.md`).
It was produced by the same inference code as the verified files above,
just not run twice this round, and is not one of the 74.

**Not covered by this document**: model training (checkpoint weights
depend on training-time CUDA kernel selection and are not asserted
reproducible here -- this includes the T/W and V/W checkpoints under
`code/superseded/`, which are shipped as fixed weights, not
retrainable-to-match), any file not listed below, and anything under
`results/superseded/` or `code/superseded/`.

## Two root causes fixed this round (found via the double-run diagnosis)

The first attempt at the two-independent-reruns check for this round
surfaced 15 of 71 files differing between runs. Per this project's
standing rule that any file inconsistent across two runs is unfixed
output (not something to paper over by rerunning again), both causes were
root-caused and fixed at the source, then reverified with a fresh pair of
runs:

1. **Missing cuDNN determinism guard.** 3 ensemble scripts
   (`eval_lstm_deep_ensemble.py`, `eval_lstm_fd003_deep_ensemble.py`,
   `eval_transformer_deep_ensemble.py`) never called
   `common.require_fixed_hashseed()`, so cuDNN's non-deterministic kernel
   selection leaked into their output. Fixed by adding the call.
2. **Unseeded live MC-Dropout sampling.** 5 scripts performing per-engine
   or full-trajectory MC-Dropout inference sampled T=50/100 forward passes
   without seeding `torch.manual_seed(...)` first. Fixed by deriving a
   seed via `common.stable_seed(dataset, ..., seed, tag)` before each
   sampling call.

## Environment used to produce the checksums below

- Python 3.12.3
- PyTorch 2.11.0+cu130 (CUDA build 13.0), cuDNN 9.1.9 (`torch.backends.cudnn.version()` = 91900)
- GPU: NVIDIA RTX A5000 (Laptop), driver 596.47, 16 GB VRAM
- OS: WSL2/Ubuntu (Linux 6.18 kernel)
- `PYTHONHASHSEED=0` (hard-asserted by `common.require_fixed_hashseed()` at
  the top of every script that touches the GPU -- the script refuses to
  run without it)

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

This runs, in order, the 73 `run_pipeline.sh` outputs below (see
`run_pipeline.sh` for the exact list and grouping) using the checkpoints
already provided under `results/checkpoints/` -- including the T/W and V/W
leakage-baseline checkpoints under
`results/checkpoints/lstm_leaked_test_select_whole_file_scaler/` and
`results/checkpoints/lstm_leaked_val_select_whole_file_scaler/`, needed by
`build_table1.py` (Table I), and the 10 extra-seed checkpoints under
`results/checkpoints/lstm/{ds}_LSTM_extraseed{seed}.pt`, needed by
`eval_lstm_ensemble_n5_vs_n15.py` (Appendix D). Every script writes into
`results/generated/`; the final step, `publish_results.py`, copies each
output to its released name and location under `results/{category}/` (see
`results/table_provenance.csv` for the mapping).

The 74th file, `results/diagnostics/A1_perturbation_target_diagnostic.json`,
is not part of `run_pipeline.sh` -- it is a one-off pre-flight check, run
separately:

```bash
cd code
export PYTHONHASHSEED=0
python3 diagnostic_perturbation_target_a1.py
```

To verify reproducibility yourself: run both the pipeline and this script
twice (snapshotting the 74 released files between runs), then compare
MD5s -- this is exactly what `code/check_md5.py` does (edit its snapshot
directory to point at your two runs' outputs).

## The 74 files and their MD5 (this environment, this codebase revision)

```
e44c3ba1cb450a033c38d5265fa7b6da  results/attribution/attribution_bootstrap_by_order_exact_interp.json
a44cf0d66ac3067777be780365f7f6ba  results/attribution/attribution_lstm_exact_interp.json
017873ed397f23956bfa86bb8194292c  results/attribution/attribution_raw_per_seed_nearest_grid.json
633143b2898fd6ba2a1ba2a0cc05f340  results/attribution/attribution_transformer_exact_interp.json
1b4a829da5e7f9d3e7c5fe6a5686205b  results/attribution/frozen_output_decomposition_2x2_nearest_grid.json
4ebcead6a9efe1dd9be2e802d50363f5  results/clean/fair_calibration_main_table.json
74af1cb26e7d399f63ab2abea97cb4e7  results/clean/fair_calibration_sigma_fixed.json
4e94153e156f3e364b40d7e658d556f4  results/clean/interval_score_wis_clean.json
b5cef65b1ca68222953f281b66772e2b  results/clean/lstm_clean_ece_pooled.json
257121a8c7298b25f8fa095146b68b10  results/clean/lstm_clean_per_seed.json
84ff29024b6f3cd353db731285980074  results/clean/lstm_cost_table_fd001.csv
741a250e4e0d199b068e843b9f1cb473  results/clean/lstm_cost_table_fd002.csv
ca1080cd46b1ed13b42f71597a60a54e  results/clean/lstm_cost_table_fd003.csv
53d726f2cb6116dd93f4e871114ebe74  results/clean/lstm_cost_table_fd004.csv
aae737c5b2ebbe7ce2a112f8ccfbcfdc  results/clean/lstm_deep_ensemble.json  [latency-exempt]
f626e302ed56853b26157d0f4fd1950c  results/clean/lstm_fd003_clean_per_seed.json
ad6a7e3f629d9a0e65e3b20c2efc83a8  results/clean/lstm_fd003_deep_ensemble.json  [latency-exempt]
49feffe11c5ce4448b486b7177520803  results/clean/lstm_fd003_mcdropout_mse_per_seed.json
b767bdd8397c053ad09baf08d6638755  results/clean/lstm_fd003_per_engine_coverage.json
5d8b31f2ef54d28d1d74d41e052025a2  results/clean/lstm_fd003_split_cp_per_seed.json
74b925ae2afbf35ca912b7d73fcf7e41  results/clean/lstm_mcdropout_per_seed.json
f87936901feedae5f06087819eff5552  results/clean/lstm_mse_fixed_training_residual_per_seed.json
611e8b313757bbe306b34b2e277098e9  results/clean/lstm_per_engine_coverage.json
1c7dba3751e73446fdb262509e39adfb  results/clean/lstm_split_cp_per_seed.json
27b16c8162ada8aa81aa3883c3488579  results/clean/table1_2x2_per_seed.json
6eb06548c2d6565521e47dbb39cac180  results/clean/table1_2x2_summary.json
b08c164f298672ea9a69568ea6105211  results/clean/table2_clean_full.json
56b674100806aaff1620bc55c9115bea  results/clean/transformer_clean_per_seed.json
5899e342f25c2acff54086dae6488134  results/clean/transformer_cost_table_fd001.csv
68fcbcf0f0314cd6f5d87dea64bc3134  results/clean/transformer_cost_table_fd002.csv
714950ae40fd00faaad78f250b2c0159  results/clean/transformer_cost_table_fd003.csv
8f9a001c041b6a751b83e3f16852f164  results/clean/transformer_cost_table_fd004.csv
c9833228d16e6c87d364b214e8d62e06  results/clean/transformer_deep_ensemble.json  [latency-exempt]
e27dc0f858719f0d694e50350c93108a  results/clean/transformer_mcdropout_mse_per_seed.json
5846814338ca186be28b656cfa0e5d7d  results/clean/transformer_per_engine_coverage.json
9fb6a2c0b4fbd6e6e562aa7e860c243b  results/clean/transformer_split_cp_per_seed.json
e3cc9d83cd627169595b9a703b38b2ee  results/decision/maintenance_decision_one_sided.json
549e3ef99ccb094a9ecd9f13a19ec567  results/decision/maintenance_decision_rul_corrected_v2.json
05c0ecb17fea6dd4397d06d5d57ce9b3  results/decision/maintenance_decision_rul_correction_exact_match_check.json
35aa1c513e725a7f203d0544657087c5  results/decision/maintenance_decision_two_sided.json
83a322e32e355a4689258914962101f4  results/degradation/degradation_dose_response_overlap_summary.json
6728d7d0451ffbb2d4c014145355e70f  results/degradation/degradation_half_life_bias_gain_drift.json
81d047bb99213bbf4e08f47f30982fc2  results/degradation/degradation_sweep_bias_gain_drift.json
35ffe123bcc6ebdf266c1c2d4f6ae2c0  results/degradation/dose_response_feat_oob.json
83e349a8b40e9191f7215022e43ec495  results/degradation/drift_controls.json
c9ea4380c505903a89af3d5dd181053e  results/degradation/drift_fixed_endpoint_shuffle_control.json
7b450011bbc24dc75b50f14373892af5  results/degradation/lstm_armc_mu_std.json
5de422b97196a4995fbdf0aef60a3117  results/degradation/lstm_clamp_frac_armc.json
1e977be6c2acbe17bc1e5a95153f4ecf  results/degradation/lstm_clamp_frac_main_arm.json
97a942d265d17661475c053449c226de  results/degradation/lstm_ensemble_scale_sweep.json
9531318a22a107e8ef426ade057ee6c0  results/degradation/lstm_ensemble_n5_vs_n15_variance.json
81d9ae0d534cc5b694922e67479d81ad  results/degradation/lstm_fd003_armc_mu_std.json
f1e42943f29c772cbdd11214813fdbdf  results/degradation/lstm_fd003_ensemble_scale_sweep.json
a85aa0e48249309d9da7aba457c6c29b  results/degradation/lstm_fd003_frozen_sigma_decomposition.json
e34e00bc0f67ed8eb0d521aa0fb689ff  results/degradation/lstm_fd003_noise_armc.json
92c7e08166a8fcdee2a44efe5fb6b485  results/degradation/lstm_fd003_noise_main_arm.json
ec77aaa44992fa170afcd9f672b95617  results/degradation/lstm_fd003_sweep.json
7faafeb76f732d5950438d73fe5a9a28  results/degradation/lstm_frozen_sigma_decomposition.json
fa7f36098da3a1ecea3085b1e93c8333  results/degradation/lstm_frozen_sigma_decomposition_relative.json
5f47db47a944be2f9111c20962c4b6bf  results/degradation/lstm_noise_armc.json
cbac5a3288549d4a4a905fa20135a85a  results/degradation/lstm_noise_main_arm.json
1602c36aa3dc8e05ee2168712bad356a  results/degradation/lstm_relative_half_life.json
84d10c9d95630f2f5ec8d0e60a29ba6a  results/degradation/noise_backbone_comparison_summary.json
9e3264825e1c0c577df8322ef1db50ad  results/degradation/threshold_crossover_leftside_extension.json
0431d493457118707dc62a75556dae68  results/degradation/threshold_crossover_refinement.json
0913a145af62ab6e4dc898b167b382b4  results/degradation/transformer_clamp_frac.json
f2611802e0bfec4dd7bb00e879f0e58a  results/degradation/transformer_frozen_sigma_decomposition.json
04a784a8aef49646d8883fa4478f7065  results/degradation/transformer_half_life.json
d0576fa10fb3d5de025f062964ee2083  results/degradation/transformer_noise_armc.json
8b1996e28c2660c2f723e022398b3783  results/degradation/transformer_noise_main_arm.json
b31403907e599ddf0e9683a362c6aba7  results/diagnostics/A1_perturbation_target_diagnostic.json
e6fc8d99b88bdefd0dc7ee99624ca1f8  results/seeds/ensemble_independent_replication.json
fe82a360216851aa1a1d67faec927a45  results/seeds/ensemble_size_sweep_fd004_interval_score.json
7ca3fafbed144632be9451d650f0c6a5  results/seeds/samesplit_ensemble_control.json
```

`[latency-exempt]` marks the 3 files whose only cross-run difference is
`latency_ms_per_sample`, per the scope note above.

## Numerical self-consistency of this manifest

- 71 files bit-identical across two independent reruns (2026-09-22) -- the
  MD5s above. 70 are `run_pipeline.sh` outputs; 1
  (`results/diagnostics/A1_perturbation_target_diagnostic.json`) is the
  standalone diagnostic described above.
- 3 files identical except for `latency_ms_per_sample` (wall-clock timing,
  explicitly out of scope) -- the MD5s above are from the second of the
  two runs; a fresh run's `latency_ms_per_sample` value will differ from
  it but every other field will match.
- 74 = 71 + 3, matching the file count claimed at the top of this
  document.
- 1 additional file (`results/diagnostics/A2_target_alignment_diagnostic.json`)
  exists alongside these 74 but is excluded from the reproducibility claim
  for the documented reason given above (single-run diagnostic, not
  independently re-verified).
