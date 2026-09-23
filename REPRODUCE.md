# Reproducing the result files

This document has two levels, matching the paper's Data Availability
section.

## Level 1: reader reproducibility (tables from the released result files)

Anyone who clones this repository can regenerate every table and number
quoted in the paper and its supplement, from the result files already
released here, without a GPU, PyTorch, or the raw C-MAPSS data:

```bash
git clone https://github.com/jeff-1980/rul-interval-reliability
cd rul-interval-reliability
python paper/make_tables.py .
```

This (re)writes every file under `paper/tables/`, including
`numbers.tex`'s `\NReproBit` macro, purely from `results/` and
`REPRODUCE.md`'s own MD5 manifest below (nothing is hardcoded). Diffing
the output against what is already committed at this tag should show no
changes. This is the level the paper's Data Availability section claims
is reproducible by any reader.

## Level 2: author verification (result files from the released checkpoints)

This level covers reproduction of the **66 result files** listed below,
all of them `run_pipeline.sh` outputs (including
`results/diagnostics/A1_perturbation_target_diagnostic.json`, folded into
the pipeline because it is now cited in the main text). The current
protocol, relative to the predecessor pipeline, is inference-level except
for one deliberate retraining (noted below):

- **Sensor-only perturbation on FD002/FD004.** These two datasets' feature
  columns are 3 operating-condition settings + 15 sensor channels. Every
  degradation experiment (Gaussian noise arms A/B/C, bias, gain, drift and
  its controls, threshold sweeps, attribution, the decision stress test)
  perturbs only the 15 sensor channels, leaving the 3 setting columns
  untouched -- settings are the commanded operating regime, not sensor
  measurements subject to degradation. FD001/FD003 have no setting columns
  and are unaffected by this change. `f_oob` is likewise computed only over
  the perturbed sensor channels -- via a single shared implementation,
  `noise_injection.feat_oob(X_scaled, sensor_mask)`, that every sweep,
  control, attribution and threshold script now calls (previously several
  of them independently reimplemented the same reduction with a plain
  18-column denominator; see "What changed in this experiment" below).
  `results/diagnostics/A1_perturbation_target_diagnostic.json` is the
  pre-registered check that confirmed the sensor-only restriction was safe
  to make; it also retains the joint (18-column) perturbation's numbers for
  comparison, so no separate joint-perturbation control file is shipped.
- **Test-truth convention.** Every table (clean, degradation, attribution,
  decision, the Table I/II families) scores against
  `min(official_RUL - 1, 125)`, matching the training/calibration label
  convention's counting origin (the last row of a run-to-failure training
  trajectory has RUL = 0), not the official RUL value directly.
- **Table I's T/W and V/W cells are now controlled (T/W', V/W').** See
  "What changed in this experiment" below -- this is the one exception to
  "inference-level": these two cells were retrained (not reproducibility-
  asserted; see the Scope section).

This list supersedes the predecessor 32-file list entirely -- every file
in that list is either included below (with updated content and checksum)
or was superseded by a file that is.

### What changed in this experiment

1. **Table I's T/W and V/W cells retrained as T/W', V/W'.** The original
   T/W cell fit its checkpoint on 100% of the training-file engines, and
   the original V/W cell on 80%, while V/F and T/F both use
   `canonical_splits.json`'s 60% `fit_units`. This confounded "amount of
   fit data" with the two variables Table I's Sel./Norm./Int. contrasts
   are supposed to isolate (selection criterion and scaler scope).
   `protocol_2x2_controlled_retrain.py` retrains both cells with the same
   60% `fit_units` as V/F/T/F, changing only the selection criterion (test
   set for T/W', `val_units` for V/W') and scaler scope (whole-file, W).
   The resulting 30 checkpoints (3 datasets x 5 seeds x 2 cells) are
   shipped under `results/checkpoints/lstm_2x2_controlled/`. The original,
   uncontrolled T/W and V/W checkpoints are kept for provenance under
   `results/superseded/checkpoints_tw_vw_uncontrolled/` and are no longer
   read by `build_table1.py`. See `code/superseded/README.md`.
2. **Gain injection was never actually sensor-only.** `inject_gain_fixed_pct_raw`
   multiplies each channel by `(1 +/- k)`, a factor that does not depend on
   `full_scale_range` -- so the `sensor_only_scale()` fix
   (which zeroes `full_scale_range` on the 3 setting columns) had no effect
   on it. Gain injection kept perturbing FD002/FD004's setting columns even
   after that fix. `inject_gain_fixed_pct_raw` now takes an explicit
   `sensor_mask` parameter; its one call site
   (`run_sweep_degradation_transformer.py`) now passes one. Verified by
   `code/test_masks.py::test_gain_mask` (setting columns byte-for-byte
   unchanged; sensor columns scaled by exactly `1 +/- 0.05`).
3. **`f_oob` denominator unified and corrected.** Multiple scripts computed
   `f_oob` as the fraction of out-of-[-1,1] entries over all 18 feature
   columns (settings included), even though settings are never perturbed
   under the sensor-only protocol -- diluting the ratio. All of them now
   call the single `noise_injection.feat_oob(X_scaled, sensor_mask)`
   (denominator restricted to sensor columns). Two scripts
   (`eval_lstm_armc_mu_std.py`, `train_lstm_fd003_nll_and_mechanism.py`)
   also had a second, independent bug from the same era: they read
   `f_oob` from only the first of 5 seeds' un-windowed trajectories instead
   of averaging all 5 seeds' windowed test inputs; fixed at the same time.
   Verified by `code/test_masks.py::test_feat_oob_fd002_drift5pct`: FD002
   drift-5% `f_oob` is now 11.43% (sensor denominator, matches
   `diagnostic_perturbation_target_a1.py`'s own number), not the previous
   9.53% (18-column denominator).
4. **A2's `y_official` was silently identical to `y_alt`.**
   `diagnostic_target_alignment_a2.py` built `y_official` from
   `extract_raw_windows(mode='test')`'s return value -- but that function
   itself already applies the `min(official_RUL-1,125)` convention (a
   prior-round fix), so `y_official` and `y_alt` were computing the same
   quantity twice, silently collapsing the whole diagnostic's contrast to
   nothing. Both are now built independently, directly from the raw
   official RUL label, with an explicit
   `assert not np.array_equal(y_official, y_alt)` guarding against a
   regression.

### Scope: what is and isn't guaranteed

**Reproducible (verified bit-identical across two independent reruns,
2026-09-23)**: 63 of the 66 files below, for every mechanism except
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

**Not independently re-verified for this experiment**:
`results/diagnostics/A2_target_alignment_diagnostic.json` (a pre-registered
diagnostic run once, to motivate the RUL-1 convention change -- see
`results/diagnostics/README.md`). It was produced by the same inference
code as the verified files above, just not run twice for this experiment,
and is not one of the 66.

**Not covered by this document**: model training (checkpoint weights
depend on training-time CUDA kernel selection and are not asserted
reproducible here). This now includes:
- `results/checkpoints/lstm_2x2_controlled/` (T/W', V/W', trained once by
  `protocol_2x2_controlled_retrain.py` for this experiment);
- the original, uncontrolled T/W and V/W checkpoints under
  `results/superseded/checkpoints_tw_vw_uncontrolled/` and their training
  scripts under `code/superseded/`;
- the 10 extra-seed checkpoints under
  `results/checkpoints/lstm/{ds}_LSTM_extraseed{seed}.pt`, needed by
  `eval_lstm_ensemble_n5_vs_n15.py`;

any file not listed below, and anything under `results/superseded/` or
`code/superseded/`.

### Two reproducibility defects fixed during development (found via the double-run diagnosis)

The first attempt at the two-independent-reruns check in that round
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

### Environment used to produce the checksums below

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

### How to reproduce

```bash
cd code
export PYTHONHASHSEED=0
export RUL_DATA_DIR=/path/to/cmapss   # or place the data at ../data/
bash run_pipeline.sh myrun
```

This runs, in order, all 66 files below (see `run_pipeline.sh` for the
exact list and grouping) using the checkpoints already provided under
`results/checkpoints/` -- including the T/W', V/W' controlled-2x2
checkpoints under `results/checkpoints/lstm_2x2_controlled/`, needed by
`build_table1.py` (Table I; the original, uncontrolled T/W/V/W checkpoints
under `results/superseded/checkpoints_tw_vw_uncontrolled/` are no longer
read by it), and the 10 extra-seed checkpoints under
`results/checkpoints/lstm/{ds}_LSTM_extraseed{seed}.pt`, needed by
`eval_lstm_ensemble_n5_vs_n15.py` (Appendix D). Every script writes into
`results/generated/`; the final step, `publish_results.py`, copies each
output to its released name and location under `results/{category}/` (see
`results/table_provenance.csv` for the mapping).

To verify reproducibility yourself: run the pipeline twice (snapshotting
the 66 released files between runs), then compare MD5s -- this is exactly
what `code/check_md5.py` does (edit its snapshot directory to point at
your two runs' outputs).

#### External dependencies not shipped with this release

`run_pipeline.sh`'s first step, `eval_nll_clean_b1.py`, reads three
pre-existing files under `results/generated/` that this release does not
ship and that no script in `run_pipeline.sh` regenerates:

- `results/generated/step0c_leakfree_results.json` -- LSTM (FD001/FD002/
  FD004) clean-condition results, produced as a byproduct of `train_lstm.py`.
- `results/generated/stepFD003_nll_leakfree_results.json` -- the FD003
  counterpart, produced as a byproduct of
  `train_lstm_fd003_nll_and_mechanism.py`.
- `results/generated/leakfree_t2/t2_transformer_nll_leakfree_results.json`
  -- the Transformer counterpart (4 datasets), produced as a byproduct of
  `train_transformer_nll.py`.

Each is a per-seed record skeleton carrying training-time-only fields
(`n_fit_units`, `n_val_units`, `n_val_windows`, `best_val_rmse`,
`elapsed_train_s`) that `eval_nll_clean_b1.py` keeps as-is while
refreshing only the fields that depend on the test-truth convention
(rmse/score/picp/mpiw/ece). Those training-time fields cannot be
recomputed by inference alone -- recovering them means rerunning the
three training scripts named above, which is itself outside this
document's reproducibility claim (see Scope, above). A from-scratch run
of `run_pipeline.sh` on a machine that has never run those training
scripts will fail at this first step until the three files are supplied
by some other means. This is the concrete, structural reason the paper's
Data Availability section states that the pipeline "reads a small number
of intermediate files from earlier development runs that are not part of
the release" and does not claim it "runs unchanged on another machine" --
it is not merely an unverified generalisation.

A full audit of every other file read across `run_pipeline.sh`'s ~39
scripts found no further external dependency: every other read resolves
to either `results/canonical_splits.json` / `results/checkpoints/` (both
released and git-tracked) or a file an earlier step of the same pipeline
run already wrote.

### The 66 files and their MD5 (this environment, this codebase revision)

```
e44c3ba1cb450a033c38d5265fa7b6da  results/attribution/attribution_bootstrap_by_order_exact_interp.json
285377404751091a7eaa03e0436d2ce0  results/attribution/attribution_lstm_exact_interp.json
207d671fef37532585010d9981e64a38  results/attribution/attribution_raw_per_seed_nearest_grid.json
499c92a086bb71a61ff138d5d6fa437a  results/attribution/attribution_transformer_exact_interp.json
19390c4db9c162c526af3390a22c2ce9  results/attribution/frozen_output_decomposition_2x2_nearest_grid.json
4ebcead6a9efe1dd9be2e802d50363f5  results/clean/fair_calibration_main_table.json
74af1cb26e7d399f63ab2abea97cb4e7  results/clean/fair_calibration_sigma_fixed.json
4e94153e156f3e364b40d7e658d556f4  results/clean/interval_score_wis_clean.json
b5cef65b1ca68222953f281b66772e2b  results/clean/lstm_clean_ece_pooled.json
257121a8c7298b25f8fa095146b68b10  results/clean/lstm_clean_per_seed.json
d87441bb342adf92ceb576666310edfb  results/clean/lstm_deep_ensemble.json  [latency-exempt]
f626e302ed56853b26157d0f4fd1950c  results/clean/lstm_fd003_clean_per_seed.json
c46d7879923ef1c75dfaf0a72080b054  results/clean/lstm_fd003_deep_ensemble.json  [latency-exempt]
49feffe11c5ce4448b486b7177520803  results/clean/lstm_fd003_mcdropout_mse_per_seed.json
b767bdd8397c053ad09baf08d6638755  results/clean/lstm_fd003_per_engine_coverage.json
5d8b31f2ef54d28d1d74d41e052025a2  results/clean/lstm_fd003_split_cp_per_seed.json
74b925ae2afbf35ca912b7d73fcf7e41  results/clean/lstm_mcdropout_per_seed.json
f87936901feedae5f06087819eff5552  results/clean/lstm_mse_fixed_training_residual_per_seed.json
611e8b313757bbe306b34b2e277098e9  results/clean/lstm_per_engine_coverage.json
1c7dba3751e73446fdb262509e39adfb  results/clean/lstm_split_cp_per_seed.json
883340fef506da8db35276c188f1ea67  results/clean/table1_2x2_per_seed.json
c2edca9074181ec23bbe7804a1bc2612  results/clean/table1_2x2_summary.json
b08c164f298672ea9a69568ea6105211  results/clean/table2_clean_full.json
56b674100806aaff1620bc55c9115bea  results/clean/transformer_clean_per_seed.json
1e4e526a34b96d57900f4f0bb7bcd666  results/clean/transformer_deep_ensemble.json  [latency-exempt]
e27dc0f858719f0d694e50350c93108a  results/clean/transformer_mcdropout_mse_per_seed.json
5846814338ca186be28b656cfa0e5d7d  results/clean/transformer_per_engine_coverage.json
9fb6a2c0b4fbd6e6e562aa7e860c243b  results/clean/transformer_split_cp_per_seed.json
e3cc9d83cd627169595b9a703b38b2ee  results/decision/maintenance_decision_one_sided.json
549e3ef99ccb094a9ecd9f13a19ec567  results/decision/maintenance_decision_rul_corrected_v2.json
05c0ecb17fea6dd4397d06d5d57ce9b3  results/decision/maintenance_decision_rul_correction_exact_match_check.json
35aa1c513e725a7f203d0544657087c5  results/decision/maintenance_decision_two_sided.json
a081fcae9f2709e67b00e672062a080c  results/degradation/degradation_dose_response_overlap_summary.json
426f7706cfcad6ace59b449cf0192091  results/degradation/degradation_half_life_bias_gain_drift.json
d345b7412def8a643e8bbeb9d69101d3  results/degradation/degradation_sweep_bias_gain_drift.json
0e772661d1c47dd96155e0ac9a7c4dd3  results/degradation/dose_response_feat_oob.json
c64bccd357540e4ec2b3c46d3bd2b1c5  results/degradation/drift_controls.json
c9ea4380c505903a89af3d5dd181053e  results/degradation/drift_fixed_endpoint_shuffle_control.json
b2f990b16254800a9fe1f8c4bf3ff5b6  results/degradation/lstm_armc_mu_std.json
d2f22b1d83cfa8122412dc6327c158ec  results/degradation/lstm_clamp_frac_armc.json
0c6bfd4650dd8e61bf911a3074dc5ba0  results/degradation/lstm_clamp_frac_main_arm.json
97a942d265d17661475c053449c226de  results/degradation/lstm_ensemble_scale_sweep.json
9531318a22a107e8ef426ade057ee6c0  results/degradation/lstm_ensemble_n5_vs_n15_variance.json
81d9ae0d534cc5b694922e67479d81ad  results/degradation/lstm_fd003_armc_mu_std.json
f1e42943f29c772cbdd11214813fdbdf  results/degradation/lstm_fd003_ensemble_scale_sweep.json
14c1a3bbe71b7a450064ffc87b1fecbc  results/degradation/lstm_fd003_frozen_sigma_decomposition.json
e34e00bc0f67ed8eb0d521aa0fb689ff  results/degradation/lstm_fd003_noise_armc.json
92c7e08166a8fcdee2a44efe5fb6b485  results/degradation/lstm_fd003_noise_main_arm.json
d2069e6b4df3c17259d4790f7008110a  results/degradation/lstm_fd003_sweep.json
d765e23665b50ab9312463ff230aac7e  results/degradation/lstm_frozen_sigma_decomposition.json
83fe7e9e8a9366fc11be8ae262bef5ba  results/degradation/lstm_frozen_sigma_decomposition_relative.json
436073c9954b5ef18c45ef4e4a537d21  results/degradation/lstm_noise_armc.json
282a24a157a50ac962a6b3eab046f6bc  results/degradation/lstm_noise_main_arm.json
bf5d1f00837944348532d7a984e5e638  results/degradation/lstm_relative_half_life.json
21c588f36dd3246d9839e78c2bcab19d  results/degradation/noise_backbone_comparison_summary.json
9e3264825e1c0c577df8322ef1db50ad  results/degradation/threshold_crossover_leftside_extension.json
0431d493457118707dc62a75556dae68  results/degradation/threshold_crossover_refinement.json
0913a145af62ab6e4dc898b167b382b4  results/degradation/transformer_clamp_frac.json
0564c72ab1c1b509dc5f379b0d1168d7  results/degradation/transformer_frozen_sigma_decomposition.json
b0cf662a8efe538427137687b7e6099b  results/degradation/transformer_half_life.json
85646659825ea5e07d5eaa2fb3b1fc93  results/degradation/transformer_noise_armc.json
2f992093f0c088b651e8f485749c177a  results/degradation/transformer_noise_main_arm.json
b31403907e599ddf0e9683a362c6aba7  results/diagnostics/A1_perturbation_target_diagnostic.json
e6fc8d99b88bdefd0dc7ee99624ca1f8  results/seeds/ensemble_independent_replication.json
fe82a360216851aa1a1d67faec927a45  results/seeds/ensemble_size_sweep_fd004_interval_score.json
7ca3fafbed144632be9451d650f0c6a5  results/seeds/samesplit_ensemble_control.json
```

`[latency-exempt]` marks the 3 files whose only cross-run difference is
`latency_ms_per_sample`, per the scope note above.

### Numerical self-consistency of this manifest

- 63 files bit-identical across two independent reruns (2026-09-23) -- the
  MD5s above, all 66 of them `run_pipeline.sh` outputs.
- 3 files identical except for `latency_ms_per_sample` (wall-clock timing,
  explicitly out of scope) -- the MD5s above are this release's actual
  published values; a fresh run's `latency_ms_per_sample` value will
  differ from them but every other field will match.
- 66 = 63 + 3, matching the file count claimed at the top of this
  section.
- 1 additional file (`results/diagnostics/A2_target_alignment_diagnostic.json`)
  exists alongside these 66 but is excluded from the reproducibility claim
  for the documented reason given above (single-run diagnostic, not
  independently re-verified).
