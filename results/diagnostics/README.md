# Diagnostics (R8 pre-flight checks)

These two files are the pre-registered diagnostic checks run before the R8
full rerun (sensor-only perturbation on FD002/FD004, RUL-1 test-truth
convention). Neither feeds any paper table directly; A1 is now cited in
the main text's discussion of operating-condition settings, and A2
motivates the RUL-1 convention change.

- `A1_perturbation_target_diagnostic.json` -- three-way contrast on
  FD002/FD004 (both backbones, heteroscedastic NLL, 5 seeds, drift 5% and
  Gaussian scheme C 1%) of perturbing (i) sensors only, (ii) operating-
  condition settings only, (iii) both jointly (the old protocol). Keyed by
  `{dataset: {backbone: {sensors|settings|joint: {drift5pct|gauss1pct_armC: {...}}}}}`,
  each leaf reporting `picp_grand_mean`, `f_oob_grand_mean` (restricted to
  the perturbed channels), `premature_rate_L20_grand_mean`, and
  `unrecognised_rate_L20_grand_mean`. The pre-registered stop condition
  was: if sensors-only's drift premature-trigger rate were less than half
  of joint's, halt and do not proceed to the full rerun. It was not
  triggered in any of the 4 (dataset, backbone) cells:
  `sensors.drift5pct.premature_rate_L20_grand_mean` vs.
  `joint.drift5pct.premature_rate_L20_grand_mean` --
  FD002/LSTM: 0.6933 vs. 0.6957 (ratio 0.997);
  FD002/Transformer: 0.6917 vs. 0.6972 (ratio 0.992);
  FD004/LSTM: 0.5899 vs. 0.6802 (ratio 0.867);
  FD004/Transformer: 0.5249 vs. 0.6700 (ratio 0.783) --
  all well above the 0.5 stop threshold. **Verified bit-identical across
  two independent reruns** (see REPRODUCE.md).
- `A2_target_alignment_diagnostic.json` -- all four datasets, both
  backbones, NLL, 5 seeds, clean and drift 5%, comparing RMSE/PICP/interval
  score/premature-trigger-rate under the official-RUL test-truth convention
  (pre-R8) against min(official RUL - 1, 125) (the convention R8 adopts,
  matching the training/calibration label convention). Both are computed
  from a single inference pass per (dataset, backbone, seed, condition) --
  model outputs do not depend on the test-truth convention, only the
  downstream metrics do. Still single-run, not double-run-verified.
