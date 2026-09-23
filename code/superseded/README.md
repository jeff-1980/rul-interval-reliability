# Superseded scripts

## Training scripts

These two scripts are kept for provenance only. They are the original
(pre-audit) training pipelines behind the historical, uncontrolled T/W and
V/W checkpoints:

- `train_lstm_leaked_test_select_whole_file_scaler.py` -- selects the
  checkpoint by *test-set* RMSE each epoch, and fits the feature scaler on
  the whole training file (both leakage-affected), training on **all**
  training-file engines (not the 60% canonical fit split used everywhere
  else in this release). This was the original "T/W" cell.
- `train_lstm_val_select_whole_file_scaler.py` -- selects the checkpoint by
  held-out validation RMSE (not test), but still fits the scaler on the
  whole training file, training on an **80%** split (not the 60% canonical
  fit split). This was the original "V/W" cell.

**Note: these two cells are no longer part of Table I.** Both scripts
trained on a different *amount* of fit data than the V/F and T/F cells
(100% and 80% respectively, vs. V/F/T/F's 60% canonical `fit_units`),
confounding "scaler scope" and "selection criterion" -- the two variables
Table I's Sel./Norm./Int. contrasts are supposed to isolate -- with a third,
undocumented variable (fit-data volume). Table I now uses **T/W'** and
**V/W'**, retrained by `protocol_2x2_controlled_retrain.py` with the same
60% canonical `fit_units` as every other cell, changing only the scaler
scope (whole-file) and selection criterion (test set for T/W', `val_units`
for V/W'). See that script's docstring and `results/table_provenance.csv`
for the full protocol.

The checkpoints these two superseded scripts produced are kept for
provenance under `results/superseded/checkpoints_tw_vw_uncontrolled/`
(re-running either script will not reproduce them bit-for-bit -- training
involves non-deterministic GPU kernel selection, see `README.md`'s
scope-of-reproducibility note). They are **not** read by
`code/build_table1.py` any more.

Both scripts import `common.py`, which lives one directory up in `code/`,
not here -- they will not run from within `code/superseded/` without that
import resolving (e.g. by running them from `code/` directly, or adding
`code/` to `PYTHONPATH`). They are included for readability and provenance,
not as a tested/runnable part of the reproduction pipeline.

## Cost tables

`cost_table_lstm.py`, `cost_table_lstm_fd003.py`, and
`cost_table_transformer.py` assembled a per-method cost table (PICP/MPIW/
ECE/per-engine/coverage-half-life plus a carried-over latency column and a
recomputed parameter count) for each backbone/sub-dataset. No table or
figure in the manuscript or supplementary material reads their output, so
as of this release they are no longer run by `run_pipeline.sh`, no longer
published under `results/clean/`, and kept here for provenance only. Like
the training scripts above, they import sibling modules from `code/` and
are not runnable as-is from this directory.
