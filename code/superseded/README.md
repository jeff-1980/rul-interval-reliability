# Superseded training scripts

These two scripts are kept for provenance only. They are the original
(pre-audit) training pipelines behind two of the four cells in Table I
(the 2x2 leakage-audit protocol experiment):

- `train_lstm_leaked_test_select_whole_file_scaler.py` -- selects the
  checkpoint by *test-set* RMSE each epoch, and fits the feature scaler on
  the whole training file (both leakage-affected). This is the "T/W" cell.
- `train_lstm_val_select_whole_file_scaler.py` -- selects the checkpoint by
  held-out validation RMSE (not test), but still fits the scaler on the
  whole training file. This is the "V/W" cell.

**Re-running either script will not reproduce the checkpoints shipped in
this release.** Training involves non-deterministic GPU kernel selection
(see `README.md`'s scope-of-reproducibility note); running either script
again will train a *different* set of weights, not recover the exact ones
used to compute Table I.

**The actual T/W and V/W checkpoint weights ARE shipped in this release**,
under:

- `results/checkpoints/lstm_leaked_test_select_whole_file_scaler/` (T/W, 15
  files: 3 datasets x 5 seeds)
- `results/checkpoints/lstm_leaked_val_select_whole_file_scaler/` (V/W, 15
  files: 3 datasets x 5 seeds)

These are deliberately kept in directories separate from
`results/checkpoints/lstm/`, which holds the current, leakage-audited V/F
protocol's checkpoints -- the two must never be confused, since T/W and V/W
are the leakage-affected baselines Table I contrasts against V/F, not
alternative copies of the same weights. `code/build_table1.py` reads all
four quadrants (T/W, V/W, V/F, T/F) from their respective directories and
recomputes Table I's numbers by inference only, from these existing
checkpoints -- no retraining.

Both scripts import `common.py`, which lives one directory up in `code/`,
not here -- they will not run from within `code/superseded/` without that
import resolving (e.g. by running them from `code/` directly, or adding
`code/` to `PYTHONPATH`). They are included for readability and provenance,
not as a tested/runnable part of the reproduction pipeline.
