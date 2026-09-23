"""
Reconstructs the generation script for step0c_leakfree_ece_agg.json --
an audit found no script in the codebase that produces this file (a
`grep` across all *.py found no match), yet main.tex's Table~tab:leak
footnote cites it ("ECE here is computed on the pooled predictions of the
five seeds, ... 0.014 against 0.032 on FD001"). This reconstruction proves
the number is reproducible, not fabricated.

Method: uses the exact same source as train_lstm.py's single official
test-set evaluation -- computes each seed's (mu, sigma, y_test) from
checkpoints_leakfree/{ds}_LSTM_seed{seed}.pt (the checkpoint that
pipeline itself trained) plus that seed's own fit_units-derived scaler
(the same load_and_process_leakfree call), but instead of computing each
seed's ECE separately as the original script does, concatenates the 5
seeds' (mu, sigma, y_test) along the sample dimension (pooling) and
computes ECE once -- this is exactly what the footnote's "pooled
predictions" means, a different aggregation order from Table~tab:clean's
"average of each seed's own ECE".
"""
import os
import json

import numpy as np
import torch

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

DATASETS = ['FD001', 'FD002', 'FD004']
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    result = {}
    for ds in DATASETS:
        mu_pool, sigma_pool, y_pool = [], [], []
        for seed in C.SEEDS:
            fit_units = CANON[ds][str(seed)]['fit_units']
            train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
            X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            ckpt_path = os.path.join(CKPT_DIR, f"{ds}_LSTM_seed{seed}.pt")
            model = C.load_checkpoint_model(ckpt_path, device)
            with torch.no_grad():
                mu_out, log_sigma_out = model(X_test_t)
            mu_np = np.clip(mu_out.float().cpu().numpy().flatten() * 125.0, 0, 125)
            sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0

            mu_pool.append(mu_np); sigma_pool.append(sigma_np); y_pool.append(y_test)

        mu_all = np.concatenate(mu_pool); sigma_all = np.concatenate(sigma_pool); y_all = np.concatenate(y_pool)
        ece_pooled = C.compute_ece(mu_all, sigma_all, y_all, CONF_LEVELS)
        result[ds] = ece_pooled
        print(f"  {ds}: pooled ECE = {ece_pooled:.6f}  (n_pooled_samples={len(y_all)})")

    out_path = os.path.join(RESULTS_DIR, 'step0c_leakfree_ece_agg.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_ece_agg.json')) as f:
        existing = json.load(f)
    print("\n=== comparison against existing step0c_leakfree_ece_agg.json ===")
    all_match = True
    for ds in DATASETS:
        diff = abs(result[ds] - existing[ds])
        ok = diff < 1e-6
        all_match &= ok
        print(f"  {ds}: existing={existing[ds]:.8f}  recomputed={result[ds]:.8f}  diff={diff:.2e}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nALL MATCH (tol 1e-6): {all_match}")
