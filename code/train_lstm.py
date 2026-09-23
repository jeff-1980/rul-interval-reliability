"""
STEP 0c: builds on STEP0b (which only fixed the checkpoint-selection
leakage) by also fixing the scaler leakage -- STEP0b still calls
`C.load_and_process(ds)`, which fits MinMaxScaler on all train-side
engines at once (including val), so val's feature distribution leaks into
the scaler's min/max parameters.

This script uses `C.load_and_process_leakfree(ds, fit_units)`: the
scaler is fit only on fit_units (60%), and val/calib/test are all
transform-only. fit_units/val_units are read from
`canonical_splits.json` (the project-wide unified split) rather than
split locally here -- using exactly the same (fit_units, val_units) as
STEP1b/STEP3b, so the train/val split matches across all three training
lines, engine-by-engine.

Checkpoint-selection criterion, training hyperparameters, and
architecture are identical to STEP0/STEP0b; only the data source changes
(fit=60% instead of 80%, to leave 20% for calib -- even though STEP0
itself doesn't use calib, its gradient still must not touch
calib_units, otherwise STEP3's calibration precondition that "calib was
never seen by any model's gradient" would not hold for what STEP0
trained. STEP0 and STEP3 are different models, but keeping the fit set
identical across all three training lines -- "the split must match
engine-by-engine across all three lines" -- was an explicit
requirement).

Checkpoints are saved to a new directory `checkpoints_leakfree/`, leaving
`checkpoints/` (original, double leakage in both selection and scaler)
and `checkpoints_valselect/` (only the selection leakage fixed, scaler
still leaky) as historical versions -- not deleted, just no longer used.
"""
import os
import json
import time
import random
import copy
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = C.SEEDS
DATASETS = C.DATASETS
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
HIDDEN_DIM = C.HIDDEN_DIM
LOG_SIGMA_MIN = C.LOG_SIGMA_MIN
LOG_SIGMA_MAX = C.LOG_SIGMA_MAX
Z_SCORE = C.Z_SCORE
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()


def run_one_seed(ds_name, seed, device, use_amp):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[ds_name][str(seed)]
    fit_units, val_units = split['fit_units'], split['val_units']

    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = C.HeteroscedasticLSTM(input_dim, HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=LOG_SIGMA_MIN, log_sigma_max=LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                mu, log_sigma = model(bx)
                loss = gaussian_nll_loss(mu, log_sigma, by)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mu_val, _ = model(X_val_t)
            mu_val = np.clip(mu_val.float().cpu().numpy().flatten() * 125.0, 0, 125)
            curr_val_rmse = np.sqrt(mean_squared_error(y_val_cycles, mu_val))
        if curr_val_rmse < best_val_rmse:
            best_val_rmse = curr_val_rmse
            best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0

    model.load_state_dict(best_state); model.eval()

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, CONF_LEVELS)

    print(f"   seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)} n_val_windows={len(y_val)}  "
          f"train={elapsed:.1f}s  best_val_rmse={best_val_rmse:.3f}")
    print(f"      [official test, evaluated once] RMSE={rmse:.3f} Score={score:.1f} "
          f"PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_seed{seed}.pt")
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': LOG_SIGMA_MIN, 'log_sigma_max': LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name,
        'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
        'selection_protocol': 'canonical_split fit/val, leakfree scaler',
    }, ckpt_path)

    return {'seed': seed, 'n_fit_units': len(fit_units), 'n_val_units': len(val_units),
            'n_val_windows': int(len(y_val)), 'best_val_rmse': best_val_rmse, 'elapsed_train_s': elapsed,
            'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma_np.mean())}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  seeds={SEEDS}  epochs={EPOCHS}  CKPT_DIR={CKPT_DIR}")

    all_out = {}
    t_start_all = time.time()
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        seed_results = [run_one_seed(ds, seed, device, use_amp) for seed in SEEDS]
        all_out[ds] = seed_results
    total_elapsed = time.time() - t_start_all

    out_path = os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    rows = []
    for ds in DATASETS:
        rs = all_out[ds]
        rmse_arr = np.array([r['rmse'] for r in rs]); score_arr = np.array([r['score'] for r in rs])
        picp_arr = np.array([r['picp'] for r in rs]); mpiw_arr = np.array([r['mpiw'] for r in rs])
        ece_arr = np.array([r['ece'] for r in rs])
        rows.append({'Dataset': ds, 'Model': 'LSTM',
                      'RMSE (Mean)': rmse_arr.mean(), 'RMSE (Std)': rmse_arr.std(ddof=1),
                      'Score (Mean)': score_arr.mean(), 'Score (Std)': score_arr.std(ddof=1),
                      'PICP (Mean)': picp_arr.mean(), 'PICP (Std)': picp_arr.std(ddof=1),
                      'MPIW (Mean)': mpiw_arr.mean(), 'MPIW (Std)': mpiw_arr.std(ddof=1),
                      'ECE (Mean)': ece_arr.mean(), 'ECE (Std)': ece_arr.std(ddof=1)})
    final_df = pd.DataFrame(rows)
    final_df.to_csv(os.path.join(LOG_DIR, 'step0c_leakfree_summary.csv'), index=False)
    print("\n" + final_df.to_string(index=False))
    n_ckpt = len([f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')])
    print(f"\nCheckpoints saved: {n_ckpt} / {len(DATASETS) * len(SEEDS)} expected")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("STEP0c (leakfree: selection + scaler) complete.")
