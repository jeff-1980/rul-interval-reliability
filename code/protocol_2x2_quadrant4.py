"""
Completes the 2x2 evaluation-protocol grid; this is the one quadrant that
still needs training.

Three quadrants already exist:
  original: test-select + full-train scaler (both "leaky")
  val-select fix: val-select + full-train scaler (selection fixed, scaler
    still leaky)
  leakfree (this project's main protocol): val-select + fit-only scaler
    (both clean)
The missing fourth quadrant:
  test-select + fit-only scaler (scaler fixed, but the selection criterion
  still uses the test set)
This script trains that quadrant: LSTM, 5 seeds, FD001/FD002/FD004 (same
comparison scope as the other three quadrants; excludes FD003, which
never had a leaky baseline to begin with -- see Limitations).

Training configuration (hidden size/dropout/log_sigma range/epoch/batch/lr)
is identical to the other three quadrants; the scaler uses
canonical_splits.json's fit_units (the same fit-only scaler as the
leakfree quadrant, not a newly-fit "wrong" scaler); checkpoint selection
uses official test-set RMSE (same as the original quadrant, not
val_units).
"""
import os
import json
import time
import random
import copy
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CALIB_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'calibration_and_controls')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm_drift_controls')
os.makedirs(CKPT_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD004']
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256

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

    fit_units = CANON[ds_name][str(seed)]['fit_units']
    # fit-only scaler (identical definition to the leakfree quadrant), but
    # trains on all of fit_units (no separate val_units carve-out here --
    # this quadrant's selection criterion is the test set, not val, so
    # training data stays as fit_units, matching the original quadrant's
    # "train on all training data" spirit, just with "all" scoped to
    # fit_units since the scaler itself is only defined on fit_units)
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = C.HeteroscedasticLSTM(input_dim, C.HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=C.LOG_SIGMA_MIN, log_sigma_max=C.LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_test_rmse = float('inf')
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

        # selection criterion: official test-set RMSE (same as the original protocol, not val_units)
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mu_test, _ = model(X_test_t)
            mu_test = np.clip(mu_test.float().cpu().numpy().flatten() * 125.0, 0, 125)
            curr_test_rmse = np.sqrt(mean_squared_error(y_test, mu_test))
        if curr_test_rmse < best_test_rmse:
            best_test_rmse = curr_test_rmse
            best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=C.Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, np.arange(0.05, 1.00, 0.05))

    print(f"   [{ds_name}] seed={seed}  train={elapsed:.1f}s  best_test_rmse={best_test_rmse:.3f}  "
          f"RMSE={rmse:.3f} Score={score:.1f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_testselect_fitonlyscaler_seed{seed}.pt")
    torch.save({'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': C.HIDDEN_DIM,
                'dropout': 0.2, 'log_sigma_min': C.LOG_SIGMA_MIN, 'log_sigma_max': C.LOG_SIGMA_MAX,
                'seed': seed, 'dataset': ds_name, 'fit_units': fit_units,
                'selection_protocol': 'test-set RMSE selection + fit-only scaler'}, ckpt_path)

    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'best_test_rmse_selection': best_test_rmse, 'elapsed_train_s': elapsed}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  test-select + fit-only-scaler quadrant  DATASETS={DATASETS}  SEEDS={C.SEEDS}")

    out_path = os.path.join(CALIB_DIR, 'protocol_2x2_quadrant4_testselect_fitonlyscaler.json')
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_out = json.load(f)

    for ds in DATASETS:
        if ds in all_out and len(all_out[ds]) == len(C.SEEDS):
            print(f"[{ds}] already complete, skip")
            continue
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        all_out[ds] = [run_one_seed(ds, seed, device, use_amp) for seed in C.SEEDS]
        with open(out_path, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)

    print(f"\nSaved -> {out_path}")
    print("quadrant-4 protocol complete.")
