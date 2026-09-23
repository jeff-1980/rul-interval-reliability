"""
Table I controlled 2x2 retraining -- the only training step in this experiment.

Problem: the original T/W cell fit its checkpoint on 100% of the training
engines, and the original V/W cell on 80%, while V/F and T/F both use
canonical_splits.json's 60% fit_units (leaving 20% for calib). This means
the T/W vs V/F/T/F comparisons mixed in an extra "amount of fit data"
variable, when Table I's Sel./Norm./Int. contrasts are supposed to isolate
only "selection criterion" and "scaler fit scope".

This script retrains the T/W' and V/W' cells:
  fit_units = canonical_splits.json's 60% fit set for that seed (identical
              to V/F and T/F)
  scaler    = fit on the whole official training file (W, unfiltered, same
              as the original T/W/V/W)
  T/W' selection: official test-set RMSE (same criterion as the original
                  T/W and as T/F)
  V/W' selection: val_units RMSE (same criterion as the original V/W and
                  as V/F)
  epochs/optimizer/architecture/target definition (the "next-cycle RUL"
  label from C.create_sequences(mode='train')) match V/F (train_lstm.py)
  and T/F (protocol_2x2_quadrant4.py) exactly.

LSTM only, FD001/FD002/FD004, 5 seeds: 2 cells x 3 datasets x 5 seeds = 30 models.

gate_check reuses samesplit_ensemble_control.gate_check_split, asserting
fit_units/val_units/calib_units are pairwise disjoint (applies to both
T/W' and V/W', since those three sets are defined identically to V/F/T/F
-- T/W' selecting on the test set is, by Table I's own definition, the
cell's entire purpose, reproducing the historical test-set-selection leak
as a controlled comparison point, not a leak to be asserted away). Each
checkpoint's metadata explicitly records this intentional exception, so
it's neither silently passed nor mistakenly reported as an assertion
failure.
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
from samesplit_ensemble_control import gate_check_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
TABLE1_RETRAIN_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'table1_controlled_retrain')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm_2x2_controlled')
os.makedirs(TABLE1_RETRAIN_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD004']
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


def run_one_cell(cell, ds_name, seed, device, use_amp):
    """cell: 'TW' (test-select) or 'VW' (val-select). Both use the whole-file
    scaler (C.load_and_process) and canonical_splits' fit_units for training
    data, differing only in the checkpoint-selection DataLoader source."""
    assert cell in ('TW', 'VW')
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[ds_name][str(seed)]
    fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']
    gate_ok = gate_check_split(ds_name, fit_units, val_units, calib_units)

    # Scaler fit on the whole official training file (W) -- the only
    # difference from V/F/T/F's fit-only scaler (load_and_process_leakfree).
    train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds_name)
    input_dim = len(feat_cols)

    # Training data is still restricted to fit_units (60%, identical to
    # V/F/T/F) -- only the scaler's fit range is "whole file"; the actual
    # training row count matches V/F/T/F.
    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = C.HeteroscedasticLSTM(input_dim, HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=LOG_SIGMA_MIN, log_sigma_max=LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_sel_rmse = float('inf')
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
            if cell == 'TW':
                with autocast(device_type=device.type, enabled=use_amp):
                    mu_sel, _ = model(X_test_t)
                mu_sel = np.clip(mu_sel.float().cpu().numpy().flatten() * 125.0, 0, 125)
                curr_sel_rmse = np.sqrt(mean_squared_error(y_test, mu_sel))
            else:
                with autocast(device_type=device.type, enabled=use_amp):
                    mu_sel, _ = model(X_val_t)
                mu_sel = np.clip(mu_sel.float().cpu().numpy().flatten() * 125.0, 0, 125)
                curr_sel_rmse = np.sqrt(mean_squared_error(y_val_cycles, mu_sel))
        if curr_sel_rmse < best_sel_rmse:
            best_sel_rmse = curr_sel_rmse
            best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, CONF_LEVELS)

    sel_label = 'test-set RMSE' if cell == 'TW' else 'val_units RMSE'
    print(f"   [{ds_name}/{cell}] seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)}  "
          f"train={elapsed:.1f}s  best_sel_rmse({sel_label})={best_sel_rmse:.3f}")
    print(f"      [official test] RMSE={rmse:.3f} Score={score:.1f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    suffix = 'testselect_wholefilescaler' if cell == 'TW' else 'valselect_wholefilescaler'
    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_{suffix}_seed{seed}.pt")
    gate_check_record = {
        'fit_val_calib_pairwise_disjoint': gate_ok,
        'scaler_fit_range': 'whole official training file (all engines, W)',
        'selection_dataloader_source': 'official test set' if cell == 'TW' else 'val_units (20% canonical split)',
        'selection_source_is_test_set': (cell == 'TW'),
        'selection_source_is_test_set_note': (
            "INTENTIONAL for the T/W' cell -- this cell exists specifically to "
            "reproduce the historical test-set-selection leak as a controlled "
            "comparison point against V/F and T/F, not an undetected violation "
            "of the evaluation-protocol gate."
        ) if cell == 'TW' else None,
    }
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': LOG_SIGMA_MIN, 'log_sigma_max': LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name, 'fit_units': fit_units, 'val_units': val_units,
        'cell': cell,
        'selection_protocol': (
            f"{sel_label} selection + whole-file scaler, fit_units=canonical 60% "
            f"(controlled 2x2 retrain, {ds_name} seed {seed})"
        ),
        'gate_check': gate_check_record,
    }, ckpt_path)

    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'best_sel_rmse': best_sel_rmse, 'elapsed_train_s': elapsed, 'gate_check': gate_check_record}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  Controlled 2x2 retrain (T/W', V/W')  DATASETS={DATASETS}  SEEDS={C.SEEDS}")

    out_path = os.path.join(TABLE1_RETRAIN_DIR, 'controlled_2x2_retrain_results.json')
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_out = json.load(f)

    for cell in ('TW', 'VW'):
        all_out.setdefault(cell, {})
        for ds in DATASETS:
            if ds in all_out[cell] and len(all_out[cell][ds]) == len(C.SEEDS):
                print(f"[{cell}/{ds}] already complete, skip")
                continue
            print(f"\n{'=' * 20} {cell} / {ds} {'=' * 20}")
            all_out[cell][ds] = [run_one_cell(cell, ds, seed, device, use_amp) for seed in C.SEEDS]
            with open(out_path, 'w') as fp:
                json.dump(all_out, fp, indent=2, default=float)

    print(f"\nSaved -> {out_path}")
    print("Complete: 30 controlled T/W'/V/W' checkpoints trained.")
