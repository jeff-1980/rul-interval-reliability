"""
T2-A1：Transformer 骨干，NLL 头，leakfree 协议，4 数据集 × 5 seeds = 20 模型。

架构/超参逐字复用 E3_run_save_ece.py 的 HeteroscedasticTransformer（见
transformer_common.py 顶部说明）；训练协议（checkpoint 选择/早停用 canonical_split
的 val_units、scaler 只在 fit_units 上 fit、engine 级三向切分互不重叠）与
LSTM 的 train_lstm.py / train_lstm_extra_seeds.py
完全一致——这是与 LSTM 可比的前提，两条线除了模型本身，训练/选择协议逐字相同。
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
import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

CANON_PATH = os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')
with open(CANON_PATH) as f:
    CANON = json.load(f)


def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()


def run_one_seed(ds_name, seed, device, use_amp, all_units):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    if str(seed) in CANON.get(ds_name, {}):
        split = CANON[ds_name][str(seed)]
        fit_units, val_units = split['fit_units'], split['val_units']
    else:
        fit_units, val_units, calib_units = C.compute_canonical_split(all_units, seed)
        CANON.setdefault(ds_name, {})[str(seed)] = {
            'fit_units': fit_units, 'val_units': val_units, 'calib_units': calib_units}

    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=T2.T2_BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = T2.HeteroscedasticTransformer(
        input_dim, T2.T2_HIDDEN_DIM, T2.T2_DROPOUT, T2.SEQUENCE_LENGTH,
        T2.T2_LOG_SIGMA_MIN, T2.T2_LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=T2.T2_LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(T2.T2_EPOCHS):
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
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=C.Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, CONF_LEVELS)

    print(f"   [{ds_name}] seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)}  train={elapsed:.1f}s  "
          f"best_val_rmse={best_val_rmse:.3f}")
    print(f"      [official test] RMSE={rmse:.3f} Score={score:.1f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = T2.nll_ckpt_path('Transformer', ds_name, seed)
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': T2.T2_HIDDEN_DIM,
        'dropout': T2.T2_DROPOUT, 'log_sigma_min': T2.T2_LOG_SIGMA_MIN, 'log_sigma_max': T2.T2_LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name, 'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': T2.T2_EPOCHS,
        'selection_protocol': 'canonical_split fit/val, leakfree scaler, Transformer backbone (T2, 2026-09-18)',
    }, ckpt_path)

    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma_np.mean()), 'elapsed_train_s': elapsed, 'best_val_rmse': best_val_rmse}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  Backbone=Transformer(NLL)  DATASETS={DATASETS}  SEEDS={C.SEEDS}")

    all_out = {}
    t_start_all = time.time()
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
        train_df_raw = pd.read_csv(os.path.join(C.DATA_DIR, f'train_{ds}.txt'), sep=r'\s+', header=None, names=col_names)
        all_units = sorted(train_df_raw['unit_nr'].unique().tolist())

        seed_results = [run_one_seed(ds, seed, device, use_amp, all_units) for seed in C.SEEDS]
        all_out[ds] = seed_results

    total_elapsed = time.time() - t_start_all

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_nll_leakfree_results.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    with open(CANON_PATH, 'w') as fp:
        json.dump(CANON, fp, indent=2)
    print(f"\nSaved -> {out_path}, canonical_splits.json (unchanged if all seeds pre-existed)")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("T2-A1 (Transformer NLL) complete.")
