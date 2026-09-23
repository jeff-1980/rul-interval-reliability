"""
Same-split ensemble control (the only training step in this round).

Purpose: the existing Deep_Ensemble(M=5)'s 5 member models differ in both
random initialisation and canonical_splits' engine-level fit/val/calib
split (the seed controls both at once) -- the ensemble's variance mixes
"split variance" with "initialisation variance", which can't be attributed
separately. This script fixes canonical_splits[ds]['42']'s fit/val/calib
split and varies only the 5 models' initialisation seeds
{42,2024,7,888,123}, giving an ensemble under "pure initialisation
variance" to compare side-by-side with the existing cross-split ensemble.

The init_seed=42 model is identical to the existing
checkpoints_leakfree*/{ds}_{backbone}_seed42.pt (same split_seed=42 +
init_seed=42), reused directly, not retrained; only 4 new models
(init_seed in {2024,7,888,123}) per (backbone,ds) need training:
2 backbones x 4 datasets x 4 new inits = 32 new models.

Training protocol (architecture/hyperparameters/EPOCHS/checkpoint
selection criterion) matches train_lstm.py (LSTM) / train_transformer_nll.py
(Transformer) exactly; the only difference: torch.manual_seed uses
init_seed, but fit_units/val_units are fixed from canonical_splits[ds]['42'],
not varying with init_seed.

Gate-check (evaluation-protocol self-certification, five items):
  1. Checkpoint-selection DataLoader source = val_units (X_val_t), not test.
  2. Early-stopping criterion: same, same val_units, compared via
     best_val_rmse each epoch.
  3. Scaler fit only on fit_units (split_seed=42) (C.load_and_process_leakfree).
  4. No conformal calibration this round (only NLL + moment-matching
     ensemble trained), so no calib-overlap concern; a future CP-norm
     would need the same split_seed=42's calib_units.
  5. fit/val/calib pairwise disjointness: already asserted non-overlapping
     when canonical_splits.json was generated; this script re-asserts
     fit_units intersect val_units = empty and neither overlaps
     calib_units at runtime, as its own self-certification, not trusting
     the earlier assertion.

Computes only the single-model NLL + Deep_Ensemble(M=5) (moment-matching),
not MC-Dropout/CP-norm -- this task only cares about the "pure
initialisation variance under the same split" contrast, not a full cost
table.

Output: results/generated/leakfree_r4/samesplit_ensemble.json
Checkpoint directory: results/generated/checkpoints_leakfree_r4_samesplit/
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
import sweep_engine as E
import noise_injection as V4
from interval_score import interval_score, wis, ALPHA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R4_DIR = os.path.join(RESULTS_DIR, 'leakfree_r4')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'samesplit_control')
os.makedirs(R4_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
INIT_SEEDS = C.SEEDS  # [42, 2024, 7, 888, 123]
SPLIT_SEED = 42
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()


def gate_check_split(ds, fit_units, val_units, calib_units):
    fit_s, val_s, calib_s = set(fit_units), set(val_units), set(calib_units)
    assert len(fit_s & val_s) == 0, f"{ds}: fit/val overlap!"
    assert len(fit_s & calib_s) == 0, f"{ds}: fit/calib overlap!"
    assert len(val_s & calib_s) == 0, f"{ds}: val/calib overlap!"
    return True


def ckpt_path_r4(backbone, ds, init_seed):
    return os.path.join(CKPT_DIR, f"{ds}_{backbone}_splitseed{SPLIT_SEED}_initseed{init_seed}.pt")


def train_lstm(ds, init_seed, fit_units, val_units, device, use_amp):
    torch.manual_seed(init_seed); np.random.seed(init_seed); random.seed(init_seed)
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=256, shuffle=True, pin_memory=(device.type == 'cuda'),
        num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = C.HeteroscedasticLSTM(input_dim, C.HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=C.LOG_SIGMA_MIN, log_sigma_max=C.LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse = float('inf'); best_state = None
    t0 = time.time()
    for epoch in range(150):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
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
    model.load_state_dict(best_state)

    ckpt_path = ckpt_path_r4('LSTM', ds, init_seed)
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': C.HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': C.LOG_SIGMA_MIN, 'log_sigma_max': C.LOG_SIGMA_MAX,
        'seed': init_seed, 'dataset': ds, 'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': 150,
        'selection_protocol': f'same-split ensemble control: split_seed={SPLIT_SEED}, init_seed={init_seed}',
    }, ckpt_path)
    print(f"    [LSTM/{ds}] init_seed={init_seed} train={elapsed:.1f}s best_val_rmse={best_val_rmse:.3f} -> {ckpt_path}")
    return ckpt_path


def train_transformer(ds, init_seed, fit_units, val_units, device, use_amp):
    torch.manual_seed(init_seed); np.random.seed(init_seed); random.seed(init_seed)
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=T2.T2_BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
        num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = T2.HeteroscedasticTransformer(
        input_dim, T2.T2_HIDDEN_DIM, T2.T2_DROPOUT, T2.SEQUENCE_LENGTH,
        T2.T2_LOG_SIGMA_MIN, T2.T2_LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=T2.T2_LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse = float('inf'); best_state = None
    t0 = time.time()
    for epoch in range(T2.T2_EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
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
    model.load_state_dict(best_state)

    ckpt_path = ckpt_path_r4('Transformer', ds, init_seed)
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': T2.T2_HIDDEN_DIM,
        'dropout': T2.T2_DROPOUT, 'log_sigma_min': T2.T2_LOG_SIGMA_MIN, 'log_sigma_max': T2.T2_LOG_SIGMA_MAX,
        'seed': init_seed, 'dataset': ds, 'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': T2.T2_EPOCHS,
        'selection_protocol': f'same-split ensemble control: split_seed={SPLIT_SEED}, init_seed={init_seed}',
    }, ckpt_path)
    print(f"    [Transformer/{ds}] init_seed={init_seed} train={elapsed:.1f}s best_val_rmse={best_val_rmse:.3f} -> {ckpt_path}")
    return ckpt_path


def load_model_generic(backbone, ckpt_path, device):
    return T2.load_checkpoint_model_t2(backbone, ckpt_path, device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  INIT_SEEDS={INIT_SEEDS}  SPLIT_SEED={SPLIT_SEED}")

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            split = CANON[ds][str(SPLIT_SEED)]
            fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']
            gate_check_split(ds, fit_units, val_units, calib_units)
            print(f"  gate_check OK: n_fit={len(fit_units)} n_val={len(val_units)} n_calib={len(calib_units)} "
                  f"(pairwise disjoint, asserted at runtime)")

            ckpt_paths = {}
            for init_seed in INIT_SEEDS:
                if init_seed == SPLIT_SEED:
                    existing = T2.nll_ckpt_path(backbone, ds, SPLIT_SEED)
                    assert os.path.exists(existing), f"expected existing checkpoint missing: {existing}"
                    ckpt_paths[init_seed] = existing
                    print(f"    [{backbone}/{ds}] init_seed={init_seed} reused (split_seed==init_seed): {existing}")
                    continue
                if backbone == 'LSTM':
                    ckpt_paths[init_seed] = train_lstm(ds, init_seed, fit_units, val_units, device, use_amp)
                else:
                    ckpt_paths[init_seed] = train_transformer(ds, init_seed, fit_units, val_units, device, use_amp)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # ---- inference: terminal-unit (clean) for all 5 models ----
            _, test_df_raw, true_ruls_raw, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
            _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
            train_df_lf, test_df_lf, true_ruls_lf, feat_cols_lf, _ = C.load_and_process_leakfree(ds, fit_units)
            X_test, y_test = C.create_sequences(test_df_lf, feat_cols_lf, mode='test', true_ruls=true_ruls_lf)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df_lf, feat_cols_lf, true_ruls_lf)
            X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)

            mu_members, sigma_members = [], []
            mu_full_members, sigma_full_members = [], []
            single_model_rows = []
            for init_seed in INIT_SEEDS:
                model = load_model_generic(backbone, ckpt_paths[init_seed], device)
                mu, ls = E.infer_nll(model, X_test_t)
                sigma = np.exp(ls) * 125.0
                mu_members.append(mu); sigma_members.append(sigma)

                mu_f, ls_f = E.infer_nll(model, X_full_t)
                sigma_f = np.exp(ls_f) * 125.0
                mu_full_members.append(mu_f); sigma_full_members.append(sigma_f)

                picp_i, mpiw_i = C.picp_mpiw(y_test, mu, sigma, z=C.Z_SCORE)
                ece_i = C.compute_ece(mu, sigma, y_test, CONF_LEVELS)
                is_i = interval_score(y_test, mu, sigma, C.Z_SCORE)
                wis_i = wis(y_test, mu, sigma, C.Z_SCORE)
                single_model_rows.append({
                    'init_seed': init_seed, 'picp': picp_i, 'mpiw': mpiw_i, 'ece': ece_i,
                    'is_mean': float(is_i.mean()), 'is_std': float(is_i.std(ddof=1)),
                    'wis_mean': float(wis_i.mean()),
                })
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            is_means_across_models = np.array([r['is_mean'] for r in single_model_rows])

            # ---- Deep-Ensemble(M=5), terminal unit ----
            mu_mem = np.stack(mu_members); sigma_mem = np.stack(sigma_members)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            picp_ens, mpiw_ens = C.picp_mpiw(y_test, mu_ens, sigma_ens, z=C.Z_SCORE)
            ece_ens = C.compute_ece(mu_ens, sigma_ens, y_test, CONF_LEVELS)
            is_ens = interval_score(y_test, mu_ens, sigma_ens, C.Z_SCORE)
            wis_ens = wis(y_test, mu_ens, sigma_ens, C.Z_SCORE)

            # ---- Deep-Ensemble(M=5), full-trajectory unit -> per-engine compliance ----
            mu_full_mem = np.stack(mu_full_members); sigma_full_mem = np.stack(sigma_full_members)
            mu_full_ens = mu_full_mem.mean(0)
            sigma2_full_ens = (sigma_full_mem ** 2 + mu_full_mem ** 2).mean(0) - mu_full_ens ** 2
            sigma_full_ens = np.sqrt(np.clip(sigma2_full_ens, 0, None))
            mu_full_ens = np.clip(mu_full_ens, 0, 125)
            covered_full = (y_full >= mu_full_ens - C.Z_SCORE * sigma_full_ens) & \
                           (y_full <= mu_full_ens + C.Z_SCORE * sigma_full_ens)
            per_engine_picp = {int(u): float(np.mean(covered_full[u_full == u])) for u in np.unique(u_full)}
            compliance_rate_ge_090 = float(np.mean([v >= 0.90 for v in per_engine_picp.values()]))

            result[backbone][ds] = {
                'split_seed': SPLIT_SEED, 'init_seeds': INIT_SEEDS,
                'gate_check': 'PASS: fit/val/calib pairwise disjoint (runtime-asserted), '
                               'checkpoint selection + early stop both on val_units only, '
                               'scaler fit only on fit_units (split_seed=42)',
                'single_models': single_model_rows,
                'single_model_is_mean_across_inits': float(is_means_across_models.mean()),
                'single_model_is_std_across_inits': float(is_means_across_models.std(ddof=1)),
                'ensemble_same_split': {
                    'picp': picp_ens, 'mpiw': mpiw_ens, 'ece': ece_ens,
                    'is_mean': float(is_ens.mean()), 'is_std': float(is_ens.std(ddof=1)),
                    'wis_mean': float(wis_ens.mean()),
                    'per_engine_compliance_rate_ge_090': compliance_rate_ge_090,
                    'n_engines': len(per_engine_picp),
                },
            }
            print(f"  [{backbone}/{ds}] same-split ensemble: PICP={picp_ens:.4f} MPIW={mpiw_ens:.2f} "
                  f"ECE={ece_ens:.4f} IS={is_ens.mean():.3f} per_engine={compliance_rate_ge_090:.3f}  "
                  f"single-model IS across inits: {is_means_across_models.mean():.3f}"
                  f"+/-{is_means_across_models.std(ddof=1):.3f}")

    out_path = os.path.join(R4_DIR, 'samesplit_ensemble.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
