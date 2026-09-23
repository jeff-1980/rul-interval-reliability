"""
STEP 0d: trains 10 additional seeds for the Deep Ensemble scale sweep
(leakfree protocol), 5 original + 10 new = 15 seeds/dataset. The seed
values reuse the same deterministically-generated batch from an earlier
non-leakfree version (not regenerated, to keep seed identity comparable
with the old 15-seed pool):
  [8601, 16713, 19906, 21852, 38466, 40016, 40967, 51778, 52287, 90270]

Boundaries (unchanged, matching an earlier extra-seeds retraining script
(not included; superseded)):
  (a) The main table still uses only the original 5 seeds; these 15 seeds
      are only for the ensemble-scale sweep, appendix only, not the main
      table.
  (b) Training configuration is identical to the original 5 seeds
      (HeteroscedasticLSTM/gaussian_nll_loss/150 epochs/batch 256/lr
      1e-3); the only difference is that checkpoint selection here uses
      the leakfree protocol (canonical_split's fit/val, scaler fit only on
      fit_units).
  (c) QC: each new seed's clean-test RMSE/PICP/MPIW/ECE is compared against
      the original 5-seed (leakfree) distribution; |new-orig_mean| >
      max(3*orig_std, 5%*|orig_mean|) is flagged as an outlier.
      **Outlier seeds are recorded honestly, not excluded** -- they are
      included in the ensemble-scale sweep as usual, never dropped for
      being an outlier.

canonical split: the three-way split for the 10 additional seeds is
computed on the fly with `C.compute_canonical_split(all_units, seed)`
(same function, same derivation logic as STEP A), and written back into
`canonical_splits.json` (appended, not overwriting the existing 5-seed
entries).

Checkpoints are saved to `checkpoints_leakfree/{ds}_LSTM_extraseed{seed}.pt`
(a different filename pattern from the main 5-seed set, to avoid being
accidentally loaded by the STEP1/3/4/5 main pipeline).
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

EXTRA_SEEDS = [8601, 16713, 19906, 21852, 38466, 40016, 40967, 51778, 52287, 90270]
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
HIDDEN_DIM = C.HIDDEN_DIM
LOG_SIGMA_MIN = C.LOG_SIGMA_MIN
LOG_SIGMA_MAX = C.LOG_SIGMA_MAX
Z_SCORE = C.Z_SCORE
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

CANON_PATH = os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')
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

    fit_units, val_units, calib_units = C.compute_canonical_split(all_units, seed)
    CANON[ds_name][str(seed)] = {'fit_units': fit_units, 'val_units': val_units, 'calib_units': calib_units}

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

    print(f"   seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)}  train={elapsed:.1f}s  "
          f"best_val_rmse={best_val_rmse:.3f}")
    print(f"      [official test] RMSE={rmse:.3f} Score={score:.1f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_extraseed{seed}.pt")
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': LOG_SIGMA_MIN, 'log_sigma_max': LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name, 'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
        'selection_protocol': 'canonical_split fit/val, leakfree scaler, extra-seed pool',
    }, ckpt_path)

    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma_np.mean()), 'elapsed_train_s': elapsed, 'best_val_rmse': best_val_rmse}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  EXTRA_SEEDS={EXTRA_SEEDS}")

    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')) as f:
        orig5_results = json.load(f)

    all_out = {}
    qc_report = {}
    t_start_all = time.time()
    for ds in C.DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
        train_df_raw = pd.read_csv(os.path.join(C.DATA_DIR, f'train_{ds}.txt'), sep=r'\s+', header=None, names=col_names)
        all_units = sorted(train_df_raw['unit_nr'].unique().tolist())

        seed_results = [run_one_seed(ds, seed, device, use_amp, all_units) for seed in EXTRA_SEEDS]
        all_out[ds] = seed_results

        orig_rows = orig5_results[ds]
        qc_ds = []
        for metric in ['rmse', 'picp', 'mpiw', 'ece']:
            orig_vals = np.array([r[metric] for r in orig_rows])
            orig_mean, orig_std = orig_vals.mean(), orig_vals.std(ddof=1)
            threshold = max(3 * orig_std, 0.05 * abs(orig_mean))
            for r in seed_results:
                deviation = abs(r[metric] - orig_mean)
                flagged = deviation > threshold
                qc_ds.append({'seed': r['seed'], 'metric': metric, 'value': r[metric],
                               'orig5_mean': float(orig_mean), 'orig5_std': float(orig_std),
                               'threshold': float(threshold), 'deviation': float(deviation),
                               'flagged_out_of_range': bool(flagged)})
                if flagged:
                    print(f"  ⚠️ QC: seed={r['seed']} {metric}={r[metric]:.4f} deviates from orig5 "
                          f"mean={orig_mean:.4f} by {deviation:.4f} (threshold={threshold:.4f}) "
                          f"-- RECORDED, NOT EXCLUDED")
        qc_report[ds] = qc_ds

    total_elapsed = time.time() - t_start_all

    with open(os.path.join(RESULTS_DIR, 'step0d_extra_seeds_leakfree_results.json'), 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    with open(os.path.join(RESULTS_DIR, 'leakfree', 'step0d_qc_report.json'), 'w') as fp:
        json.dump(qc_report, fp, indent=2, default=float)
    with open(CANON_PATH, 'w') as fp:
        json.dump(CANON, fp, indent=2)
    print(f"\nSaved -> step0d_extra_seeds_leakfree_results.json, leakfree/step0d_qc_report.json, "
          f"canonical_splits.json (extended with extra seeds)")

    n_flagged = sum(1 for ds in qc_report for row in qc_report[ds] if row['flagged_out_of_range'])
    n_total_checks = sum(len(qc_report[ds]) for ds in qc_report)
    print(f"QC: {n_flagged}/{n_total_checks} (seed,metric) checks flagged out-of-range across all datasets "
          f"(all still included in downstream ensemble pool per instruction)")

    n_ckpt = sum(len([f for f in os.listdir(CKPT_DIR) if f.startswith(f"{ds}_LSTM_extraseed")]) for ds in C.DATASETS)
    print(f"Checkpoints saved: {n_ckpt} / {len(C.DATASETS) * len(EXTRA_SEEDS)} expected")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("STEP0d complete.")
