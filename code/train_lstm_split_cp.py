"""
STEP 3b: test-set leakage fix for Split-CP, plus an upgrade to a
three-way split.

Two problems with `an earlier split-CP script (not included; superseded)`:
  (a) checkpoint-selection criterion was official test-set RMSE (the same
      bug as STEP0/STEP1);
  (b) the fit/calib split was derived independently by STEP3 itself
      (80/20), not the same split as STEP0/STEP1's train/val -- so the
      same engine could land in different roles across the three
      pipelines, making methods not strictly comparable.

This script instead uses the (fit_units, val_units) from
`canonical_splits.json` (STEP A), **identical** to STEP0c/STEP1b, and
reads calib_units from the same file (disjoint from fit/val, see STEP A's
two-step derivation). Three-way roles:
  fit_units   (60%): training-style full sliding windows, used for
              gradient updates
  val_units   (20%): training-style full sliding windows, the
              checkpoint-selection criterion (RMSE), matching STEP0c/
              STEP1b engine-for-engine
  calib_units (20%): training-style full sliding windows, conformal
              score calibration, never participates in gradient updates
              or checkpoint selection

The scaler is likewise fit only on fit_units (`C.load_and_process_leakfree`).

The CP-abs/CP-norm computation, 20-level ECE convention, and compliance
check are identical to the original STEP3. Checkpoints are saved to
`checkpoints_leakfree/`.
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
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
ALPHA_MAIN = 0.10
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def sequences_for_units(df, feature_cols, unit_set):
    """Training-style full sliding windows, label in raw units (not divided by 125), used for val RMSE and calib."""
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def conformal_quantile(scores, alpha, n):
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    level = k / n
    return float(np.quantile(scores, level, method='higher'))


def picp_mpiw(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper))), float(np.mean(upper - lower))


def run_one_seed(ds_name, seed, device, use_amp):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[ds_name][str(seed)]
    fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']

    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val_raw, y_val_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, val_units)
    y_val_cycles = np.clip(y_val_raw, 0, C.MAX_RUL)
    X_val_t = torch.tensor(X_val_raw, dtype=torch.float32).to(device)

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)

    model = C.HeteroscedasticLSTM(input_dim, C.HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=C.LOG_SIGMA_MIN, log_sigma_max=C.LOG_SIGMA_MAX).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sc = GradScaler(device='cuda', enabled=use_amp)

    def nll_loss(mu, log_sigma, y):
        sigma = torch.exp(log_sigma)
        return (log_sigma + 0.5 * ((y - mu) / sigma) ** 2).mean()

    best_val_rmse, best_state = float('inf'), None
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                mu, ls = model(bx); loss = nll_loss(mu, ls, by)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mv, _ = model(X_val_t)
            mv = np.clip(mv.float().cpu().numpy().flatten() * 125, 0, 125)
            r = np.sqrt(mean_squared_error(y_val_cycles, mv))
        if r < best_val_rmse:
            best_val_rmse = r; best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0
    model.load_state_dict(best_state); model.eval()

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_SplitCP_seed{seed}.pt")
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': C.HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': C.LOG_SIGMA_MIN, 'log_sigma_max': C.LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name, 'fit_units': fit_units,
        'val_units': val_units, 'calib_units': calib_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
        'selection_protocol': 'canonical_split fit/val/calib, leakfree scaler',
    }, ckpt_path)

    # ---- calibration-set inference ----
    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu_c, ls_c = model(X_calib_t)
    mu_calib = np.clip(mu_c.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_calib = torch.exp(ls_c).float().cpu().numpy().flatten() * 125.0

    with torch.no_grad():
        mu_t, ls_t = model(X_test_t)
    mu_test = np.clip(mu_t.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_test = torch.exp(ls_t).float().cpu().numpy().flatten() * 125.0

    rmse, score = C.rmse_score(y_test, mu_test)

    n_calib = len(y_calib)
    s_abs = np.abs(y_calib - mu_calib)
    s_norm = np.abs(y_calib - mu_calib) / np.clip(sigma_calib, 1e-6, None)

    q_abs = conformal_quantile(s_abs, ALPHA_MAIN, n_calib)
    q_norm = conformal_quantile(s_norm, ALPHA_MAIN, n_calib)
    picp_abs, mpiw_abs = picp_mpiw(y_test, mu_test - q_abs, mu_test + q_abs)
    picp_norm, mpiw_norm = picp_mpiw(y_test, mu_test - q_norm * sigma_test, mu_test + q_norm * sigma_test)

    def ece_for_variant(is_norm):
        emp = []; q_by_level = {}
        for p in CONF_LEVELS:
            a = 1 - p
            scores = s_norm if is_norm else s_abs
            q = conformal_quantile(scores, a, n_calib)
            q_by_level[f"{p:.2f}"] = q
            if is_norm:
                lo, hi = mu_test - q * sigma_test, mu_test + q * sigma_test
            else:
                lo, hi = mu_test - q, mu_test + q
            emp.append(np.mean((y_test >= lo) & (y_test <= hi)))
        emp = np.array(emp)
        return float(np.mean(np.abs(emp - CONF_LEVELS))), q_by_level

    ece_abs, q_abs_by_level = ece_for_variant(False)
    ece_norm, q_norm_by_level = ece_for_variant(True)
    compliance_abs = abs(picp_abs - 0.90)
    compliance_norm = abs(picp_norm - 0.90)

    print(f"   seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)} n_calib_units={len(calib_units)} "
          f"n_calib_windows={n_calib}  train={elapsed:.1f}s  best_val_rmse={best_val_rmse:.3f}")
    print(f"      CP-norm: PICP={picp_norm:.3f} MPIW={mpiw_norm:.2f} ECE={ece_norm:.4f} "
          f"|dev|={compliance_norm:.3f}{'  <-- outside +/-0.03 tolerance' if compliance_norm > 0.03 else ''}")

    return {
        'seed': seed, 'train_epochs': EPOCHS, 'elapsed_train_s': elapsed, 'best_val_rmse': best_val_rmse,
        'n_fit_units': len(fit_units), 'n_val_units': len(val_units), 'n_calib_units': len(calib_units),
        'n_calib_windows': int(n_calib), 'rmse': rmse, 'score': score,
        'cp_abs': {'picp': picp_abs, 'mpiw': mpiw_abs, 'ece': ece_abs, 'q': q_abs,
                   'deviation_from_090': compliance_abs, 'q_by_level': q_abs_by_level},
        'cp_norm': {'picp': picp_norm, 'mpiw': mpiw_norm, 'ece': ece_norm, 'q': q_norm,
                    'deviation_from_090': compliance_norm, 'q_by_level': q_norm_by_level},
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  ALPHA={ALPHA_MAIN}  seeds={C.SEEDS}")

    all_out = {}
    for ds in C.DATASETS:
        print(f"\n{'='*20} {ds} {'='*20}")
        seed_results = []
        for seed in C.SEEDS:
            r = run_one_seed(ds, seed, device, use_amp)
            seed_results.append(r)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        all_out[ds] = seed_results

    out_path = os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    print("\n=== summary (mean +/- std(ddof=1), n=5) ===")
    for ds in C.DATASETS:
        rs = all_out[ds]
        for variant in ['cp_abs', 'cp_norm']:
            picps = np.array([r[variant]['picp'] for r in rs])
            mpiws = np.array([r[variant]['mpiw'] for r in rs])
            eces = np.array([r[variant]['ece'] for r in rs])
            dev = np.mean(np.abs(picps.mean() - 0.90))
            print(f"{ds} {variant}: PICP={picps.mean():.4f}+/-{picps.std(ddof=1):.4f}  "
                  f"MPIW={mpiws.mean():.2f}+/-{mpiws.std(ddof=1):.2f}  ECE={eces.mean():.4f}+/-{eces.std(ddof=1):.4f}  "
                  f"|mean_PICP-0.90|={dev:.4f}{'  <-- outside 0.03 tolerance' if dev > 0.03 else ''}")
    print("\nSTEP3b complete.")
