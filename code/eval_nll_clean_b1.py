"""
Inference-level, no retraining: re-evaluates three "NLL clean-condition
result files computed as a byproduct of the training scripts" --
checkpoint weights are unchanged, only the fields that depend on the
official test-set labels are recomputed (rmse/score/picp/mpiw/ece;
sigma_mean is independent of y and unchanged). q values / calib-based
quantities are not in these three files and are not touched.

Covers:
  step0c_leakfree_results.json          (LSTM, FD001/FD002/FD004)
  stepFD003_nll_leakfree_results.json   (LSTM, FD003)
  t2_transformer_nll_leakfree_results.json (Transformer, all 4 datasets)

Non-evaluation fields (n_fit_units/n_val_units/n_val_windows/
best_val_rmse/elapsed_train_s, etc.) are kept as-is from the old files
(training was not rerun, so these fields are unchanged anyway).
"""
import os
import json

import numpy as np
import torch

import common as C
import transformer_common as T2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')


def eval_lstm(ds, seed, canon, device, ckpt_dir):
    fit_units = canon[ds][str(seed)]['fit_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    ckpt = os.path.join(ckpt_dir, f"{ds}_LSTM_seed{seed}.pt")
    model = C.load_checkpoint_model(ckpt, device)
    with torch.no_grad():
        mu_out, log_sigma_out = model(X_t)
    mu = np.clip(mu_out.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma = np.exp(log_sigma_out.cpu().numpy().flatten()) * 125.0
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    rmse, score = C.rmse_score(y_test, mu)
    picp, mpiw = C.picp_mpiw(y_test, mu, sigma, z=C.Z_SCORE)
    ece = C.compute_ece(mu, sigma, y_test, np.arange(0.05, 1.00, 0.05))
    return {'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma.mean())}


def eval_transformer(ds, seed, device):
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(
        ds, T2_CANON[ds][str(seed)]['fit_units'])
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)
    with torch.no_grad():
        mu_out, log_sigma_out = model(X_t)
    mu = np.clip(mu_out.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma = np.exp(log_sigma_out.cpu().numpy().flatten()) * 125.0
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    rmse, score = C.rmse_score(y_test, mu)
    picp, mpiw = C.picp_mpiw(y_test, mu, sigma, z=C.Z_SCORE)
    ece = C.compute_ece(mu, sigma, y_test, np.arange(0.05, 1.00, 0.05))
    return {'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma.mean())}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        CANON = json.load(f)
    T2_CANON = CANON

    ckpt_dir_leakfree = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

    # ---- 1. LSTM FD001/FD002/FD004 ----
    path1 = os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')
    with open(path1) as f:
        old1 = json.load(f)
    for ds in C.DATASETS:
        print(f"\n{'=' * 10} LSTM NLL {ds} {'=' * 10}")
        for rec in old1[ds]:
            new_metrics = eval_lstm(ds, rec['seed'], CANON, device, ckpt_dir_leakfree)
            rec.update(new_metrics)
            print(f"  seed={rec['seed']}: RMSE={rec['rmse']:.3f} PICP={rec['picp']:.3f}")
    with open(path1, 'w') as fp:
        json.dump(old1, fp, indent=2, default=float)
    print(f"Saved -> {path1}")

    # ---- 2. LSTM FD003 ----
    path2 = os.path.join(RESULTS_DIR, 'stepFD003_nll_leakfree_results.json')
    with open(path2) as f:
        old2 = json.load(f)
    print(f"\n{'=' * 10} LSTM NLL FD003 {'=' * 10}")
    for rec in old2:
        new_metrics = eval_lstm('FD003', rec['seed'], CANON, device, ckpt_dir_leakfree)
        rec.update(new_metrics)
        print(f"  seed={rec['seed']}: RMSE={rec['rmse']:.3f} PICP={rec['picp']:.3f}")
    with open(path2, 'w') as fp:
        json.dump(old2, fp, indent=2, default=float)
    print(f"Saved -> {path2}")

    # ---- 3. Transformer, all 4 datasets ----
    path3 = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_nll_leakfree_results.json')
    with open(path3) as f:
        old3 = json.load(f)
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"\n{'=' * 10} Transformer NLL {ds} {'=' * 10}")
        for rec in old3[ds]:
            new_metrics = eval_transformer(ds, rec['seed'], device)
            rec.update(new_metrics)
            print(f"  seed={rec['seed']}: RMSE={rec['rmse']:.3f} PICP={rec['picp']:.3f}")
    with open(path3, 'w') as fp:
        json.dump(old3, fp, indent=2, default=float)
    print(f"Saved -> {path3}")

    print("\nNLL clean re-eval complete (no retraining).")
