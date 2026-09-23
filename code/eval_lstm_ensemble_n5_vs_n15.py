"""
STEP 2e: n=5 vs n=15 seed-to-seed variance comparison, leakfree protocol.

The 10 additional seeds are no longer read from a separate training
script's result file (that would conflict with "no retraining" here);
instead, inference is run directly against the existing
`results/checkpoints/lstm/{ds}_LSTM_extraseed{seed}.pt` (the 10 additional
seeds' checkpoints, still the original training weights, not retrained
here), recomputing rmse/picp/mpiw under this repo's unified test-label
convention (min(official RUL-1,125), which applies to all tables from
here on). The 5-seed side reads directly from the already-updated
`step0c_leakfree_results.json`.

Same statistical convention as the old (non-leakfree) version: for each
dataset x metric (rmse/picp/mpiw), compares the original 5 seeds' mean/std
against all 15 seeds' (5 original + 10 additional) mean/std, reporting
std_ratio_15_over_5.
"""
import os
import json

import numpy as np
import torch

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

DATASETS = ['FD001', 'FD002', 'FD004']
EXTRA_SEEDS = [8601, 16713, 19906, 21852, 38466, 40016, 40967, 51778, 52287, 90270]


def eval_extraseed(ds, seed, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{ds}_LSTM_extraseed{seed}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fit_units = ckpt['fit_units']
    model = C.load_checkpoint_model(ckpt_path, device)

    train_df, test_df, true_ruls, feat_cols, _ = C.load_and_process_leakfree(ds, fit_units)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = np.exp(log_sigma_out.cpu().numpy().flatten()) * 125.0

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=C.Z_SCORE)
    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw,
            'sigma_mean': float(sigma_np.mean())}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')) as f:
        n5 = json.load(f)

    results = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        extra10_rows = [eval_extraseed(ds, seed, device) for seed in EXTRA_SEEDS]
        for r in extra10_rows:
            print(f"   seed={r['seed']}  RMSE={r['rmse']:.3f} Score={r['score']:.1f} "
                  f"PICP={r['picp']:.3f} MPIW={r['mpiw']:.2f}")

        results[ds] = {}
        n15_rows = n5[ds] + extra10_rows
        for metric in ['rmse', 'picp', 'mpiw']:
            n5_vals = np.array([r[metric] for r in n5[ds]])
            n15_vals = np.array([r[metric] for r in n15_rows])
            n5_mean, n5_std = float(n5_vals.mean()), float(n5_vals.std(ddof=1))
            n15_mean, n15_std = float(n15_vals.mean()), float(n15_vals.std(ddof=1))
            results[ds][metric] = {
                'n5_mean': n5_mean, 'n5_std': n5_std,
                'n15_mean': n15_mean, 'n15_std': n15_std,
                'std_ratio_15_over_5': n15_std / n5_std if n5_std > 0 else None,
            }
            print(f"{ds} {metric}: n5={n5_mean:.4f}±{n5_std:.4f}  n15={n15_mean:.4f}±{n15_std:.4f}  "
                  f"ratio={results[ds][metric]['std_ratio_15_over_5']:.3f}")

    out_path = os.path.join(LEAKFREE_DIR, 'n5_vs_n15_variance_comparison_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP2e complete (extra-10-seed metrics recomputed by inference from existing "
          "checkpoints under min(official_RUL-1,125), no retraining).")
