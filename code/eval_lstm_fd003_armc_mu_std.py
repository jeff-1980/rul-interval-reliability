"""
FD003 补全：臂C下 μ̂ 跨样本 std 相对 clean 的变化率，与 eval_lstm_armc_mu_std 同一
统计口径，scoped to FD003（single condition, arm C only needed for this
specific number）。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

DS = 'FD003'
N_TRIALS = V4.N_TRIALS

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def infer_mu(model, X_t, batch=8192):
    mus = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mu, _ = model(X_t[i:i + batch])
            mus.append(mu.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)

    models_by_seed, scalers_by_seed = {}, {}
    for seed in C.SEEDS:
        fit_units = CANON[DS][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(DS, fit_units)
        scalers_by_seed[seed] = scaler
        models_by_seed[seed] = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt"), device)

    ds_out = {}
    levels = [('clean', None)] + [(f"{pct}pctFS", pct) for pct in V4.PCT_LEVELS]
    for level_label, pct in levels:
        all_mu = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(DS, 'armC_mustd_leakfree', level_label, t)))
            if pct is None:
                raw_noisy = test_df_raw[feat_cols].values.astype(np.float64)
            else:
                raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, pct, rng, full_scale)
            for seed in C.SEEDS:
                df_noisy, _ = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scalers_by_seed[seed])
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                all_mu.append(infer_mu(models_by_seed[seed], X_t))

        all_mu = np.concatenate(all_mu)
        mu_std = float(np.std(all_mu, ddof=1))
        ds_out[level_label] = {'mu_std_across_samples': mu_std, 'n_predictions_pooled': int(len(all_mu))}
        print(f"  {level_label:10}: mu_std={mu_std:7.3f}")

    mu_clean = ds_out['clean']['mu_std_across_samples']
    for level_label, _ in levels[1:]:
        pct_change = (ds_out[level_label]['mu_std_across_samples'] - mu_clean) / mu_clean * 100.0
        ds_out[level_label]['mu_std_pct_change_vs_clean'] = pct_change
        print(f"  {level_label:10}: mu_std change vs clean = {pct_change:+.1f}%")

    out_path = os.path.join(LEAKFREE_DIR, 'FD003_armC_mu_std_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(ds_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
