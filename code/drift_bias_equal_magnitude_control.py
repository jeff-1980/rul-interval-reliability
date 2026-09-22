"""
R5-3："等幅值 bias" 对照（main.tex 里 Table~tab:controls 的 "Bias of equal
end magnitude" 行）此前也没有单独的生成脚本留存（同 D1/relative_half_life
的情况）。定义直接取自正文："a bias of magnitude equal to the ramp's end
value"——用 `V4.inject_bias_fixed_pct_raw` 在同一个 PCT=5.0（与漂移斜坡的
终值幅度相同，因为两者都定义成 full_scale 的同一百分比）上做逐通道随机
符号的固定偏置，与 `drift_controls.py` 的 reverse/shuffled/
singlech/continuous 四个对照同一套 NLL/MSE/Ensemble PICP 口径，跑完后把
结果合并写回 `drift_controls.json` 的 'bias_equal' 键（先读一次已重跑过的
drift_controls.json，做原地追加，不覆盖其余变体）。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

DATASETS = ['FD001', 'FD002']
BACKBONES = ['LSTM', 'Transformer']
PCT = 5.0
N_TRIALS = V4.N_TRIALS


def scalers_for(ds, canon):
    s = {}
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        s[seed] = scaler
    return s


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    with open(os.path.join(R2_DIR, 'drift_controls.json')) as f:
        existing = json.load(f)

    for backbone in BACKBONES:
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
            scalers_by_seed = scalers_for(ds, canon)
            X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')

            mc_by_seed, _ = E.load_mc_cp_for_seed(ds, backbone)
            nll_model_by_seed = {seed: T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                                  for seed in C.SEEDS}
            mc_model_by_seed = {seed: T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
                                 for seed in C.SEEDS}

            all_picp = {'NLL': [], 'MSE_fixed': []}
            all_feat_oob = []
            nll_mu_grid, nll_sigma_grid = [], []
            for t in range(N_TRIALS):
                rng = np.random.RandomState(C.stable_seed(ds, backbone, 'r5_bias_equal', t))
                raw_noisy = V4.inject_bias_fixed_pct_raw(test_df_raw, feat_cols, PCT, rng, full_scale)
                df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_noisy
                X_raw, _, _ = V4.extract_raw_windows(df_tmp, feat_cols, true_ruls, mode='test')

                trial_mu, trial_sigma, trial_fo = [], [], []
                for seed in C.SEEDS:
                    scaler = scalers_by_seed[seed]
                    X_scaled = V4.scale_raw_windows(X_raw, scaler)
                    fo = float(np.mean((X_scaled < -1.0) | (X_scaled > 1.0)))
                    trial_fo.append(fo)
                    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                    mu, ls = E.infer_nll(nll_model_by_seed[seed], X_t)
                    sigma = np.exp(ls) * 125.0
                    trial_mu.append(mu); trial_sigma.append(sigma)
                    all_picp['NLL'].append(E.picp_mpiw(y_ref, mu, sigma)[0])

                    # 2026-09-21 公平校准修复：aleatory_var 改用校准集残差方差。
                    aleatory_var = E.calib_aleatory_var(ds, backbone, seed, device, mc_model=mc_model_by_seed[seed])
                    mu_mse, sigma_mse = E.infer_mse_fixed(mc_model_by_seed[seed], X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
                    all_picp['MSE_fixed'].append(E.picp_mpiw(y_ref, mu_mse, sigma_mse)[0])

                all_feat_oob.append(float(np.mean(trial_fo)))
                nll_mu_grid.append(np.stack(trial_mu)); nll_sigma_grid.append(np.stack(trial_sigma))

            ens_picps = []
            for t in range(N_TRIALS):
                mu_mem = nll_mu_grid[t]; sigma_mem = nll_sigma_grid[t]
                mu_ens = mu_mem.mean(0)
                sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
                sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
                mu_ens = np.clip(mu_ens, 0, 125)
                ens_picps.append(E.picp_mpiw(y_ref, mu_ens, sigma_ens)[0])

            r = {
                'feat_oob': float(np.mean(all_feat_oob)),
                'picp_NLL': float(np.mean(all_picp['NLL'])),
                'picp_MSE_fixed': float(np.mean(all_picp['MSE_fixed'])),
                'picp_Deep_Ensemble': float(np.mean(ens_picps)),
            }
            existing.setdefault(backbone, {}).setdefault(ds, {})['bias_equal'] = r
            print(f"  [bias_equal] feat_oob={r['feat_oob']:.5f}  NLL_PICP={r['picp_NLL']:.3f}  "
                  f"MSE_PICP={r['picp_MSE_fixed']:.3f}  Ens_PICP={r['picp_Deep_Ensemble']:.3f}")

            del nll_model_by_seed, mc_model_by_seed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(R2_DIR, 'drift_controls.json')
    with open(out_path, 'w') as fp:
        json.dump(existing, fp, indent=2, default=float)
    print(f"\nSaved (bias_equal merged in) -> {out_path}")
