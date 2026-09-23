"""
FD003 补全 (4/6)：三臂噪声扫描，全部5方法（NLL/MC-Dropout/Ensemble/
CP-norm/MSE）+ NLL/CP-norm冻结σ̂反事实，与
`run_sweep_noise_lstm.py`同一协议，FD003单一工况（臂A=臂B）。

取代`train_lstm_fd003_nll_and_mechanism.py`里只算NLL+CP-norm的窄范围
sweep——本脚本产出的`FD003_sweep_leakfree.json`是完整版，之前那份窄范围
的产出文件保留不覆盖（文件名不同），完整版供代价表/per-engine下游使用。
"""
import os
import json

import numpy as np
import torch

import common as C
import mc_dropout_model as S1
import noise_injection as V4
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

DS = 'FD003'
with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)
with open(os.path.join(RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json')) as f:
    MC_JSON = {str(r['seed']): r for r in json.load(f)}
with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
    CP_JSON = {str(r['seed']): r for r in json.load(f)}

CONF_LEVELS = np.arange(0.05, 1.00, 0.05)
Z_SCORE = 1.645
SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]
PCT_LEVELS = V4.PCT_LEVELS
N_TRIALS = V4.N_TRIALS


def picp_mpiw(y_true, mu, sigma, z=Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi))), float(np.mean(hi - lo))


def metrics_basic(y_true, mu, sigma):
    rmse, score = C.rmse_score(y_true, mu)
    picp, mpiw = picp_mpiw(y_true, mu, sigma)
    ece = C.compute_ece(mu, sigma, y_true, CONF_LEVELS)
    return {'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece, 'sigma_mean': float(sigma.mean())}


def grand_and_marginal_stats(cells, n_trials=N_TRIALS):
    keys = ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']
    out = {}
    arr = {k: np.array([[cells[i][t][k] for t in range(n_trials)] for i in range(5)]) for k in keys}
    for k in keys:
        a = arr[k]
        model_marg = a.mean(axis=1); trial_marg = a.mean(axis=0)
        out[k] = {'grand_mean': float(a.mean()), 'grand_std': float(a.std(ddof=1)),
                   'model_marginal_mean': float(model_marg.mean()), 'model_marginal_std': float(model_marg.std(ddof=1)),
                   'trial_marginal_mean': float(trial_marg.mean()), 'trial_marginal_std': float(trial_marg.std(ddof=1))}
    return out


def load_models_for_seed(seed, device):
    nll_model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt"), device)
    ck = torch.load(os.path.join(CKPT_DIR, f"{DS}_MCDropoutMSE_seed{seed}.pt"), map_location=device, weights_only=False)
    mc_model = S1.MC_LSTM(ck['input_dim'], ck['hidden_dim'], ck['dropout']).to(device)
    mc_model.load_state_dict(ck['state_dict'])
    cp_model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_SplitCP_seed{seed}.pt"), device)
    return nll_model, mc_model, cp_model


def infer_nll(model, X_t, batch=8192):
    mus, ls = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            m, s = model(X_t[i:i + batch])
            mus.append(m.cpu().numpy().flatten()); ls.append(s.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0, np.concatenate(ls)


def infer_mc_dropout(mc_model, X_t, T, aleatory_var, batch=4096, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    mc_model.train()
    all_samples = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            xb = X_t[i:i + batch]
            samples = [mc_model(xb).cpu().numpy().flatten() * 125.0 for _ in range(T)]
            all_samples.append(np.stack(samples))
    samples = np.concatenate(all_samples, axis=1)
    mu = np.clip(samples.mean(0), 0, 125)
    sigma = np.sqrt(aleatory_var + samples.var(0))
    return mu, sigma


def infer_mse_fixed(mc_model, X_t, sigma_fixed, batch=8192):
    mc_model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mus.append(mc_model(X_t[i:i + batch]).cpu().numpy().flatten())
    mu = np.clip(np.concatenate(mus) * 125.0, 0, 125)
    sigma = np.full_like(mu, sigma_fixed)
    return mu, sigma


def run_arm(arm, test_df_raw, true_ruls, feat_cols, device, levels, is_pct, global_std, scalers_by_seed, full_scale=None):
    out = {'NLL': {}, 'MC_Dropout_fixed': {}, 'Deep_Ensemble': {}, 'CP_norm': {}, 'MSE_fixed': {}, 'feat_oob': {},
           'NLL_frozen_sigma': {}, 'CP_norm_frozen_sigma': {}}

    clean_sigma_nll, clean_sigma_cp = {}, {}
    models_by_seed = {}
    for seed in C.SEEDS:
        nll_model, mc_model, cp_model = load_models_for_seed(seed, device)
        models_by_seed[seed] = (nll_model, mc_model, cp_model)
        scaler = scalers_by_seed[seed]
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_clean_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        _, ls_nll = infer_nll(nll_model, X_clean_t)
        clean_sigma_nll[seed] = np.exp(ls_nll) * 125.0
        _, ls_cp = infer_nll(cp_model, X_clean_t)
        clean_sigma_cp[seed] = np.exp(ls_cp) * 125.0

    # 2026-09-21 公平校准修复：aleatory_var 改在 calib_units 上现算。
    aleatory_var_calib_by_seed = {
        seed: E.calib_aleatory_var(DS, 'LSTM', seed, device, mc_model=models_by_seed[seed][1])
        for seed in C.SEEDS
    }

    for level in levels:
        level_key = ('inf' if (not is_pct and np.isinf(level)) else str(level))
        trial_X, trial_y, trial_feat_oob = {}, None, []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(DS, arm, level_key, t)))
            if is_pct:
                raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, level, rng, full_scale)
            else:
                raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                # 2026-09-21 f_oob 口径统一：改在实际送入模型的末端窗口 X_test 上算，
                # 见 run_sweep_noise_lstm.py 同日同条注释。
                # R9-Part3: V4.feat_oob 全项目唯一实现（FD003 无工况设定列，
                # sensor_mask 全True，数值上是no-op，但保持全项目同一实现）。
                trial_feat_oob.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))

        nll_cells = [[None] * N_TRIALS for _ in range(5)]
        mc_cells = [[None] * N_TRIALS for _ in range(5)]
        cp_cells = [[None] * N_TRIALS for _ in range(5)]
        mse_cells = [[None] * N_TRIALS for _ in range(5)]
        nll_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
        cp_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
        nll_mu_grid = [[None] * N_TRIALS for _ in range(5)]
        nll_sigma_grid = [[None] * N_TRIALS for _ in range(5)]

        for i, seed in enumerate(C.SEEDS):
            nll_model, mc_model, cp_model = models_by_seed[seed]
            aleatory_var = aleatory_var_calib_by_seed[seed]
            q_norm = CP_JSON[str(seed)]['cp_norm']['q']
            q_norm_by_level = CP_JSON[str(seed)]['cp_norm']['q_by_level']

            for t in range(N_TRIALS):
                X_t = trial_X[t][seed]

                mu_n, ls_n = infer_nll(nll_model, X_t)
                sigma_n = np.exp(ls_n) * 125.0
                nll_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n)
                nll_mu_grid[i][t] = mu_n; nll_sigma_grid[i][t] = sigma_n
                sigma_n_frozen = clean_sigma_nll[seed]
                nll_frozen_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n_frozen)

                mc_seed = C.stable_seed(DS, arm, level_key, t, seed, 'mc_dropout')
                mu_m, sigma_m = infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
                mc_cells[i][t] = metrics_basic(trial_y, mu_m, sigma_m)

                mu_mse, sigma_mse = infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
                mse_cells[i][t] = metrics_basic(trial_y, mu_mse, sigma_mse)

                mu_c, ls_c = infer_nll(cp_model, X_t)
                sigma_c = np.exp(ls_c) * 125.0
                picp_c, mpiw_c = picp_mpiw(trial_y, mu_c, sigma_c, z=q_norm)
                rmse_c, score_c = C.rmse_score(trial_y, mu_c)
                emp = []
                for p in CONF_LEVELS:
                    q = q_norm_by_level[f"{p:.2f}"]
                    lo, hi = mu_c - q * sigma_c, mu_c + q * sigma_c
                    emp.append(np.mean((trial_y >= lo) & (trial_y <= hi)))
                ece_c = float(np.mean(np.abs(np.array(emp) - CONF_LEVELS)))
                cp_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_c, 'mpiw': mpiw_c,
                                   'ece': ece_c, 'sigma_mean': float(sigma_c.mean())}
                sigma_c_frozen = clean_sigma_cp[seed]
                picp_cf, mpiw_cf = picp_mpiw(trial_y, mu_c, sigma_c_frozen, z=q_norm)
                cp_frozen_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_cf, 'mpiw': mpiw_cf,
                                          'ece': float('nan'), 'sigma_mean': float(sigma_c_frozen.mean())}

        out['NLL'][level_key] = grand_and_marginal_stats(nll_cells)
        out['MC_Dropout_fixed'][level_key] = grand_and_marginal_stats(mc_cells)
        out['CP_norm'][level_key] = grand_and_marginal_stats(cp_cells)
        out['MSE_fixed'][level_key] = grand_and_marginal_stats(mse_cells)
        out['NLL_frozen_sigma'][level_key] = grand_and_marginal_stats(nll_frozen_cells)
        out['CP_norm_frozen_sigma'][level_key] = grand_and_marginal_stats(cp_frozen_cells)

        ens_trial_metrics = []
        for t in range(N_TRIALS):
            mu_mem = np.stack([nll_mu_grid[i][t] for i in range(5)])
            sigma_mem = np.stack([nll_sigma_grid[i][t] for i in range(5)])
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            ens_trial_metrics.append(metrics_basic(trial_y, mu_ens, sigma_ens))
        ens_arr = {k: np.array([m[k] for m in ens_trial_metrics]) for k in ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']}
        out['Deep_Ensemble'][level_key] = {k: {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'n_trials': N_TRIALS}
                                            for k, v in ens_arr.items()}

        print(f"    {level_key:>6}: feat_oob={out['feat_oob'][level_key]:.4f}  "
              f"NLL_PICP={out['NLL'][level_key]['picp']['grand_mean']:.3f}  "
              f"MC_PICP={out['MC_Dropout_fixed'][level_key]['picp']['grand_mean']:.3f}  "
              f"Ens_PICP={out['Deep_Ensemble'][level_key]['picp']['mean']:.3f}  "
              f"CP_PICP={out['CP_norm'][level_key]['picp']['grand_mean']:.3f}  "
              f"MSE_PICP={out['MSE_fixed'][level_key]['picp']['grand_mean']:.3f}")

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    scalers_by_seed = {}
    for seed in C.SEEDS:
        fit_units = CANON[DS][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(DS, fit_units)
        scalers_by_seed[seed] = scaler
    global_std = np.std(train_df_raw[feat_cols].values, axis=0)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)

    print("  --- main+extended SNR grid (single condition) ---")
    main_result = run_arm('B_pooled', test_df_raw, true_ruls, feat_cols, device, SNR_LEVELS_ALL,
                           is_pct=False, global_std=global_std, scalers_by_seed=scalers_by_seed)
    print("  --- Arm C ---")
    armC_result = run_arm('C_fixedpct', test_df_raw, true_ruls, feat_cols, device, PCT_LEVELS,
                           is_pct=True, global_std=None, scalers_by_seed=scalers_by_seed, full_scale=full_scale)

    main_wrapped = {DS: {'A_percondition': main_result, 'B_pooled': main_result,
                          'note': 'FD003 single condition; per-condition degenerates to pooled'}}
    armC_wrapped = {DS: armC_result}

    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json'), 'w') as fp:
        json.dump(main_wrapped, fp, indent=2, default=float)
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json'), 'w') as fp:
        json.dump(armC_wrapped, fp, indent=2, default=float)
    print("\nSaved -> noise_sensitivity_leakfree_FD003.json, noise_sensitivity_leakfree_armC_FD003.json")
    print("FD003 full sweep (5 methods) complete.")
