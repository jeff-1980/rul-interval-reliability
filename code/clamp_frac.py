"""
STEP 5k：机制诊断表1（三臂 feat_oob vs clamp_frac 配对序列），leakfree checkpoint。

只算 NLL / CP-norm 的 log_sigma → clamp_frac，不跑 MC-Dropout/Ensemble/MSE
（那些在完整 sweep `run_sweep_noise_lstm.py` 里已经跑过，PICP/σ̂
数字直接复用其输出；本脚本只补 clamp_frac 这一个此前 leakfree sweep 没有
记录的量）。噪声注入协议与完整sweep一致：每个trial的原始噪声共享（同一份
物理扰动），每个seed各自用自己的leakfree scaler标准化。
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
os.makedirs(LEAKFREE_DIR, exist_ok=True)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)

SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]
PCT_LEVELS = V4.PCT_LEVELS
N_TRIALS = V4.N_TRIALS
CLAMP_EPS = V4.CLAMP_EPS


def infer_log_sigma(model, X_t, batch=8192):
    lss = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            _, ls = model(X_t[i:i + batch])
            lss.append(ls.cpu().numpy().flatten())
    return np.concatenate(lss)


def run_arm(ds, arm, test_df_raw, true_ruls, feat_cols, device, levels, is_pct,
            global_std=None, km=None, cond_std=None, full_scale=None, scalers_by_seed=None):
    out = {'feat_oob': {}, 'NLL_clamp_frac': {}, 'CP_norm_clamp_frac': {}}
    nll_models, cp_models = {}, {}
    for seed in C.SEEDS:
        nll_models[seed] = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{ds}_LSTM_seed{seed}.pt"), device)
        cp_models[seed] = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{ds}_SplitCP_seed{seed}.pt"), device)

    for level in levels:
        level_key = ('inf' if (not is_pct and np.isinf(level)) else str(level))
        nll_ls_pool, cp_ls_pool = [], []
        feat_oob_trials = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(ds, arm, level_key, t)))
            if is_pct:
                raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, level, rng, full_scale)
            else:
                raw_noisy = V4.inject_noise_raw(
                    test_df_raw, feat_cols, level, rng,
                    'per_condition' if arm == 'A_percondition' else 'global',
                    global_std=global_std, km=km, cond_std=cond_std)
            fo_this_trial_per_seed = []
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                ls_n = infer_log_sigma(nll_models[seed], X_t)
                ls_c = infer_log_sigma(cp_models[seed], X_t)
                nll_ls_pool.append(ls_n)
                cp_ls_pool.append(ls_c)
                # 2026-09-21（本轮）：此前只在 seed==C.SEEDS[0] 时记一次，且用整段轨迹
                # scaled_feat——同一类此前漏掉的旧bug，见 threshold_crossover_refinement.py
                # refinement.py 同日同条注释。改成5个seed各自在窗口化X_test上算，取平均。
                fo_this_trial_per_seed.append(float(np.mean((X_test < -1.0) | (X_test > 1.0))))
            feat_oob_trials.append(float(np.mean(fo_this_trial_per_seed)))

        nll_clamp = float(np.mean(np.concatenate(nll_ls_pool) <= (C.LOG_SIGMA_MIN + CLAMP_EPS)))
        cp_clamp = float(np.mean(np.concatenate(cp_ls_pool) <= (C.LOG_SIGMA_MIN + CLAMP_EPS)))
        out['feat_oob'][level_key] = float(np.mean(feat_oob_trials))
        out['NLL_clamp_frac'][level_key] = nll_clamp
        out['CP_norm_clamp_frac'][level_key] = cp_clamp
        print(f"    {level_key:>6}: feat_oob={out['feat_oob'][level_key]:.4f}  "
              f"NLL_clamp={nll_clamp:.3f}  CP_clamp={cp_clamp:.3f}")

    for seed in C.SEEDS:
        del nll_models[seed], cp_models[seed]
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    main_results, armC_results = {}, {}
    for ds in C.DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        scalers_by_seed = {}
        for seed in C.SEEDS:
            fit_units = CANON[ds][str(seed)]['fit_units']
            _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
            scalers_by_seed[seed] = scaler
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)

        if ds == 'FD001':
            print("  --- arm B_pooled (single condition, A==B) ---")
            r = run_arm(ds, 'B_pooled', test_df_raw, true_ruls, feat_cols, device, SNR_LEVELS_ALL,
                        is_pct=False, global_std=np.std(train_df_raw[feat_cols].values, axis=0),
                        scalers_by_seed=scalers_by_seed)
            main_results[ds] = {'A_percondition': r, 'B_pooled': r}
        else:
            km, cond_std, global_std = V4.fit_condition_model(train_df_raw, feat_cols)
            print("  --- arm A_percondition ---")
            ra = run_arm(ds, 'A_percondition', test_df_raw, true_ruls, feat_cols, device, SNR_LEVELS_ALL,
                         is_pct=False, km=km, cond_std=cond_std, scalers_by_seed=scalers_by_seed)
            print("  --- arm B_pooled ---")
            rb = run_arm(ds, 'B_pooled', test_df_raw, true_ruls, feat_cols, device, SNR_LEVELS_ALL,
                         is_pct=False, global_std=global_std, scalers_by_seed=scalers_by_seed)
            main_results[ds] = {'A_percondition': ra, 'B_pooled': rb}

        print("  --- Arm C ---")
        armC_results[ds] = run_arm(ds, 'C_fixedpct', test_df_raw, true_ruls, feat_cols, device, PCT_LEVELS,
                                    is_pct=True, full_scale=full_scale, scalers_by_seed=scalers_by_seed)

    with open(os.path.join(LEAKFREE_DIR, 'clamp_frac_leakfree.json'), 'w') as fp:
        json.dump(main_results, fp, indent=2, default=float)
    with open(os.path.join(LEAKFREE_DIR, 'clamp_frac_leakfree_armC.json'), 'w') as fp:
        json.dump(armC_results, fp, indent=2, default=float)
    print("\nSaved -> leakfree/clamp_frac_leakfree.json, leakfree/clamp_frac_leakfree_armC.json")
    print("STEP5k complete.")
