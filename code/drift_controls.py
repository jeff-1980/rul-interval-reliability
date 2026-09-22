"""
R2-5：漂移对照（推理级，不重训）。两骨干 × FD001/FD002，5%FS 档，四个
对照变体：
  (1) reverse   反向斜坡：k·FS -> 0（原版 0 -> k·FS）
  (2) shuffled  同幅值分布时间乱序：同一组幅值，随机打乱到各时间步
  (3) singlech  单通道：只在一个通道上加标准斜坡（其余通道 clean）
  (4) continuous 连续轨迹：在整条测试轨迹（物理时间顺序）上生成连续斜坡
                再切出末端窗口，同一物理时刻在各窗口中扰动一致
对每个变体报 PICP（NLL/MSE/Ensemble）、f_oob、σ̂均值变化（相对clean）。
这是检验"漂移被读成退化"假说的直接对照——如果(4)连续版本的PICP/f_oob
明显不同于原版（每窗口独立重新起算），说明原版效应部分来自"按窗口重新
起算斜坡"这个人为设计，而不是真实连续漂移本身。

singlech 选用 full_scale_range 最大的通道（代表信号绝对幅度最大的传感器）
作为代表通道，理由和选择标准写进输出，不藏在代码里。
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


def run_variant(ds, backbone, variant, device, scalers_by_seed, full_scale, models_by_seed,
                 X_raw_clean, y_ref, channel_idx=None):
    all_picp = {'NLL': [], 'MSE_fixed': [], 'Deep_Ensemble': []}
    all_feat_oob = []
    all_sigma_mean = {'NLL': []}
    nll_mu_grid, nll_sigma_grid = [], []

    mc_by_seed, _ = E.load_mc_cp_for_seed(ds, backbone)

    for t in range(N_TRIALS):
        rng = np.random.RandomState((C.stable_seed(ds, backbone, 'r2_drift_control', variant, t)))
        if variant == 'reverse':
            X_drifted = V4.inject_drift_reverse_fixed_pct_windows(X_raw_clean, PCT, full_scale)
        elif variant == 'shuffled':
            X_drifted = V4.inject_drift_shuffled_fixed_pct_windows(X_raw_clean, PCT, full_scale, rng)
        elif variant == 'singlech':
            X_drifted = V4.inject_drift_singlechannel_fixed_pct_windows(X_raw_clean, PCT, full_scale, channel_idx)
        else:
            raise ValueError(variant)

        trial_mu, trial_sigma, trial_feat_oob = [], [], []
        for seed in C.SEEDS:
            nll_model, mc_model, _ = models_by_seed[seed]
            scaler = scalers_by_seed[seed]
            X_scaled = V4.scale_raw_windows(X_drifted, scaler)
            fo = float(np.mean((X_scaled < -1.0) | (X_scaled > 1.0)))
            trial_feat_oob.append(fo)
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            mu, ls = E.infer_nll(nll_model, X_t)
            sigma = np.exp(ls) * 125.0
            trial_mu.append(mu); trial_sigma.append(sigma)

            # 2026-09-21 公平校准修复：aleatory_var 改用校准集残差方差。
            aleatory_var = E.calib_aleatory_var(ds, backbone, seed, device, mc_model=mc_model)
            mu_mc, sigma_mc = E.infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var)
            mu_mse, sigma_mse = E.infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))

            all_picp['NLL'].append(E.picp_mpiw(y_ref, mu, sigma)[0])
            all_picp['MSE_fixed'].append(E.picp_mpiw(y_ref, mu_mse, sigma_mse)[0])
            all_sigma_mean['NLL'].append(float(sigma.mean()))

        all_feat_oob.append(float(np.mean(trial_feat_oob)))
        nll_mu_grid.append(np.stack(trial_mu)); nll_sigma_grid.append(np.stack(trial_sigma))

    ens_picps = []
    for t in range(N_TRIALS):
        mu_mem = nll_mu_grid[t]; sigma_mem = nll_sigma_grid[t]
        mu_ens = mu_mem.mean(0)
        sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
        sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
        mu_ens = np.clip(mu_ens, 0, 125)
        ens_picps.append(E.picp_mpiw(y_ref, mu_ens, sigma_ens)[0])
    all_picp['Deep_Ensemble'] = ens_picps

    return {
        'feat_oob': float(np.mean(all_feat_oob)),
        'picp_NLL': float(np.mean(all_picp['NLL'])),
        'picp_MSE_fixed': float(np.mean(all_picp['MSE_fixed'])),
        'picp_Deep_Ensemble': float(np.mean(all_picp['Deep_Ensemble'])),
        'sigma_mean_NLL': float(np.mean(all_sigma_mean['NLL'])),
    }


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
            scalers_by_seed = scalers_for(ds, canon)
            channel_idx = int(np.argmax(full_scale))
            print(f"  singlech representative channel: {feat_cols[channel_idx]} (largest full_scale={full_scale[channel_idx]:.2f})")

            models_by_seed = {}
            for seed in C.SEEDS:
                nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
                models_by_seed[seed] = (nll_model, mc_model, None)

            X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')

            # clean baseline (variant-free) for reference
            clean_picps = {'NLL': [], 'MSE_fixed': []}
            for seed in C.SEEDS:
                nll_model, mc_model, _ = models_by_seed[seed]
                scaler = scalers_by_seed[seed]
                X_scaled = V4.scale_raw_windows(X_raw_clean, scaler)
                X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                mu, ls = E.infer_nll(nll_model, X_t)
                sigma = np.exp(ls) * 125.0
                clean_picps['NLL'].append(E.picp_mpiw(y_ref, mu, sigma)[0])

            variants_out = {'clean_picp_NLL': float(np.mean(clean_picps['NLL'])),
                             'singlech_channel': feat_cols[channel_idx]}
            for variant in ['reverse', 'shuffled', 'singlech']:
                r = run_variant(ds, backbone, variant, device, scalers_by_seed, full_scale, models_by_seed,
                                 X_raw_clean, y_ref, channel_idx=channel_idx)
                variants_out[variant] = r
                print(f"  [{variant:10s}] feat_oob={r['feat_oob']:.5f}  NLL_PICP={r['picp_NLL']:.3f}  "
                      f"MSE_PICP={r['picp_MSE_fixed']:.3f}  Ens_PICP={r['picp_Deep_Ensemble']:.3f}  "
                      f"sigma_mean={r['sigma_mean_NLL']:.2f}")

            # variant (4): continuous trajectory -- 需要单独的窗口提取路径
            X_raw_continuous = V4.inject_drift_continuous_trajectory_raw(test_df_raw, feat_cols, PCT, full_scale)
            r4_picp_nll, r4_picp_mse, r4_picp_ens, r4_fo, r4_sigma = [], [], [], [], []
            nll_mu_grid4, nll_sigma_grid4 = [], []
            for seed in C.SEEDS:
                nll_model, mc_model, _ = models_by_seed[seed]
                scaler = scalers_by_seed[seed]
                X_scaled = V4.scale_raw_windows(X_raw_continuous, scaler)
                fo = float(np.mean((X_scaled < -1.0) | (X_scaled > 1.0)))
                r4_fo.append(fo)
                X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                mu, ls = E.infer_nll(nll_model, X_t)
                sigma = np.exp(ls) * 125.0
                nll_mu_grid4.append(mu); nll_sigma_grid4.append(sigma)
                r4_sigma.append(float(sigma.mean()))
                r4_picp_nll.append(E.picp_mpiw(y_ref, mu, sigma)[0])
                aleatory_var = E.calib_aleatory_var(ds, backbone, seed, device, mc_model=mc_model)
                mu_mse, sigma_mse = E.infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
                r4_picp_mse.append(E.picp_mpiw(y_ref, mu_mse, sigma_mse)[0])
            mu_mem = np.stack(nll_mu_grid4); sigma_mem = np.stack(nll_sigma_grid4)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            r4_picp_ens.append(E.picp_mpiw(y_ref, mu_ens, sigma_ens)[0])
            variants_out['continuous'] = {
                'feat_oob': float(np.mean(r4_fo)), 'picp_NLL': float(np.mean(r4_picp_nll)),
                'picp_MSE_fixed': float(np.mean(r4_picp_mse)), 'picp_Deep_Ensemble': float(np.mean(r4_picp_ens)),
                'sigma_mean_NLL': float(np.mean(r4_sigma)),
            }
            print(f"  [continuous] feat_oob={variants_out['continuous']['feat_oob']:.5f}  "
                  f"NLL_PICP={variants_out['continuous']['picp_NLL']:.3f}  "
                  f"MSE_PICP={variants_out['continuous']['picp_MSE_fixed']:.3f}  "
                  f"Ens_PICP={variants_out['continuous']['picp_Deep_Ensemble']:.3f}")

            result[backbone][ds] = variants_out
            for seed in C.SEEDS:
                del models_by_seed[seed]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(R2_DIR, 'drift_controls.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
