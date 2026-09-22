"""
R8-B1（FD003 分支，推理级，不重训）：重跑 train_lstm_fd003_nll_and_mechanism.py
的 STEP4 以后部分（三臂扫描 + clamp_frac + 冻结σ̂分解），跳过 STEP1-3（训练
NLL/CP-norm checkpoint）——R8 全篇"只做推理，不重训"，FD003 的 NLL/SplitCP
checkpoint 权重保持原样，只是重新算一遍它们在新协议（B2: 测试真值
min(官方RUL-1,125)）下的评估指标。FD003 没有工况设定列（get_feature_names
只给14个传感器），B1 的"传感器/设定"限制对它是 no-op，不需要额外处理。

直接 import train_lstm_fd003_nll_and_mechanism 复用 run_sweep_arm /
crossover_feat_oob / interp 等函数（import 本身只读 canonical_splits.json，
不触发任何训练——原脚本的训练代码全部在 `if __name__=='__main__':` 内）。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import train_lstm_fd003_nll_and_mechanism as M

if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(M.RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
        cp_results = json.load(f)
    cp_json = {str(r['seed']): r for r in cp_results}

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(M.DS)
    scalers_by_seed = {}
    for seed in C.SEEDS:
        fit_units = M.CANON[M.DS][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(M.DS, fit_units)
        scalers_by_seed[seed] = scaler
    global_std = np.std(train_df_raw[feat_cols].values, axis=0)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
    # feat_cols has no setting columns for FD003 -- sensor_only_scale is a no-op,
    # applied anyway for consistency with the other active scripts.
    full_scale = V4.sensor_only_scale(feat_cols, full_scale)
    global_std = V4.sensor_only_scale(feat_cols, global_std)

    print("  --- main+extended SNR grid (single condition) ---")
    main_result = M.run_sweep_arm('B_pooled', test_df_raw, true_ruls, feat_cols, device, M.SNR_LEVELS_ALL,
                                   is_pct=False, global_std=global_std, scalers_by_seed=scalers_by_seed,
                                   cp_json=cp_json)
    print("  --- Arm C ---")
    armC_result = M.run_sweep_arm('C_fixedpct', test_df_raw, true_ruls, feat_cols, device, M.PCT_LEVELS,
                                   is_pct=True, global_std=None, scalers_by_seed=scalers_by_seed, cp_json=cp_json,
                                   full_scale=full_scale)

    with open(os.path.join(M.LEAKFREE_DIR, 'FD003_sweep_leakfree.json'), 'w') as fp:
        json.dump({'main': main_result, 'armC': armC_result}, fp, indent=2, default=float)

    print("\n=== Decomposition ===")
    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = [str(p) for p in M.PCT_LEVELS]
    decomposition = {}
    clamp_at_clean = {}
    for method, real_key, frozen_key, clamp_key in [('NLL', 'NLL', 'NLL_frozen_sigma', 'NLL_clamp_frac'),
                                                      ('CP_norm', 'CP_norm', 'CP_norm_frozen_sigma', 'CP_norm_clamp_frac')]:
        pts_real, pts_frozen = [], []
        for k in snr_keys_all:
            fo = main_result['feat_oob'][k]
            pts_real.append((fo, main_result[real_key][k]['picp']['grand_mean']))
            pts_frozen.append((fo, main_result[frozen_key][k]['picp']['grand_mean']))
        for k in pct_keys:
            fo = armC_result['feat_oob'][k]
            pts_real.append((fo, armC_result[real_key][k]['picp']['grand_mean']))
            pts_frozen.append((fo, armC_result[frozen_key][k]['picp']['grand_mean']))

        pts_real_sorted = sorted(pts_real, key=lambda p: p[0])
        pts_frozen_sorted = sorted(pts_frozen, key=lambda p: p[0])
        co = M.crossover_feat_oob(pts_real, threshold=0.80)
        picp_clean_real = min(pts_real_sorted, key=lambda p: p[0])[1]
        picp_clean_frozen = min(pts_frozen_sorted, key=lambda p: p[0])[1]
        clamp_at_clean[method] = main_result[clamp_key]['inf']

        if co is None:
            decomposition[method] = {'note': 'never crosses PICP=0.80 in measured range'}
            continue
        picp_frozen_at_co = M.interp(pts_frozen_sorted, co)
        delta_total = picp_clean_real - 0.80
        delta_mu_only = picp_clean_frozen - picp_frozen_at_co
        delta_sigma = delta_total - delta_mu_only
        sigma_fraction = delta_sigma / delta_total if delta_total != 0 else None
        decomposition[method] = {
            'crossover_feat_oob': co, 'picp_clean_real': picp_clean_real,
            'picp_clean_frozen_sigma_counterfactual': picp_clean_frozen,
            'picp_frozen_sigma_counterfactual_at_crossover': picp_frozen_at_co,
            'delta_picp_total': delta_total, 'delta_picp_mu_only_frozen_sigma_counterfactual': delta_mu_only,
            'delta_picp_sigma_contribution': delta_sigma, 'sigma_contribution_fraction': sigma_fraction,
            'clamp_frac_at_clean': clamp_at_clean[method],
        }
        print(f"{M.DS} {method}: clean_clamp_frac={clamp_at_clean[method]:.3f}  crossover_feat_oob={co:.4f}  "
              f"dPICP_total={delta_total:.4f}  dPICP_mu_only={delta_mu_only:.4f}  "
              f"dPICP_sigma={delta_sigma:.4f}  sigma_contribution_fraction={sigma_fraction:.3f}")

    with open(os.path.join(M.LEAKFREE_DIR, 'FD003_frozen_sigma_decomposition_leakfree.json'), 'w') as fp:
        json.dump(decomposition, fp, indent=2, default=float)
    print("\nSaved -> leakfree/FD003_sweep_leakfree.json, leakfree/FD003_frozen_sigma_decomposition_leakfree.json")
    print("R8-B1 FD003 mechanism eval complete (no retraining).")
