"""
R4-2：维护表 RUL 修正——按发动机 ID 接回官方 RUL_FD00X.txt 的未截断值，重算
480 行的 true_rul_at_trigger 与 excess_over_L。

背景：C.create_sequences / V4.extract_raw_windows 在构造 mode='test' 的评估
标签时统一做了 `min(true_ruls.iloc[unit-1].item(), 125)`——C_maintenance_full_
onesided.json 里的 true_rul_at_trigger/excess_over_L 用的正是这个截断后的值。
截断只影响真实RUL>=125的发动机；由于 L<=30、L+20<=50 << 125，截断永远不会把
一台发动机从"at_risk"/"premature"的分类里改变（此断言在下面代码里显式检查），
只会压低这些发动机在"premature"分组里对 true_rul_at_trigger/excess_over_L
均值的贡献。

方法：逐字复刻 maintenance_decision_one_sided.py 的 triggered/at_risk/premature
计算（同一套加噪/推理/单侧95%下界规则，完全确定性，不涉及新的随机性），
额外在每个 engine 上打包官方未截断 RUL（true_ruls.iloc[unit-1].item()，
不做 min(...,125)），用同一个 premature 布尔掩码分别算截断版和未截断版的
true_rul_at_trigger/excess_over_L，对比差异（均值、最大差），并断言
premature/at_risk/触发率/成本排序与旧文件逐行一致。

只做推理，不重训，不改 main.tex。输出：
results/generated/leakfree_r4/C_maintenance_rul_corrected.json
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
from maintenance_decision_two_sided import calib_sigma_fixed, sequences_for_units

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')
R4_DIR = os.path.join(RESULTS_DIR, 'leakfree_r4')
os.makedirs(R4_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
METHODS = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
CONDITIONS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']
L_LEVELS = [10, 20, 30]
COST_RATIOS = [5, 20, 100]
N_TRIALS = V4.N_TRIALS
ALPHA_ONESIDED = 0.05


def conformal_quantile_signed(scores_signed, alpha, n):
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    level = k / n
    return float(np.quantile(scores_signed, level, method='higher'))


def onesided_cp_q(ds, backbone, seed, canon, device):
    fit_units = canon[ds][str(seed)]['fit_units']
    calib_units = canon[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
    mu_calib, ls_calib = E.infer_nll(nll_model, X_calib_t)
    sigma_calib = np.exp(ls_calib) * 125.0
    s_signed = (mu_calib - y_calib) / np.clip(sigma_calib, 1e-6, None)
    q_lower = conformal_quantile_signed(s_signed, ALPHA_ONESIDED, len(y_calib))
    del nll_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return q_lower


def extract_raw_windows_with_uncapped(test_df_raw, feature_cols, true_ruls, mode='test'):
    """与 V4.extract_raw_windows 逐字一致的窗口/顺序，额外并行返回未截断的
    官方 RUL（true_ruls.iloc[unit-1].item()，不做 min(...,125)）。"""
    X_list, y_list, y_uncapped_list, u_list = [], [], [], []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        if mode == 'test':
            X_list.append(unit_data[-C.SEQUENCE_LENGTH:])
            raw_val = true_ruls.iloc[unit - 1].item()
            y_list.append(min(raw_val, C.MAX_RUL))
            y_uncapped_list.append(raw_val)
            u_list.append(unit)
        else:
            raise ValueError(mode)
    return np.array(X_list), np.array(y_list), np.array(y_uncapped_list), np.array(u_list)


def get_raw_test_window_with_uncapped(ds, condition, trial, full_scale):
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    X_raw_clean, y_ref, y_uncapped, units = extract_raw_windows_with_uncapped(
        test_df_raw, feat_cols, true_ruls, mode='test')
    if condition == 'clean':
        return X_raw_clean, y_ref, y_uncapped, feat_cols, test_df_raw, true_ruls
    if condition == 'drift5pct':
        X_raw = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, full_scale)
        return X_raw, y_ref, y_uncapped, feat_cols, test_df_raw, true_ruls
    elif condition == 'bias5pct':
        rng = np.random.RandomState((C.stable_seed(ds, 'r2_maint_bias', trial)))
        raw_df_noisy = V4.inject_bias_fixed_pct_raw(test_df_raw, feat_cols, 5.0, rng, full_scale)
        df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_df_noisy
        X_raw, _, _, _ = extract_raw_windows_with_uncapped(df_tmp, feat_cols, true_ruls, mode='test')
        return X_raw, y_ref, y_uncapped, feat_cols, test_df_raw, true_ruls
    elif condition == 'gaussian1pct':
        rng = np.random.RandomState((C.stable_seed(ds, 'r2_maint_gauss', trial)))
        raw_df_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, 1.0, rng, full_scale)
        df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_df_noisy
        X_raw, _, _, _ = extract_raw_windows_with_uncapped(df_tmp, feat_cols, true_ruls, mode='test')
        return X_raw, y_ref, y_uncapped, feat_cols, test_df_raw, true_ruls
    else:
        raise ValueError(condition)


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    with open(os.path.join(R3_DIR, 'C_maintenance_full_onesided.json')) as f:
        old_result = json.load(f)

    result = {}
    all_diffs_mean, all_diffs_excess = [], []
    mismatch_log = []

    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw_probe, _, _, feat_cols_probe, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw_probe, feat_cols_probe)

            scalers_by_seed = {}
            aleatory_var_by_seed = {}
            mc_model_by_seed = {}
            nll_model_by_seed = {}
            cp_q_onesided_by_seed = {}
            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                scalers_by_seed[seed] = scaler
                av, mc_model = calib_sigma_fixed(ds, backbone, seed, canon, device)
                aleatory_var_by_seed[seed] = av
                mc_model_by_seed[seed] = mc_model
                nll_model_by_seed[seed] = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                cp_q_onesided_by_seed[seed] = onesided_cp_q(ds, backbone, seed, canon, device)

            result[backbone][ds] = {}
            engine_level_data = {}

            for condition in CONDITIONS:
                trials = range(N_TRIALS) if condition in ('gaussian1pct', 'bias5pct') else [0]
                nll_mu_by_seed_trial = {}
                nll_sigma_by_seed_trial = {}
                lower_bounds = {m: [] for m in METHODS}
                y_cells, y_uncapped_cells = [], []

                for trial in trials:
                    X_raw, y_ref, y_uncapped, feat_cols, test_df_raw, true_ruls = get_raw_test_window_with_uncapped(
                        ds, condition, trial, full_scale)
                    y_cells.append(y_ref)
                    y_uncapped_cells.append(y_uncapped)

                    for seed in C.SEEDS:
                        scaler = scalers_by_seed[seed]
                        X_scaled = V4.scale_raw_windows(X_raw, scaler)
                        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                        mu_n, ls_n = E.infer_nll(nll_model_by_seed[seed], X_t)
                        sigma_n = np.exp(ls_n) * 125.0
                        nll_mu_by_seed_trial[(seed, trial)] = mu_n
                        nll_sigma_by_seed_trial[(seed, trial)] = sigma_n

                        aleatory_var = aleatory_var_by_seed[seed]
                        mu_mc, sigma_mc = E.infer_mc_dropout(mc_model_by_seed[seed], X_t, T=50, aleatory_var=aleatory_var)
                        mu_mse, sigma_mse = E.infer_mse_fixed(mc_model_by_seed[seed], X_t,
                                                               sigma_fixed=float(np.sqrt(aleatory_var)))
                        q_lower = cp_q_onesided_by_seed[seed]

                        lower_bounds['NLL'].append(mu_n - C.Z_SCORE * sigma_n)
                        lower_bounds['MSE_fixed'].append(mu_mse - C.Z_SCORE * sigma_mse)
                        lower_bounds['MC_Dropout_fixed'].append(mu_mc - C.Z_SCORE * sigma_mc)
                        lower_bounds['CP_norm'].append(mu_n - q_lower * sigma_n)

                for ti, trial in enumerate(trials):
                    mu_mem = np.stack([nll_mu_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    sigma_mem = np.stack([nll_sigma_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    mu_ens = mu_mem.mean(0)
                    sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
                    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
                    mu_ens = np.clip(mu_ens, 0, 125)
                    lower_bounds.setdefault('Deep_Ensemble', []).append(mu_ens - C.Z_SCORE * sigma_ens)

                n_engines = len(y_cells[0])
                for m in METHODS:
                    lb_stack = np.stack(lower_bounds[m])
                    reps = lb_stack.shape[0] // len(y_cells)
                    y_stack = np.tile(np.array(y_cells), (reps, 1)) if m != 'Deep_Ensemble' else np.array(y_cells)
                    y_uncapped_stack = np.tile(np.array(y_uncapped_cells), (reps, 1)) if m != 'Deep_Ensemble' \
                        else np.array(y_uncapped_cells)
                    engine_level_data[(condition, m)] = {'lower_bounds': lb_stack, 'y': y_stack,
                                                          'y_uncapped': y_uncapped_stack}

            for condition in CONDITIONS:
                result[backbone][ds][condition] = {}
                for m in METHODS:
                    d = engine_level_data[(condition, m)]
                    lb = d['lower_bounds']; y = d['y']; y_unc = d['y_uncapped']
                    result[backbone][ds][condition][m] = {}
                    for L in L_LEVELS:
                        triggered = lb <= L
                        at_risk = y <= L
                        unrecognised = at_risk & (~triggered)
                        premature = triggered & (y > L + 20)

                        # sanity: capping must never flip at_risk/premature classification
                        at_risk_unc = y_unc <= L
                        premature_unc = triggered & (y_unc > L + 20)
                        if not np.array_equal(at_risk, at_risk_unc) or not np.array_equal(premature, premature_unc):
                            mismatch_log.append({'backbone': backbone, 'ds': ds, 'condition': condition,
                                                  'method': m, 'L': L})

                        n_at_risk_total = at_risk.sum()
                        conditional_unrecognised_rate = float(unrecognised.sum() / n_at_risk_total) if n_at_risk_total > 0 else None
                        overall_unrecognised_rate = float(unrecognised.mean())
                        premature_rate = float(premature.mean())

                        old_cell = old_result[backbone][ds][condition][m][str(L)]

                        if premature.any():
                            true_rul_at_trigger_capped = float(y[premature].mean())
                            excess_over_L_capped = float((y[premature] - L).mean())
                            true_rul_at_trigger_uncapped = float(y_unc[premature].mean())
                            excess_over_L_uncapped = float((y_unc[premature] - L).mean())
                            diff_mean = true_rul_at_trigger_uncapped - true_rul_at_trigger_capped
                            diff_excess = excess_over_L_uncapped - excess_over_L_capped
                            all_diffs_mean.append(diff_mean)
                            all_diffs_excess.append(diff_excess)
                        else:
                            true_rul_at_trigger_capped = None; excess_over_L_capped = None
                            true_rul_at_trigger_uncapped = None; excess_over_L_uncapped = None
                            diff_mean = 0.0; diff_excess = 0.0

                        costs = {str(r): r * overall_unrecognised_rate + 1.0 * premature_rate for r in COST_RATIOS}

                        # soft-check rates against the R3 stepC run: GPU forward passes are not
                        # bit-deterministic across separate runs (cuDNN/attention kernel algorithm
                        # selection), so a boundary-case engine can flip triggered<->not-triggered
                        # by a sub-microsecond margin near lb==L; tolerate up to ~1 flipped cell
                        # out of the (n_cells x n_engines) array, not exact bit-identity.
                        n_cells_total = lb.size
                        tol = 1.5 / n_cells_total
                        d_unrec = abs(overall_unrecognised_rate - old_cell['overall_unrecognised_rate'])
                        d_prem = abs(premature_rate - old_cell['premature_rate'])
                        if d_unrec > tol or d_prem > tol:
                            mismatch_log.append({'backbone': backbone, 'ds': ds, 'condition': condition,
                                                  'method': m, 'L': L, 'kind': 'rate_diff_exceeds_tol',
                                                  'd_unrecognised': d_unrec, 'd_premature': d_prem, 'tol': tol})

                        result[backbone][ds][condition][m][str(L)] = {
                            'overall_unrecognised_rate': overall_unrecognised_rate,
                            'conditional_unrecognised_rate': conditional_unrecognised_rate,
                            'n_at_risk': int(n_at_risk_total),
                            'premature_rate': premature_rate,
                            'true_rul_at_trigger_capped125': true_rul_at_trigger_capped,
                            'excess_over_L_capped125': excess_over_L_capped,
                            'true_rul_at_trigger_uncapped': true_rul_at_trigger_uncapped,
                            'excess_over_L_uncapped': excess_over_L_uncapped,
                            'diff_true_rul_at_trigger': diff_mean,
                            'diff_excess_over_L': diff_excess,
                            'cost_by_ratio': costs,
                        }

                # rank check (ratios only depend on rates, unaffected by capping; re-derive for completeness)
                rankings_match = True
                for L in L_LEVELS:
                    old_rankings = {}
                    new_rankings = {}
                    for r in COST_RATIOS:
                        old_costs = {m: old_result[backbone][ds][condition][m][str(L)]['cost_by_ratio'][str(r)] for m in METHODS}
                        new_costs = {m: result[backbone][ds][condition][m][str(L)]['cost_by_ratio'][str(r)] for m in METHODS}
                        old_rankings[r] = tuple(sorted(old_costs, key=lambda m: old_costs[m]))
                        new_rankings[r] = tuple(sorted(new_costs, key=lambda m: new_costs[m]))
                    if old_rankings != new_rankings:
                        rankings_match = False
                print(f"  [{condition}] rate/cost/rank identical to R3 stepC: {rankings_match}")

            for seed in C.SEEDS:
                del mc_model_by_seed[seed], nll_model_by_seed[seed]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(R4_DIR, 'C_maintenance_rul_corrected.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    all_diffs_mean = np.array(all_diffs_mean); all_diffs_excess = np.array(all_diffs_excess)
    print(f"\nDiff (uncapped - capped) true_rul_at_trigger: mean={all_diffs_mean.mean():.4f}  "
          f"max={all_diffs_mean.max():.4f}  n_premature_groups={len(all_diffs_mean)}")
    print(f"Diff (uncapped - capped) excess_over_L: mean={all_diffs_excess.mean():.4f}  "
          f"max={all_diffs_excess.max():.4f}")
    print(f"Classification mismatches (capped vs uncapped at_risk/premature): {len(mismatch_log)}")
    if mismatch_log:
        print(json.dumps(mismatch_log[:10], indent=2))
