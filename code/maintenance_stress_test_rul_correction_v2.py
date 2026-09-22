"""
R4-2b：未截断 RUL 修正重做，底本固定为 leakfree_r3/C_maintenance_full_onesided.json
（单侧95%规则），要求 premature_rate 与底本逐行一致，不再只是"容差内"。

上一版（maintenance_stress_test_rul_correction_v1.py）重新推理时，NLL/CP-norm/
MSE-fixed/Deep-Ensemble 四种机制的 premature_rate 出现了最多 0.0044 的偏差
——排查后确认原因是 PyTorch/cuDNN 默认允许非确定性算法选择（LSTM/Transformer
的 cuDNN kernel 在多次调用间可能选择不同的规约顺序），不是逻辑错误。本版在
装载模型之前设置 `torch.backends.cudnn.deterministic=True` +
`torch.backends.cudnn.benchmark=False`（以及 `torch.use_deterministic_algorithms
(True, warn_only=True)`），消除这四种机制的推理路径里的非确定性来源——它们
的前向传播在 eval() 模式下没有任何依赖运行时随机数的步骤，理论上应能与底本
逐比特一致。

**MC_Dropout_fixed 是唯一例外，如实说明**：`sweep_engine.infer_mc_dropout`
用 `mc_model.train()` 做 T=50 次真实的随机 dropout 采样，而
`maintenance_decision_one_sided.py`（底本的生成脚本）全文没有在调用它之前
设置过 `torch.manual_seed(...)`——底本那次跑出来的 dropout 采样结果，用的是
当时进程里未被记录、事后也无法回放的全局 RNG 状态。这意味着 MC_Dropout_fixed
这一档的 premature_rate**原则上不可能被逐比特复现**，不是这一版脚本能力不够，
是底本本身没有留下可回放的随机性凭证。本版对 MC_Dropout_fixed 改为在每次
调用前用确定性标签播种（保证本脚本自己可重复），premature_rate 会保留与
底本的小幅差异，并在输出里明确标注这一点，不假装它也逐行一致。

只做推理，不重训，不改 main.tex。输出：
results/generated/leakfree_r4/C_maintenance_rul_corrected_v2.json
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
DETERMINISTIC_METHODS = {'NLL', 'MSE_fixed', 'Deep_Ensemble', 'CP_norm'}
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
    X_list, y_list, y_uncapped_list, u_list = [], [], [], []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        if mode == 'test':
            X_list.append(unit_data[-C.SEQUENCE_LENGTH:])
            raw_val = true_ruls.iloc[unit - 1].item() - 1  # R8-B2, see common.create_sequences
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  cudnn.deterministic={torch.backends.cudnn.deterministic}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    with open(os.path.join(R3_DIR, 'C_maintenance_full_onesided.json')) as f:
        old_result = json.load(f)

    result = {}
    all_diffs_mean, all_diffs_excess = [], []
    exact_match_log = []  # per (backbone,ds,condition,method,L): bool exact match to baseline

    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw_probe, _, _, feat_cols_probe, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw_probe, feat_cols_probe)
            full_scale = V4.sensor_only_scale(feat_cols_probe, full_scale)  # R8-B1

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
                        # MC-Dropout: deterministic-but-not-baseline-matching seed (see module docstring)
                        torch.manual_seed(C.stable_seed(ds, backbone, condition, trial, seed, 'mc_dropout'))
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

                        at_risk_unc = y_unc <= L
                        premature_unc = triggered & (y_unc > L + 20)
                        assert np.array_equal(at_risk, at_risk_unc), \
                            f"capping flipped at_risk classification: {backbone}/{ds}/{condition}/{m}/L={L}"
                        assert np.array_equal(premature, premature_unc), \
                            f"capping flipped premature classification: {backbone}/{ds}/{condition}/{m}/L={L}"

                        n_at_risk_total = at_risk.sum()
                        conditional_unrecognised_rate = float(unrecognised.sum() / n_at_risk_total) if n_at_risk_total > 0 else None
                        overall_unrecognised_rate = float(unrecognised.mean())
                        premature_rate = float(premature.mean())

                        old_cell = old_result[backbone][ds][condition][m][str(L)]
                        exact = (overall_unrecognised_rate == old_cell['overall_unrecognised_rate'] and
                                 premature_rate == old_cell['premature_rate'])
                        exact_match_log.append({'backbone': backbone, 'ds': ds, 'condition': condition,
                                                 'method': m, 'L': L, 'exact_match_to_baseline': exact,
                                                 'is_deterministic_method': m in DETERMINISTIC_METHODS,
                                                 'd_premature': premature_rate - old_cell['premature_rate']})

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

                        result[backbone][ds][condition][m][str(L)] = {
                            'overall_unrecognised_rate': overall_unrecognised_rate,
                            'conditional_unrecognised_rate': conditional_unrecognised_rate,
                            'n_at_risk': int(n_at_risk_total),
                            'premature_rate': premature_rate,
                            'exact_match_to_baseline_rates': exact,
                            'true_rul_at_trigger_capped125': true_rul_at_trigger_capped,
                            'excess_over_L_capped125': excess_over_L_capped,
                            'true_rul_at_trigger_uncapped': true_rul_at_trigger_uncapped,
                            'excess_over_L_uncapped': excess_over_L_uncapped,
                            'diff_true_rul_at_trigger': diff_mean,
                            'diff_excess_over_L': diff_excess,
                            'cost_by_ratio': costs,
                        }

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
                print(f"  [{condition}] rate/cost/rank identical to baseline: {rankings_match}")

            for seed in C.SEEDS:
                del mc_model_by_seed[seed], nll_model_by_seed[seed]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(R4_DIR, 'C_maintenance_rul_corrected_v2.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    all_diffs_mean = np.array(all_diffs_mean); all_diffs_excess = np.array(all_diffs_excess)
    print(f"\nDiff (uncapped - capped) true_rul_at_trigger: mean={all_diffs_mean.mean():.4f}  "
          f"max={all_diffs_mean.max():.4f}  n_premature_groups={len(all_diffs_mean)}")
    print(f"Diff (uncapped - capped) excess_over_L: mean={all_diffs_excess.mean():.4f}  "
          f"max={all_diffs_excess.max():.4f}")

    det_rows = [r for r in exact_match_log if r['is_deterministic_method']]
    nondet_rows = [r for r in exact_match_log if not r['is_deterministic_method']]
    det_exact = sum(1 for r in det_rows if r['exact_match_to_baseline'])
    nondet_exact = sum(1 for r in nondet_rows if r['exact_match_to_baseline'])
    print(f"\nDeterministic methods (NLL/MSE_fixed/Deep_Ensemble/CP_norm), "
          f"{len(det_rows)} rows: exact match to baseline = {det_exact}/{len(det_rows)}")
    print(f"MC_Dropout_fixed (inherently non-reproducible, no seed logged in baseline run), "
          f"{len(nondet_rows)} rows: exact match to baseline = {nondet_exact}/{len(nondet_rows)}")
    if det_exact < len(det_rows):
        mism = [r for r in det_rows if not r['exact_match_to_baseline']]
        print(f"Deterministic-method mismatches (should be empty): {json.dumps(mism[:10], indent=2)}")

    with open(os.path.join(R4_DIR, 'exact_match_log.json'), 'w') as fp:
        json.dump(exact_match_log, fp, indent=2, default=float)
