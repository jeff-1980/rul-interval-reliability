"""
FD003 completion, last step: assembles COST_TABLE_FD003_leakfree.csv,
with the same column structure as cost_table_lstm.py. Latency uses the
median pooled over all independent latency runs (not a single run's
value).
"""
import os
import json

import numpy as np
import pandas as pd
pd.set_option('future.infer_string', False)

import common as C
import mc_dropout_model as S1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
DS = 'FD003'

if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'stepFD003_nll_leakfree_results.json')) as f:
        nll_rows = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json')) as f:
        mc_mse_rows = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
        cp_rows = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'stepFD003_deep_ensemble_leakfree.json')) as f:
        ens = json.load(f)[DS]
    with open(os.path.join(RESULTS_DIR, 'stepFD003_per_engine_coverage_leakfree.json')) as f:
        peng = json.load(f)[DS]
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json')) as f:
        main_sweep = json.load(f)[DS]['B_pooled']
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
        armC_sweep = json.load(f)[DS]

    # ---- coverage_half_life_feat_oob: pool main(11 levels)+armC(5 levels), crossover@0.80 ----
    def picp_of(cell, method):
        return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']

    def crossover(pts, threshold=0.80):
        pts = sorted(pts, key=lambda p: p[0])
        for i in range(len(pts) - 1):
            fo0, p0 = pts[i]; fo1, p1 = pts[i + 1]
            if p0 >= threshold and p1 < threshold:
                frac = 0 if p1 == p0 else (threshold - p0) / (p1 - p0)
                return fo0 + frac * (fo1 - fo0)
        if pts and pts[0][1] < threshold:
            return pts[0][0]
        return None

    def half_life(method):
        pts = []
        for k, fo in main_sweep['feat_oob'].items():
            pts.append((fo, picp_of(main_sweep[method][k], method)))
        for k, fo in armC_sweep['feat_oob'].items():
            pts.append((fo, picp_of(armC_sweep[method][k], method)))
        co = crossover(pts)
        return co if co is not None else 'not_reached_in_measured_range'

    # ---- latency: pool all independent platform_batch512 runs found ----
    import glob
    lat_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'latency_FD003_platform_batch512_*.json')))
    assert lat_files, "run latency_lstm_fd003.py (>=2 independent runs) first"
    method_key_map = {'MSE (no UQ, fixed sigma)': 'MSE', 'Heteroscedastic NLL': 'NLL',
                       'MC Dropout (Kendall&Gal-corrected)': 'MC_Dropout_T50',
                       'Deep Ensemble (M=5, option A)': 'Deep_Ensemble_M5', 'Split-CP (norm)': 'CP_norm'}
    pooled_rounds = {k: [] for k in method_key_map.values()}
    n_runs = 0
    for lf in lat_files:
        with open(lf) as f:
            d = json.load(f)
        n_runs += 1
        for k in pooled_rounds:
            pooled_rounds[k].extend(d[k]['round_ms_raw'])
    pooled_median = {k: float(np.median(v)) for k, v in pooled_rounds.items()}
    print(f"Latency pooled across {n_runs} independent runs x11 rounds = {n_runs*11} rounds/method")
    nll_med = pooled_median['NLL']
    checks = {
        'CP_over_NLL_within_2pct': abs(pooled_median['CP_norm'] / nll_med - 1.0) < 0.02,
        'Ensemble_over_NLL_in_4.5_5.5': 4.5 <= pooled_median['Deep_Ensemble_M5'] / nll_med <= 5.5,
        'MCDropout_over_NLL_in_45_55': 45 <= pooled_median['MC_Dropout_T50'] / nll_med <= 55,
        'MSE_over_NLL_within_5pct': abs(pooled_median['MSE'] / nll_med - 1.0) < 0.05,
    }
    n_pass = sum(checks.values())
    print(f"Pooled latency hard-criteria: {n_pass}/4 pass ({checks})")

    def lat_str(method_label):
        key = method_key_map[method_label]
        per_smp = pooled_median[key] / 512.0
        return (f"{per_smp:.6f} (platform batch=512, tiled real test windows, pooled median across "
                f"{n_runs} independent full-protocol reruns x11 rounds={n_runs*11} rounds; "
                f"{n_pass}/4 hard acceptance ratios pass on pooled median; see latency_FD003_platform_batch512_*.json)")

    rmse_arr = np.array([r['rmse'] for r in nll_rows]); picp_arr = np.array([r['picp'] for r in nll_rows])
    mpiw_arr = np.array([r['mpiw'] for r in nll_rows]); ece_arr = np.array([r['ece'] for r in nll_rows])

    mc_picp = np.array([r['T50']['kendall_gal_full']['picp'] for r in mc_mse_rows])
    mc_mpiw = np.array([r['T50']['kendall_gal_full']['mpiw'] for r in mc_mse_rows])
    mc_ece = np.array([r['T50']['kendall_gal_full']['ece'] for r in mc_mse_rows])

    mse_picp = np.array([r['mse_row']['picp'] for r in mc_mse_rows])
    mse_mpiw = np.array([r['mse_row']['mpiw'] for r in mc_mse_rows])
    mse_ece = np.array([r['mse_row']['ece'] for r in mc_mse_rows])

    cp_picp = np.array([r['cp_norm']['picp'] for r in cp_rows])
    cp_mpiw = np.array([r['cp_norm']['mpiw'] for r in cp_rows])
    cp_ece = np.array([r['cp_norm']['ece'] for r in cp_rows])

    params_nll_cp = 53890  # same input_dim(14)/hidden(64) as FD001 -> identical param count
    params_mse_mc = 53825
    params_ens = 5 * params_nll_cp

    rows = [
        {'method': 'MSE (no UQ, fixed sigma)', 'forward_passes': 1,
         'picp_mean': float(mse_picp.mean()), 'picp_std': float(mse_picp.std(ddof=1)),
         'mpiw_mean': float(mse_mpiw.mean()), 'mpiw_std': float(mse_mpiw.std(ddof=1)),
         'ece_mean': float(mse_ece.mean()), 'ece_std': float(mse_ece.std(ddof=1)),
         'per_engine_compliance': peng['MSE_fixed']['compliance_rate_ge_090'],
         'latency_ms_per_sample': lat_str('MSE (no UQ, fixed sigma)'), 'params': params_mse_mc,
         'coverage_half_life_feat_oob': half_life('MSE_fixed')},
        {'method': 'Heteroscedastic NLL', 'forward_passes': 1,
         'picp_mean': float(picp_arr.mean()), 'picp_std': float(picp_arr.std(ddof=1)),
         'mpiw_mean': float(mpiw_arr.mean()), 'mpiw_std': float(mpiw_arr.std(ddof=1)),
         'ece_mean': float(ece_arr.mean()), 'ece_std': float(ece_arr.std(ddof=1)),
         'per_engine_compliance': peng['NLL']['compliance_rate_ge_090'],
         'latency_ms_per_sample': lat_str('Heteroscedastic NLL'), 'params': params_nll_cp,
         'coverage_half_life_feat_oob': half_life('NLL')},
        {'method': 'MC Dropout (Kendall&Gal-corrected)', 'forward_passes': 'T=50',
         'picp_mean': float(mc_picp.mean()), 'picp_std': float(mc_picp.std(ddof=1)),
         'mpiw_mean': float(mc_mpiw.mean()), 'mpiw_std': float(mc_mpiw.std(ddof=1)),
         'ece_mean': float(mc_ece.mean()), 'ece_std': float(mc_ece.std(ddof=1)),
         'per_engine_compliance': peng['MC_Dropout_fixed']['compliance_rate_ge_090'],
         'latency_ms_per_sample': lat_str('MC Dropout (Kendall&Gal-corrected)'), 'params': params_mse_mc,
         'coverage_half_life_feat_oob': half_life('MC_Dropout_fixed')},
        {'method': 'Deep Ensemble (M=5, option A)', 'forward_passes': 'M=5',
         'picp_mean': ens['picp'], 'picp_std': 'n/a (single ensemble)',
         'mpiw_mean': ens['mpiw'], 'mpiw_std': 'n/a (single ensemble)',
         'ece_mean': ens['ece'], 'ece_std': 'n/a (single ensemble)',
         'per_engine_compliance': peng['Deep_Ensemble']['compliance_rate_ge_090'],
         'latency_ms_per_sample': lat_str('Deep Ensemble (M=5, option A)'), 'params': params_ens,
         'coverage_half_life_feat_oob': half_life('Deep_Ensemble')},
        {'method': 'Split-CP (norm)', 'forward_passes': '1 + calibration',
         'picp_mean': float(cp_picp.mean()), 'picp_std': float(cp_picp.std(ddof=1)),
         'mpiw_mean': float(cp_mpiw.mean()), 'mpiw_std': float(cp_mpiw.std(ddof=1)),
         'ece_mean': float(cp_ece.mean()), 'ece_std': float(cp_ece.std(ddof=1)),
         'per_engine_compliance': peng['CP_norm']['compliance_rate_ge_090'],
         'latency_ms_per_sample': lat_str('Split-CP (norm)'), 'params': params_nll_cp,
         'coverage_half_life_feat_oob': half_life('CP_norm')},
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'COST_TABLE_FD003_leakfree.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(df[['method', 'picp_mean', 'mpiw_mean', 'ece_mean', 'per_engine_compliance', 'coverage_half_life_feat_oob']].to_string(index=False))
    print("\nFD003 cost table complete. FD003 now covered in all tables.")
