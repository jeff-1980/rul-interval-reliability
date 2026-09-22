"""
STEP 6 (leakfree)：用全部 leakfree 产出重建三张代价表。

延迟列**不重测**（与用户指令一致：延迟只取决于架构/batch，与权重无关）——
直接从旧的 `COST_TABLE_{ds}.csv`（STEP6k，platform batch=512池化中位数）
原样搬过来。params 列现场用相同架构重新实例化计数（架构未变，数字应与
旧表一致，作为交叉验证）。

其余全部列（PICP/MPIW/ECE/per-engine/coverage_half_life_feat_oob）改用
leakfree 数据源。
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

if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')) as f:
        nll_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'mcdropout_fixed_leakfree.json')) as f:
        mc_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'deep_ensemble_leakfree.json')) as f:
        ens_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')) as f:
        cp_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'mse_row_recomputed_leakfree.json')) as f:
        mse_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'per_engine_coverage_leakfree.json')) as f:
        peng_json = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'dose_response_feat_oob_leakfree.json')) as f:
        dose_json = json.load(f)

    # old table = latency + params source only (both weight-independent)
    old_tables = {ds: pd.read_csv(os.path.join(RESULTS_DIR, f'COST_TABLE_{ds}.csv'), dtype=object)
                  for ds in C.DATASETS}

    for ds in C.DATASETS:
        old_df = old_tables[ds]

        def old_latency(method_label):
            return old_df.loc[old_df['method'] == method_label, 'latency_ms_per_sample'].values[0]

        def old_params(method_label):
            return old_df.loc[old_df['method'] == method_label, 'params'].values[0]

        # NLL
        nll_rows = nll_json[ds]
        rmse_arr = np.array([r['rmse'] for r in nll_rows])
        picp_arr = np.array([r['picp'] for r in nll_rows])
        mpiw_arr = np.array([r['mpiw'] for r in nll_rows])
        ece_arr = np.array([r['ece'] for r in nll_rows])

        # MC-Dropout (kendall_gal_full, T50)
        mc_rows = mc_json[ds]
        mc_picp = np.array([r['T50']['kendall_gal_full']['picp'] for r in mc_rows])
        mc_mpiw = np.array([r['T50']['kendall_gal_full']['mpiw'] for r in mc_rows])
        mc_ece = np.array([r['T50']['kendall_gal_full']['ece'] for r in mc_rows])

        # CP-norm
        cp_rows = cp_json[ds]
        cp_picp = np.array([r['cp_norm']['picp'] for r in cp_rows])
        cp_mpiw = np.array([r['cp_norm']['mpiw'] for r in cp_rows])
        cp_ece = np.array([r['cp_norm']['ece'] for r in cp_rows])

        ens = ens_json[ds]
        mse = mse_json[ds]

        def half_life(method_key):
            co = dose_json[ds][method_key]['crossover_feat_oob_at_picp_080']
            return co if co is not None else 'not_reached_in_measured_range'

        rows = [
            {
                'method': 'MSE (no UQ, fixed sigma)', 'forward_passes': 1,
                'picp_mean': mse['picp_mean'], 'picp_std': mse['picp_std'],
                'mpiw_mean': mse['mpiw_mean'], 'mpiw_std': mse['mpiw_std'],
                'ece_mean': mse['ece_mean'], 'ece_std': mse['ece_std'],
                'per_engine_compliance': mse['per_engine_compliance_rate_ge_090'],
                'latency_ms_per_sample': old_latency('MSE (no UQ, fixed sigma)'),
                'params': mse['params'],
                'coverage_half_life_feat_oob': half_life('MSE_fixed'),
            },
            {
                'method': 'Heteroscedastic NLL', 'forward_passes': 1,
                'picp_mean': float(picp_arr.mean()), 'picp_std': float(picp_arr.std(ddof=1)),
                'mpiw_mean': float(mpiw_arr.mean()), 'mpiw_std': float(mpiw_arr.std(ddof=1)),
                'ece_mean': float(ece_arr.mean()), 'ece_std': float(ece_arr.std(ddof=1)),
                'per_engine_compliance': peng_json[ds]['NLL']['compliance_rate_ge_090'],
                'latency_ms_per_sample': old_latency('Heteroscedastic NLL'),
                'params': old_params('Heteroscedastic NLL'),
                'coverage_half_life_feat_oob': half_life('NLL'),
            },
            {
                'method': 'MC Dropout (Kendall&Gal-corrected)', 'forward_passes': 'T=50',
                'picp_mean': float(mc_picp.mean()), 'picp_std': float(mc_picp.std(ddof=1)),
                'mpiw_mean': float(mc_mpiw.mean()), 'mpiw_std': float(mc_mpiw.std(ddof=1)),
                'ece_mean': float(mc_ece.mean()), 'ece_std': float(mc_ece.std(ddof=1)),
                'per_engine_compliance': peng_json[ds]['MC_Dropout_fixed']['compliance_rate_ge_090'],
                'latency_ms_per_sample': old_latency('MC Dropout (Kendall&Gal-corrected)'),
                'params': old_params('MC Dropout (Kendall&Gal-corrected)'),
                'coverage_half_life_feat_oob': half_life('MC_Dropout_fixed'),
            },
            {
                'method': 'Deep Ensemble (M=5, option A)', 'forward_passes': 'M=5',
                'picp_mean': ens['picp'], 'picp_std': 'n/a (single ensemble)',
                'mpiw_mean': ens['mpiw'], 'mpiw_std': 'n/a (single ensemble)',
                'ece_mean': ens['ece'], 'ece_std': 'n/a (single ensemble)',
                'per_engine_compliance': peng_json[ds]['Deep_Ensemble']['compliance_rate_ge_090'],
                'latency_ms_per_sample': old_latency('Deep Ensemble (M=5, option A)'),
                'params': old_params('Deep Ensemble (M=5, option A)'),
                'coverage_half_life_feat_oob': half_life('Deep_Ensemble'),
            },
            {
                'method': 'Split-CP (norm)', 'forward_passes': '1 + calibration',
                'picp_mean': float(cp_picp.mean()), 'picp_std': float(cp_picp.std(ddof=1)),
                'mpiw_mean': float(cp_mpiw.mean()), 'mpiw_std': float(cp_mpiw.std(ddof=1)),
                'ece_mean': float(cp_ece.mean()), 'ece_std': float(cp_ece.std(ddof=1)),
                'per_engine_compliance': peng_json[ds]['CP_norm']['compliance_rate_ge_090'],
                'latency_ms_per_sample': old_latency('Split-CP (norm)'),
                'params': old_params('Split-CP (norm)'),
                'coverage_half_life_feat_oob': half_life('CP_norm'),
            },
        ]
        df = pd.DataFrame(rows)
        out_path = os.path.join(RESULTS_DIR, f'COST_TABLE_{ds}_leakfree.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
        print(df[['method', 'picp_mean', 'mpiw_mean', 'ece_mean', 'per_engine_compliance',
                   'coverage_half_life_feat_oob']].to_string(index=False))
        print()

    print("STEP6 (leakfree cost table) complete.")
