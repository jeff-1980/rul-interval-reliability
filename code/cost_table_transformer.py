"""
T2-A8：组装 COST_TABLE_{DS}_Transformer_leakfree.csv，与
cost_table_lstm_fd003.py 同一列结构，4 数据集。
半衰(abs/rel)取自 mechanism_transformer.py 产出的
t2_transformer_half_life_feat_oob.json（臂C，5档，见该脚本顶部关于范围
选择的说明——不是主SNR臂+臂C两者合并，与 LSTM 侧口径不完全相同，如实
在 NOTES 里记录）。延迟列先留空/占位，由 latency_transformer.py 统一批量
重测后单独回填（10行同会话协议）。
"""
import os
import json

import numpy as np
import pandas as pd
pd.set_option('future.infer_string', False)

import common as C
import transformer_common as T2
import sweep_engine as E

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']

if __name__ == '__main__':
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_nll_leakfree_results.json')) as f:
        nll_all = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json')) as f:
        mc_mse_all = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')) as f:
        cp_all = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_ensemble_clean_leakfree.json')) as f:
        ens_all = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_per_engine_coverage_leakfree.json')) as f:
        peng_all = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_half_life_feat_oob.json')) as f:
        halflife_all = json.load(f)

    for ds in DATASETS:
        nll_rows = nll_all[ds]
        mc_mse_rows = mc_mse_all[ds]
        cp_rows = cp_all[ds]
        ens = ens_all[ds]
        peng = peng_all[ds]
        halflife = halflife_all[ds]

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

        input_dim = nll_rows[0].get('input_dim')  # may be absent; recompute if needed below
        ckpt = __import__('torch').load(T2.nll_ckpt_path('Transformer', ds, C.SEEDS[0]), map_location='cpu', weights_only=False)
        params_nll_cp = sum(p.numel() for p in
                             T2.HeteroscedasticTransformer(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout'],
                                                            T2.SEQUENCE_LENGTH, ckpt['log_sigma_min'],
                                                            ckpt['log_sigma_max']).parameters())
        ckpt_mc = __import__('torch').load(T2.mc_ckpt_path('Transformer', ds, C.SEEDS[0]), map_location='cpu', weights_only=False)
        params_mse_mc = sum(p.numel() for p in
                             T2.MC_Transformer(ckpt_mc['input_dim'], ckpt_mc['hidden_dim'], ckpt_mc['dropout'],
                                                T2.SEQUENCE_LENGTH).parameters())
        params_ens = 5 * params_nll_cp

        def hl(method, key):
            v = halflife[method][key]
            return v if v is not None else 'not_reached_in_measured_range'

        rows = [
            {'method': 'MSE (no UQ, fixed sigma)', 'forward_passes': 1,
             'picp_mean': float(mse_picp.mean()), 'picp_std': float(mse_picp.std(ddof=1)),
             'mpiw_mean': float(mse_mpiw.mean()), 'mpiw_std': float(mse_mpiw.std(ddof=1)),
             'ece_mean': float(mse_ece.mean()), 'ece_std': float(mse_ece.std(ddof=1)),
             'per_engine_compliance': peng['MSE_fixed']['compliance_rate_ge_090'],
             'latency_ms_per_sample': 'TBD_see_latency_step', 'params': params_mse_mc,
             'coverage_half_life_feat_oob_abs': hl('MSE_fixed', 'abs_co'),
             'coverage_half_life_feat_oob_rel': hl('MSE_fixed', 'rel_co')},
            {'method': 'Heteroscedastic NLL', 'forward_passes': 1,
             'picp_mean': float(picp_arr.mean()), 'picp_std': float(picp_arr.std(ddof=1)),
             'mpiw_mean': float(mpiw_arr.mean()), 'mpiw_std': float(mpiw_arr.std(ddof=1)),
             'ece_mean': float(ece_arr.mean()), 'ece_std': float(ece_arr.std(ddof=1)),
             'per_engine_compliance': peng['NLL']['compliance_rate_ge_090'],
             'latency_ms_per_sample': 'TBD_see_latency_step', 'params': params_nll_cp,
             'coverage_half_life_feat_oob_abs': hl('NLL', 'abs_co'),
             'coverage_half_life_feat_oob_rel': hl('NLL', 'rel_co')},
            {'method': 'MC Dropout (Kendall&Gal-corrected)', 'forward_passes': 'T=50',
             'picp_mean': float(mc_picp.mean()), 'picp_std': float(mc_picp.std(ddof=1)),
             'mpiw_mean': float(mc_mpiw.mean()), 'mpiw_std': float(mc_mpiw.std(ddof=1)),
             'ece_mean': float(mc_ece.mean()), 'ece_std': float(mc_ece.std(ddof=1)),
             'per_engine_compliance': peng['MC_Dropout_fixed']['compliance_rate_ge_090'],
             'latency_ms_per_sample': 'TBD_see_latency_step', 'params': params_mse_mc,
             'coverage_half_life_feat_oob_abs': hl('MC_Dropout_fixed', 'abs_co'),
             'coverage_half_life_feat_oob_rel': hl('MC_Dropout_fixed', 'rel_co')},
            {'method': 'Deep Ensemble (M=5, option A)', 'forward_passes': 'M=5',
             'picp_mean': ens['picp'], 'picp_std': 'n/a (single ensemble)',
             'mpiw_mean': ens['mpiw'], 'mpiw_std': 'n/a (single ensemble)',
             'ece_mean': ens['ece'], 'ece_std': 'n/a (single ensemble)',
             'per_engine_compliance': peng['Deep_Ensemble']['compliance_rate_ge_090'],
             'latency_ms_per_sample': 'TBD_see_latency_step', 'params': params_ens,
             'coverage_half_life_feat_oob_abs': hl('Deep_Ensemble', 'abs_co'),
             'coverage_half_life_feat_oob_rel': hl('Deep_Ensemble', 'rel_co')},
            {'method': 'Split-CP (norm)', 'forward_passes': '1 + calibration',
             'picp_mean': float(cp_picp.mean()), 'picp_std': float(cp_picp.std(ddof=1)),
             'mpiw_mean': float(cp_mpiw.mean()), 'mpiw_std': float(cp_mpiw.std(ddof=1)),
             'ece_mean': float(cp_ece.mean()), 'ece_std': float(cp_ece.std(ddof=1)),
             'per_engine_compliance': peng['CP_norm']['compliance_rate_ge_090'],
             'latency_ms_per_sample': 'TBD_see_latency_step', 'params': params_nll_cp,
             'coverage_half_life_feat_oob_abs': hl('CP_norm', 'abs_co'),
             'coverage_half_life_feat_oob_rel': hl('CP_norm', 'rel_co')},
        ]
        df = pd.DataFrame(rows)
        out_path = os.path.join(T2.TRANSFORMER_DIR, f'COST_TABLE_{ds}_Transformer_leakfree.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
        print(df[['method', 'picp_mean', 'mpiw_mean', 'ece_mean', 'per_engine_compliance',
                   'coverage_half_life_feat_oob_abs', 'coverage_half_life_feat_oob_rel']].to_string(index=False))
        print()

    print("T2-A8 (Transformer cost tables, latency TBD) complete.")
