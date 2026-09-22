"""
R8（发布仓库导出前置任务）：重建 leakfree_r2/table2_clean_full.json 的生成脚本
——排查发现现有代码库没有任何脚本产出这个文件。这个文件本身不含任何新计算，
是把已有的、各自独立算出的分方法结果文件按 (backbone, dataset, method) 合并
成 Table II 用的统一形状：
  - MSE_fixed / MC_Dropout_fixed：直接取 leakfree_r2/fair_calibration_main_table.json
    （这两个机制的 sigma 来自校准集残差方差，"公平校准"修复后的口径，picp_mean/
    mpiw_mean/ece_mean/per_engine_compliance/interval_score_mean/wis_mean 六个
    字段已经是这张表要的形状，不需要再算）。
  - NLL / CP_norm：sigma 来自模型自己的方差头/conformal校准，不受"公平校准"
    修复影响，picp/mpiw/ece 取5个seed的均值（main.tex Table tab:leak 脚注：
    "Table tab:clean reports the mean of per-seed ECE"），per_engine 取
    per_engine_coverage_leakfree.json 的 compliance_rate_ge_090，is/wis 取
    interval_score_wis_clean.json。
  - Deep_Ensemble：本身就是单点估计（不是5个seed各自ensemble再平均），直接取用。
"""
import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']


def _load(rel):
    with open(os.path.join(RESULTS_DIR, rel)) as f:
        return json.load(f)


def mean_of(values, *keys):
    return {k: float(np.mean([v[k] for v in values])) for k in keys}


def nll_cp_ens(backbone, ds):
    if backbone == 'LSTM' and ds == 'FD003':
        nll_list = _load('stepFD003_nll_leakfree_results.json')
        cp_list = _load('stepFD003_splitcp_leakfree_results.json')
        ens = _load('stepFD003_deep_ensemble_leakfree.json')['FD003']
    elif backbone == 'LSTM':
        nll_list = _load('step0c_leakfree_results.json')[ds]
        cp_list = _load('split_cp_leakfree.json')[ds]
        ens = _load('deep_ensemble_leakfree.json')[ds]
    else:
        nll_list = _load('leakfree_t2/t2_transformer_nll_leakfree_results.json')[ds]
        cp_list = _load('leakfree_t2/t2_transformer_splitcp_leakfree_results.json')[ds]
        ens = _load('leakfree_t2/t2_transformer_ensemble_clean_leakfree.json')[ds]

    nll_agg = mean_of(nll_list, 'picp', 'mpiw', 'ece')
    cp_agg = mean_of([r['cp_norm'] for r in cp_list], 'picp', 'mpiw', 'ece')
    ens_agg = {'picp': ens['picp'], 'mpiw': ens['mpiw'], 'ece': ens['ece']}
    return {'NLL': nll_agg, 'CP_norm': cp_agg, 'Deep_Ensemble': ens_agg}


if __name__ == '__main__':
    fair = _load('leakfree_r2/fair_calibration_main_table.json')
    per_engine_lstm = _load('per_engine_coverage_leakfree.json')
    per_engine_lstm_fd003 = _load('stepFD003_per_engine_coverage_leakfree.json')
    per_engine_t2 = _load('leakfree_t2/t2_transformer_per_engine_coverage_leakfree.json')
    iswis = _load('leakfree_r2/interval_score_wis_clean.json')

    result = {'LSTM': {}, 'Transformer': {}}
    for backbone in ['LSTM', 'Transformer']:
        for ds in DATASETS:
            other = nll_cp_ens(backbone, ds)
            pe = (per_engine_lstm_fd003['FD003'] if (backbone == 'LSTM' and ds == 'FD003')
                  else (per_engine_lstm[ds] if backbone == 'LSTM' else per_engine_t2[ds]))

            result[backbone].setdefault(ds, {})
            for m in ['MSE_fixed', 'MC_Dropout_fixed']:
                f = fair[backbone][ds][m]
                result[backbone][ds][m] = {
                    'picp': f['picp_mean'], 'mpiw': f['mpiw_mean'], 'ece': f['ece_mean'],
                    'per_engine': f['per_engine_compliance'],
                    'is': f['interval_score_mean'], 'wis': f['wis_mean'],
                }
            for m in ['NLL', 'CP_norm', 'Deep_Ensemble']:
                result[backbone][ds][m] = {
                    **other[m],
                    'per_engine': pe[m]['compliance_rate_ge_090'],
                    'is': iswis[backbone][ds][m]['interval_score_mean'],
                    'wis': iswis[backbone][ds][m]['wis_mean'],
                }

    out_path = os.path.join(R2_DIR, 'table2_clean_full.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"Saved -> {out_path}")

    with open(os.path.join(R2_DIR, 'table2_clean_full.json')) as f:
        existing = json.load(f)

    print("\n=== comparison against existing table2_clean_full.json (tol 1e-6) ===")
    all_match = True
    for backbone in ['LSTM', 'Transformer']:
        for ds in DATASETS:
            for m in METHODS:
                for field in ['picp', 'mpiw', 'ece', 'per_engine', 'is', 'wis']:
                    a = existing[backbone][ds][m][field]
                    b = result[backbone][ds][m][field]
                    diff = abs(a - b)
                    if diff >= 1e-6:
                        all_match = False
                        print(f"  MISMATCH {backbone}/{ds}/{m}/{field}: existing={a} recomputed={b} diff={diff:.2e}")
    print(f"\nALL MATCH (tol 1e-6): {all_match}")
