"""
STEP 2e：n=5 vs n=15 种子间方差对比，leakfree协议。

同一统计口径的旧版（非leakfree）：对每个数据集×指标(rmse/picp/mpiw)，
比较原始5 seeds的均值/std 与 全部15 seeds（5原始+10额外）的均值/std，
报告 std_ratio_15_over_5。数据源：`step0c_leakfree_results.json`（5-seed，
官方测试集单次评价）+ `step0d_extra_seeds_leakfree_results.json`（额外10
seed，同协议）。
"""
import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')

DATASETS = ['FD001', 'FD002', 'FD004']

if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')) as f:
        n5 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'step0d_extra_seeds_leakfree_results.json')) as f:
        extra10 = json.load(f)

    results = {}
    for ds in DATASETS:
        results[ds] = {}
        n15_rows = n5[ds] + extra10[ds]
        for metric in ['rmse', 'picp', 'mpiw']:
            n5_vals = np.array([r[metric] for r in n5[ds]])
            n15_vals = np.array([r[metric] for r in n15_rows])
            n5_mean, n5_std = float(n5_vals.mean()), float(n5_vals.std(ddof=1))
            n15_mean, n15_std = float(n15_vals.mean()), float(n15_vals.std(ddof=1))
            results[ds][metric] = {
                'n5_mean': n5_mean, 'n5_std': n5_std,
                'n15_mean': n15_mean, 'n15_std': n15_std,
                'std_ratio_15_over_5': n15_std / n5_std if n5_std > 0 else None,
            }
            print(f"{ds} {metric}: n5={n5_mean:.4f}±{n5_std:.4f}  n15={n15_mean:.4f}±{n15_std:.4f}  "
                  f"ratio={results[ds][metric]['std_ratio_15_over_5']:.3f}")

    out_path = os.path.join(LEAKFREE_DIR, 'n5_vs_n15_variance_comparison_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP2e complete.")
