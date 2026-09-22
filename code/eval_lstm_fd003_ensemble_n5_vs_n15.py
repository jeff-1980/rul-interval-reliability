"""
FD003 补全：n=5 vs n=15 种子间 PICP/RMSE/MPIW 方差对比，leakfree协议。
逐字复用 eval_lstm_ensemble_n5_vs_n15.py 的统计口径，仅数据源
换成 FD003 专用文件（stepFD003_nll_leakfree_results.json /
stepFD003_extra_seeds_leakfree_results.json）。
"""
import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')

DS = 'FD003'

if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'stepFD003_nll_leakfree_results.json')) as f:
        n5_rows = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'stepFD003_extra_seeds_leakfree_results.json')) as f:
        extra10_rows = json.load(f)

    n15_rows = n5_rows + extra10_rows
    result = {}
    for metric in ['rmse', 'picp', 'mpiw']:
        n5_vals = np.array([r[metric] for r in n5_rows])
        n15_vals = np.array([r[metric] for r in n15_rows])
        n5_mean, n5_std = float(n5_vals.mean()), float(n5_vals.std(ddof=1))
        n15_mean, n15_std = float(n15_vals.mean()), float(n15_vals.std(ddof=1))
        result[metric] = {
            'n5_mean': n5_mean, 'n5_std': n5_std,
            'n15_mean': n15_mean, 'n15_std': n15_std,
            'std_ratio_15_over_5': n15_std / n5_std if n5_std > 0 else None,
        }
        print(f"{DS} {metric}: n5={n5_mean:.4f}+/-{n5_std:.4f}  n15={n15_mean:.4f}+/-{n15_std:.4f}  "
              f"ratio={result[metric]['std_ratio_15_over_5']:.3f}")

    out_path = os.path.join(LEAKFREE_DIR, 'FD003_n5_vs_n15_variance_comparison_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump({DS: result}, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP FD003-2e complete.")
