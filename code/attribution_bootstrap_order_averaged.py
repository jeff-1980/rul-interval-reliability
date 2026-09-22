"""
R6-1（纯后处理，无新增推理，不需要 PYTHONHASHSEED）：

(a) G/H 归因 JSON 的 interaction 字段改号：新定义 = (C11-C10)-(C01-C00)
    = orderA_sigma_effect - orderB_sigma_effect，与旧定义
    orderA_mu_effect - orderB_mu_effect 互为相反数（代数恒等：
    total_delta = orderA_mu+orderA_sigma = orderB_mu+orderB_sigma，
    所以 orderA_mu-orderB_mu == -(orderA_sigma-orderB_sigma)），直接
    从已存的 C00/C10/C01/C11 重新算，不是简单取负号了事。

(b) |Δμ|-|Δσ| 的跨种子配对差 + 95% 区间：用
    B_attribution_raw_per_seed.json 的逐 seed C00/C10/C01/C11
    （16 个 cell：LSTM 8 + Transformer 8）。逐 seed 算
    avg_mu = mean(orderA_mu, orderB_mu)，avg_sigma = mean(orderA_sigma,
    orderB_sigma)（与正文"mean effect/scale effect 是两次序均值"同一
    定义），再算 |avg_mu|-|avg_sigma|，对 5 个 seed 值做 bootstrap
    （n=2000，有放回重采样 5 个种子，report mean/95% CI）。

    **口径提醒**：B_attribution_raw_per_seed.json 里的 C00/C10/C01/C11
    是在"最近实测网格点"（grid_point_used，与 frozen_decomposition_2x2.json
    同一个点）上算的，不是 G/H 用的"精确插值交叉点"——这个 bootstrap 结果
    严格来说和 G/H 报的 C 值不是同一个操作点，用户已经指定用这个文件，
    这里如实标注这个口径差异，不悄悄改成别的数据源。
"""
import os
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['NLL', 'CP_norm']
N_BOOT = 2000
RNG = np.random.RandomState(2026)


def fix_interaction(path):
    with open(path) as f:
        d = json.load(f)
    for ds in DATASETS:
        for m in METHODS:
            cell = d[ds][m]
            if 'note' in cell:
                continue
            c00, c10, c01, c11 = cell['C00'], cell['C10'], cell['C01'], cell['C11']
            new_interaction = (c11 - c10) - (c01 - c00)
            old_interaction = cell['interaction']
            assert abs(new_interaction + old_interaction) < 1e-9, \
                f"sign-flip identity failed at {ds}/{m}: old={old_interaction} new={new_interaction}"
            cell['interaction_OLD_mu_convention'] = old_interaction
            cell['interaction'] = new_interaction
            cell['interaction_definition'] = "(C11-C10)-(C01-C00), i.e. orderA_sigma_effect - orderB_sigma_effect"
    with open(path, 'w') as f:
        json.dump(d, f, indent=2, default=float)
    return d


def bootstrap_abs_mu_minus_abs_sigma(per_seed):
    c00 = np.array(per_seed['C00_per_seed']); c10 = np.array(per_seed['C10_per_seed'])
    c01 = np.array(per_seed['C01_per_seed']); c11 = np.array(per_seed['C11_per_seed'])
    orderA_mu = c10 - c00; orderA_sigma = c11 - c10
    orderB_sigma = c01 - c00; orderB_mu = c11 - c01
    avg_mu = (orderA_mu + orderB_mu) / 2.0
    avg_sigma = (orderA_sigma + orderB_sigma) / 2.0
    per_seed_diff = np.abs(avg_mu) - np.abs(avg_sigma)  # 5 values, one per seed
    n = len(per_seed_diff)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        boot[i] = per_seed_diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'per_seed_diff_absmu_minus_abssigma': per_seed_diff.tolist(),
        'mean': float(per_seed_diff.mean()),
        'bootstrap_mean': float(boot.mean()),
        'ci95_lo': float(lo), 'ci95_hi': float(hi),
        'mean_dominant_by_ci': lo > 0,  # entire 95% CI above 0 => mean robustly dominates
    }


if __name__ == '__main__':
    g_path = os.path.join(R3_DIR, 'G_transformer_exact_interp_attribution.json')
    h_path = os.path.join(R3_DIR, 'H_lstm_exact_interp_attribution.json')
    fix_interaction(g_path)
    fix_interaction(h_path)
    print("Interaction sign fixed in G and H (verified via algebraic identity check).")

    with open(os.path.join(R3_DIR, 'B_attribution_raw_per_seed.json')) as f:
        raw = json.load(f)

    result = {}
    for backbone in ['LSTM', 'Transformer']:
        result[backbone] = {}
        for ds in DATASETS:
            result[backbone][ds] = {}
            for m in METHODS:
                cell = raw[backbone][ds][m]
                if 'note' in cell:
                    result[backbone][ds][m] = {'note': cell['note']}
                    continue
                r = bootstrap_abs_mu_minus_abs_sigma(cell)
                r['caveat'] = ("computed at the nearest-grid-point (same operating point as "
                                "frozen_decomposition_2x2.json / B_attribution_raw_per_seed.json), "
                                "NOT the exact-interpolation point used by G/H")
                result[backbone][ds][m] = r
                print(f"  {backbone}/{ds}/{m}: |mu|-|sigma| mean={r['mean']:.4f}  "
                      f"95%CI=[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]  "
                      f"dominant_by_CI={r['mean_dominant_by_ci']}")

    out_path = os.path.join(R3_DIR, 'bootstrap_absmu_minus_abssigma.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
