"""
STEP 5m：LSTM 冻结σ̂分解，改在**相对**半衰点（PICP从clean跌10点）计算，
不是`frozen_sigma_decomposition_leakfree.json`原来用的**绝对**半衰点
（PICP首次跌破0.80）——2026-09-19核实：论文`paper2_main.tex`
Table~attribution标题写"relative half-life point"，但底层文件实际是绝对
半衰点算的，标注与实现不一致。本脚本重新计算相对半衰点版本，写到新文件
`frozen_sigma_decomposition_RELATIVE_leakfree.json`，不覆盖/不修改旧的
绝对版本文件（旧结果只读，新旧并存，各自命名清楚）。

数据源：FD001/FD002/FD004 从 `dose_response_feat_oob_leakfree.json`
（已经pool了主臂A+B+臂C）；FD003 从 `noise_sensitivity_leakfree_FD003.json`
（主臂，单一工况A=B退化）+ `noise_sensitivity_leakfree_armC_FD003.json`
现场pool。逻辑与 dose_response_frozen_sigma.py 完全
一致，只改阈值来源（picp_clean-0.10 而不是固定0.80）。
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
DECOMPOSE_METHODS = ['NLL', 'CP_norm']


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def crossover(pts, threshold):
    pts = sorted(pts, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i][0], pts[i][1]
        fo1, p1 = pts[i + 1][0], pts[i + 1][1]
        if p0 >= threshold and p1 < threshold:
            frac = 0 if p1 == p0 else (threshold - p0) / (p1 - p0)
            return fo0 + frac * (fo1 - fo0)
    if pts and pts[0][1] < threshold:
        return pts[0][0]
    return None


def interp_picp(sorted_pts, x):
    xs = [p[0] for p in sorted_pts]; ys = [p[1] for p in sorted_pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = 0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]


def decompose(real_pts, frozen_pts):
    real_pts = sorted(real_pts, key=lambda p: p[0])
    frozen_pts = sorted(frozen_pts, key=lambda p: p[0])
    picp_clean = real_pts[0][1]
    rel_threshold = picp_clean - 0.10
    co = crossover(real_pts, rel_threshold)
    if co is None:
        return {'note': 'never crosses relative threshold in measured range', 'picp_clean': picp_clean}
    picp_clean_frozen = frozen_pts[0][1]
    picp_frozen_at_co = interp_picp(frozen_pts, co)
    delta_total = picp_clean - rel_threshold
    delta_mu_only = picp_clean_frozen - picp_frozen_at_co
    delta_sigma = delta_total - delta_mu_only
    frac = delta_sigma / delta_total if delta_total != 0 else None
    return {
        'crossover_feat_oob_relative': co, 'picp_clean': picp_clean,
        'picp_clean_real': picp_clean, 'picp_clean_frozen_sigma_counterfactual': picp_clean_frozen,
        'picp_frozen_sigma_counterfactual_at_crossover': picp_frozen_at_co,
        'delta_picp_total': delta_total, 'delta_picp_mu_only_frozen_sigma_counterfactual': delta_mu_only,
        'delta_picp_sigma_contribution': delta_sigma, 'sigma_contribution_fraction': frac,
    }


def fd001_2_4(ds, dose_response):
    out = {}
    for method in DECOMPOSE_METHODS:
        frozen_key = f"{method}_frozen_sigma"
        real_pts = [(p[0], p[1]) for p in dose_response[ds][method]['points_feat_oob_picp_sigma']]
        frozen_pts = [(p[0], p[1]) for p in dose_response[ds][frozen_key]['points_feat_oob_picp_sigma']]
        out[method] = decompose(real_pts, frozen_pts)
    return out


def fd003():
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json')) as f:
        main_lf = json.load(f)['FD003']['B_pooled']
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
        armc_lf = json.load(f)['FD003']
    out = {}
    for method in DECOMPOSE_METHODS:
        frozen_key = f"{method}_frozen_sigma"
        real_pts, frozen_pts = [], []
        for k, fo in main_lf['feat_oob'].items():
            real_pts.append((fo, picp_of(main_lf[method][k], method)))
            frozen_pts.append((fo, picp_of(main_lf[frozen_key][k], method)))
        for k, fo in armc_lf['feat_oob'].items():
            real_pts.append((fo, picp_of(armc_lf[method][k], method)))
            frozen_pts.append((fo, picp_of(armc_lf[frozen_key][k], method)))
        out[method] = decompose(real_pts, frozen_pts)
    return out


if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'dose_response_feat_oob_leakfree.json')) as f:
        dose_response = json.load(f)

    result = {}
    for ds in ['FD001', 'FD002', 'FD004']:
        result[ds] = fd001_2_4(ds, dose_response)
    result['FD003'] = fd003()

    for ds in DATASETS:
        print(f"\n{ds}:")
        for m in DECOMPOSE_METHODS:
            v = result[ds][m]
            if 'sigma_contribution_fraction' in v:
                print(f"  {m}: crossover={v['crossover_feat_oob_relative']:.5f}  "
                      f"sigma_contribution_fraction={v['sigma_contribution_fraction']:+.4f}")
            else:
                print(f"  {m}: {v['note']}")

    out_path = os.path.join(RESULTS_DIR, 'frozen_sigma_decomposition_RELATIVE_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("(旧的绝对半衰点版本 frozen_sigma_decomposition_leakfree.json 保留不动)")
