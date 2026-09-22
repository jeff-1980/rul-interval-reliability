"""
T2 对照：Part A 要回答的两个问题。

(a) 扰动下失效是否仍以 μ̂ 为主？方差贡献的符号在四个数据集上是否与 LSTM
    一致？
(b) 相对半衰的机制排序是否与 LSTM 一致？

方法论说明（重要，如实记录）：LSTM 侧已有的
`frozen_sigma_decomposition_leakfree.json`（FD001/2/4）实际是在**绝对**
半衰点（PICP 首次跌破 0.80）计算的frozen-sigma分解，尽管论文正文
`paper2_main.tex`的Table~attribution标题写的是"relative half-life
point"——这是一处独立于本次T2任务、此前就存在的标注/实现不一致，本脚本
不去动论文或旧结果文件（旧结果只读），而是从底层原始 dose-response 点
（`dose_response_feat_oob_leakfree.json` 的 FD001/2/4 + FD003 侧
`noise_sensitivity_leakfree_FD003.json`/`_armC_FD003.json`）**在本脚本内部
重新计算**LSTM在**相对**半衰点的分解，使其与T2 Transformer侧（本来就按
相对半衰点计算）严格同口径可比。这个重算结果只用于T2的(a)问题对比，
不回写任何旧文件。已单独把这条不一致报给用户，供其决定是否需要回头修
论文的Table~attribution标题或底层脚本。

只读取 LSTM 已有结果 + Transformer 的 T2 产出，不改动任何已有文件。
"""
import os
import json

import numpy as np

import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
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


def lstm_relative_decomposition_fd001_2_4(ds, dose_response):
    out = {}
    for method in DECOMPOSE_METHODS:
        frozen_key = f"{method}_frozen_sigma"
        real_pts = [(p[0], p[1]) for p in dose_response[ds][method]['points_feat_oob_picp_sigma']]
        frozen_pts = [(p[0], p[1]) for p in dose_response[ds][frozen_key]['points_feat_oob_picp_sigma']]
        real_pts_sorted = sorted(real_pts, key=lambda p: p[0])
        picp_clean = real_pts_sorted[0][1]
        rel_threshold = picp_clean - 0.10
        co = crossover(real_pts, rel_threshold)
        if co is None:
            out[method] = {'note': 'never crosses relative threshold'}
            continue
        picp_clean_frozen = sorted(frozen_pts, key=lambda p: p[0])[0][1]
        picp_frozen_at_co = interp_picp(sorted(frozen_pts, key=lambda p: p[0]), co)
        delta_total = picp_clean - rel_threshold
        delta_mu_only = picp_clean_frozen - picp_frozen_at_co
        delta_sigma = delta_total - delta_mu_only
        frac = delta_sigma / delta_total if delta_total != 0 else None
        out[method] = {'crossover_feat_oob_relative': co, 'sigma_contribution_fraction': frac}
    return out


def lstm_relative_decomposition_fd003():
    with open(os.path.join(T2.RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json')) as f:
        main_lf = json.load(f)['FD003']['B_pooled']
    with open(os.path.join(T2.RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
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
        real_pts_sorted = sorted(real_pts, key=lambda p: p[0])
        picp_clean = real_pts_sorted[0][1]
        rel_threshold = picp_clean - 0.10
        co = crossover(real_pts, rel_threshold)
        if co is None:
            out[method] = {'note': 'never crosses relative threshold'}
            continue
        frozen_pts_sorted = sorted(frozen_pts, key=lambda p: p[0])
        picp_clean_frozen = frozen_pts_sorted[0][1]
        picp_frozen_at_co = interp_picp(frozen_pts_sorted, co)
        delta_total = picp_clean - rel_threshold
        delta_mu_only = picp_clean_frozen - picp_frozen_at_co
        delta_sigma = delta_total - delta_mu_only
        frac = delta_sigma / delta_total if delta_total != 0 else None
        out[method] = {'crossover_feat_oob_relative': co, 'sigma_contribution_fraction': frac}
    return out


if __name__ == '__main__':
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_frozen_sigma_decomposition.json')) as f:
        t2_decomp = json.load(f)
    with open(os.path.join(T2.RESULTS_DIR, 'dose_response_feat_oob_leakfree.json')) as f:
        dose_response = json.load(f)

    lstm_decomp = {}
    for ds in ['FD001', 'FD002', 'FD004']:
        lstm_decomp[ds] = lstm_relative_decomposition_fd001_2_4(ds, dose_response)
    lstm_decomp['FD003'] = lstm_relative_decomposition_fd003()

    print("=" * 70)
    print("(a) Variance-contribution sign at the RELATIVE half-life point: Transformer vs LSTM")
    print("(LSTM side recomputed in this script at the relative threshold for apples-to-apples;")
    print(" the pre-existing frozen_sigma_decomposition_leakfree.json uses the ABSOLUTE threshold")
    print(" despite the paper caption saying 'relative' -- separate, pre-existing inconsistency,")
    print(" reported to the user, not fixed here.)")
    print("=" * 70)
    sign_matches, sign_total = 0, 0
    for ds in DATASETS:
        for method in DECOMPOSE_METHODS:
            t2_frac = t2_decomp.get(ds, {}).get(method, {}).get('sigma_contribution_fraction')
            lstm_frac = lstm_decomp.get(ds, {}).get(method, {}).get('sigma_contribution_fraction')
            if t2_frac is None or lstm_frac is None:
                print(f"  {ds:6} {method:10}: T2={t2_frac}  LSTM={lstm_frac}  (unavailable)")
                continue
            same_sign = (t2_frac >= 0) == (lstm_frac >= 0)
            sign_total += 1; sign_matches += int(same_sign)
            print(f"  {ds:6} {method:10}: T2={t2_frac:+.3f}  LSTM={lstm_frac:+.3f}  "
                  f"{'SAME sign' if same_sign else 'OPPOSITE sign'}")
    print(f"\nSign agreement: {sign_matches}/{sign_total}")

    print("\n" + "=" * 70)
    print("(b) Relative half-life ordering: Transformer vs LSTM")
    print("=" * 70)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_half_life_feat_oob.json')) as f:
        t2_halflife = json.load(f)
    with open(os.path.join(T2.RESULTS_DIR, 'leakfree', 'relative_half_life_feat_oob.json')) as f:
        lstm_halflife = json.load(f)

    order_matches = 0
    per_ds_orders = {}
    for ds in DATASETS:
        t2_rel = {m: t2_halflife[ds][m]['rel_co'] for m in METHODS if t2_halflife[ds][m]['rel_co'] is not None}
        lstm_rel = {m: lstm_halflife[ds][m]['rel_co'] for m in METHODS if lstm_halflife[ds][m]['rel_co'] is not None}
        t2_order = sorted(t2_rel, key=lambda m: t2_rel[m])
        lstm_order = sorted(lstm_rel, key=lambda m: lstm_rel[m])
        match = t2_order == lstm_order
        order_matches += int(match)
        per_ds_orders[ds] = {'t2_order': t2_order, 'lstm_order': lstm_order, 'identical': match}
        print(f"\n  {ds}:")
        print(f"    T2 order (least->most tolerant):   {t2_order}")
        print(f"    LSTM order (least->most tolerant): {lstm_order}")
        print(f"    {'IDENTICAL ordering' if match else 'DIFFERENT ordering'}")
        for m in METHODS:
            t2v = t2_halflife[ds].get(m, {}).get('rel_co')
            lstmv = lstm_halflife[ds].get(m, {}).get('rel_co')
            print(f"      {m:20}: T2 rel_co={t2v}  LSTM rel_co={lstmv}")

    print(f"\nOrdering agreement: {order_matches}/{len(DATASETS)} datasets have identical full ranking")

    out = {
        'variance_sign_agreement': f"{sign_matches}/{sign_total}",
        'halflife_ordering_agreement': f"{order_matches}/{len(DATASETS)}",
        'per_dataset_ordering': per_ds_orders,
        'lstm_relative_decomposition_recomputed': lstm_decomp,
        'note': 'LSTM frozen-sigma decomposition recomputed at relative threshold in this script for '
                'apples-to-apples comparison; pre-existing frozen_sigma_decomposition_leakfree.json uses '
                'absolute threshold despite paper caption saying relative -- flagged to user separately.',
    }
    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_partA_comparison_summary.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"\nSaved summary -> {out_path}")
