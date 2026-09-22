"""
R5-1：LSTM 侧 relative_half_life_feat_oob.json 的生成脚本此前在 code/ 里已经
找不到了（推测是当年一次性内联脚本，没有存成文件——supplementary_attribution
之类的产物也发生过同样的事）。本脚本用与 Transformer 侧
`mechanism_transformer.py::compute_half_life/pooled_points/crossover` 逐字
相同的定义重新生成，数据源换成 LSTM 的 noise_sensitivity_leakfree(.json/_armC)
+ FD003 专用文件，口径完全对齐（主SNR臂 A_percondition+B_pooled + 臂C，
pool后去重，abs阈值0.80，rel阈值picp_clean-0.10）。

不涉及新的随机扰动注入（只读已经算好的 dose-response 点做插值），不需要
PYTHONHASHSEED 断言。
"""
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'generated')
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm', 'MSE_fixed']


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def crossover(pts, threshold):
    pts = sorted(pts, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i]
        fo1, p1 = pts[i + 1]
        if p0 >= threshold and p1 < threshold:
            if p1 == p0:
                return fo0
            frac = (threshold - p0) / (p1 - p0)
            return fo0 + frac * (fo1 - fo0)
    if pts and pts[0][1] < threshold:
        return pts[0][0]
    return None


def collect_points(block, method, level_keys):
    pts = []
    for k in level_keys:
        if k not in block.get('feat_oob', {}) or k not in block.get(method, {}):
            continue
        pts.append((block['feat_oob'][k], picp_of(block[method][k], method)))
    return pts


def pooled_points_124(ds, method, main_lf, armc_lf):
    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = ['0.1', '0.5', '1', '2', '5']
    pts = []
    pts += collect_points(main_lf[ds]['A_percondition'], method, snr_keys_all)
    if ds != 'FD001':
        pts += collect_points(main_lf[ds]['B_pooled'], method, snr_keys_all)
    pts += collect_points(armc_lf[ds], method, pct_keys)
    seen = set(); uniq = []
    for p in pts:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq


def pooled_points_fd003(method, main_003, armc_003):
    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = ['0.1', '0.5', '1', '2', '5']
    pts = []
    pts += collect_points(main_003['B_pooled'], method, snr_keys_all)
    pts += collect_points(armc_003, method, pct_keys)
    seen = set(); uniq = []
    for p in pts:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq


if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree.json')) as f:
        main_lf = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC.json')) as f:
        armc_lf = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json')) as f:
        main_003 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
        armc_003 = json.load(f)

    result = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        result[ds] = {}
        for method in METHODS:
            if ds == 'FD003':
                pts = pooled_points_fd003(method, main_003['FD003'], armc_003['FD003'])
            else:
                pts = pooled_points_124(ds, method, main_lf, armc_lf)
            pts = sorted(pts, key=lambda p: p[0])
            picp_clean = pts[0][1]
            abs_co = crossover(pts, 0.80)
            rel_co = crossover(pts, picp_clean - 0.10)
            result[ds][method] = {'picp_clean': picp_clean, 'abs_co': abs_co, 'rel_co': rel_co}
            print(f"  {method:20s} picp_clean={picp_clean:.4f}  abs_co={abs_co}  rel_co={rel_co}")

    out_dir = os.path.join(RESULTS_DIR, 'leakfree')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'relative_half_life_feat_oob.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
