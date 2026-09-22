"""
R8-A2（诊断，推理级，不重训）：目标时刻对齐。

背景：训练集 RUL 标签 = (max_cycles - time_cycles).clip(125) —— 由于训练
发动机跑到真正失效（run-to-failure），最后一行 time_cycles=max_cycles，
RUL=0，即"这一行本身就是失效时刻"。测试集当前口径（V4.extract_raw_windows
mode='test'）直接用官方 RUL_FD00X.txt 的值（clip到125）作为最后一行窗口
的真值：`min(official_RUL, 125)`——official_RUL 是NASA给定的"测试文件最后
一行之后还剩多少个运行周期"，与训练标签"这一行本身还剩多少个周期（含
这一行）"是否是同一个计数起点，取决于official_RUL的确切定义，如果
official_RUL衡量的是"最后一行之后"而不是"最后一行开始"，就会跟训练标签
差1个周期。本诊断只做一件事：把测试真值换成
`min(official_RUL - 1, 125)`（与训练标签同一计数惯例），看RMSE/PICP/IS/
L=20提前触发率变化多大——不下结论哪个"对"，只如实报告差异幅度。

模型输出（mu,sigma）在两种口径下完全相同（RUL标签只用于评估，不是模型
输入），因此只需推理一次，用两套 y_true 分别算指标。

全部4数据集×两骨干×NLL×5 seeds，clean 与 drift 5% 两个条件。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R8_DIR = os.path.join(RESULTS_DIR, 'leakfree_r8')
os.makedirs(R8_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
CONDITIONS = ['clean', 'drift5pct']
L = 20
Z = C.Z_SCORE
ALPHA = 0.10


def interval_score(y, mu, sigma, z):
    lo = mu - z * sigma
    hi = mu + z * sigma
    width = hi - lo
    below = y < lo
    above = y > hi
    penalty = np.zeros_like(y)
    penalty[below] = (2.0 / ALPHA) * (lo[below] - y[below])
    penalty[above] = (2.0 / ALPHA) * (y[above] - hi[above])
    return width + penalty


def metrics_for(y, mu, sigma):
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    lo, hi = mu - Z * sigma, mu + Z * sigma
    picp = float(np.mean((y >= lo) & (y <= hi)))
    is_mean = float(interval_score(y, mu, sigma, Z).mean())
    triggered = lo <= L
    at_risk = y <= L
    premature = triggered & (y > L + 20)
    return {'rmse': rmse, 'picp': picp, 'interval_score_mean': is_mean,
            'premature_rate_L20': float(premature.mean()), 'n_at_risk': int(at_risk.sum())}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for ds in DATASETS:
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
        X_raw_clean, y_official, u_ids = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
        y_alt = np.minimum(true_ruls.iloc[u_ids - 1]['RUL'].values.astype(np.float64) - 1.0, C.MAX_RUL)
        X_raw_drift = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, full_scale)

        result[ds] = {}
        for backbone in BACKBONES:
            print(f"\n{'=' * 20} {ds} / {backbone} {'=' * 20}")
            result[ds][backbone] = {}
            for condition in CONDITIONS:
                X_raw = X_raw_clean if condition == 'clean' else X_raw_drift
                official_cells, alt_cells = [], []
                for seed in C.SEEDS:
                    fit_units = canon[ds][str(seed)]['fit_units']
                    _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                    X_scaled = V4.scale_raw_windows(X_raw, scaler)
                    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                    mu, ls = E.infer_nll(nll_model, X_t)
                    sigma = np.exp(ls) * 125.0
                    del nll_model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    official_cells.append(metrics_for(y_official, mu, sigma))
                    alt_cells.append(metrics_for(y_alt, mu, sigma))

                def agg(cells):
                    return {k: float(np.mean([c[k] for c in cells])) for k in
                            ['rmse', 'picp', 'interval_score_mean', 'premature_rate_L20']}

                result[ds][backbone][condition] = {
                    'official_RUL_current': agg(official_cells),
                    'RULminus1_train_convention': agg(alt_cells),
                }
                a, b = result[ds][backbone][condition]['official_RUL_current'], \
                    result[ds][backbone][condition]['RULminus1_train_convention']
                print(f"  [{condition:10s}] official: RMSE={a['rmse']:.3f} PICP={a['picp']:.4f} "
                      f"IS={a['interval_score_mean']:.2f} prem@L20={a['premature_rate_L20']:.4f}   |   "
                      f"RUL-1: RMSE={b['rmse']:.3f} PICP={b['picp']:.4f} "
                      f"IS={b['interval_score_mean']:.2f} prem@L20={b['premature_rate_L20']:.4f}")

    out_path = os.path.join(R8_DIR, 'A2_target_alignment_diagnostic.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
