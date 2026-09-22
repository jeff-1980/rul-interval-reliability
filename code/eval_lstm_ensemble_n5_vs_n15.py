"""
STEP 2e：n=5 vs n=15 种子间方差对比，leakfree协议。

R8-B（本轮）：额外10个种子不再从一个单独的训练脚本产出的结果文件读取（那
会与本轮"不重训"矛盾）；改为直接从已有的
`results/checkpoints/lstm/{ds}_LSTM_extraseed{seed}.pt`（10个额外种子的
checkpoint，仍是原始训练权重，本轮未重训）做推理，用本仓库统一的测试真值
口径（min(官方RUL-1,125)，本轮起对全部表格生效）重新计算 rmse/picp/mpiw。
5-seed 侧直接读取本轮已更新的 `step0c_leakfree_results.json`。

同一统计口径的旧版（非leakfree）：对每个数据集×指标(rmse/picp/mpiw)，
比较原始5 seeds的均值/std 与 全部15 seeds（5原始+10额外）的均值/std，
报告 std_ratio_15_over_5。
"""
import os
import json

import numpy as np
import torch

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

DATASETS = ['FD001', 'FD002', 'FD004']
EXTRA_SEEDS = [8601, 16713, 19906, 21852, 38466, 40016, 40967, 51778, 52287, 90270]


def eval_extraseed(ds, seed, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{ds}_LSTM_extraseed{seed}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fit_units = ckpt['fit_units']
    model = C.load_checkpoint_model(ckpt_path, device)

    train_df, test_df, true_ruls, feat_cols, _ = C.load_and_process_leakfree(ds, fit_units)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = np.exp(log_sigma_out.cpu().numpy().flatten()) * 125.0

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=C.Z_SCORE)
    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw,
            'sigma_mean': float(sigma_np.mean())}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(RESULTS_DIR, 'step0c_leakfree_results.json')) as f:
        n5 = json.load(f)

    results = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        extra10_rows = [eval_extraseed(ds, seed, device) for seed in EXTRA_SEEDS]
        for r in extra10_rows:
            print(f"   seed={r['seed']}  RMSE={r['rmse']:.3f} Score={r['score']:.1f} "
                  f"PICP={r['picp']:.3f} MPIW={r['mpiw']:.2f}")

        results[ds] = {}
        n15_rows = n5[ds] + extra10_rows
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
    print("STEP2e complete (R8: extra-10-seed metrics recomputed by inference from existing "
          "checkpoints under min(official_RUL-1,125), no retraining).")
