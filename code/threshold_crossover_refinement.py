"""
R4-3：9.6% 阈值加密验证。Transformer/FD001/Deep_Ensemble 的相对半衰交叉点
（原两点 0dB->2.846%、-5dB->11.86% 之间线性插值得到 9.607%，D3 轮已用一份
稀疏网格证实这不是插值伪影）——本轮在 [2.85%, 11.86%] 区间内新增 6 个主臂
Gaussian-SNR 档位 {-1,-2,-3,-3.5,-4,-4.5} dB（先探测过，armC 的固定%FS
方案要到几十%FS才能进入这个feat_oob区间，不是"同一量纲"的加密，弃用；改用
和原两个端点同一条主臂SNR轴上的档位，是名副其实的"区间内加密"），5 个
trial 共享噪声（与主臂 run_snr_sweep 完全一致的 rng 标签
('mainarm','global',level_key,t)，可精确复现原两个端点的历史值）。

额外做的事：不仅报 grand-mean 的 PICP，还逐 trial 单独追踪 Ensemble PICP
（5 seed 的 moment-matching集成，对每个 trial 单独算一条 PICP-vs-foob 曲线），
在同一个 rel_threshold=picp_clean-0.10 下分别对 5 条 trial 专属曲线插值出 5 个
交叉点，报其范围——衡量"9.6%"这个数字对噪声实现本身的敏感度，而不仅仅是
对网格密度的敏感度（D3 轮已验证的是后者）。

只做推理，不重训。输出：
results/generated/leakfree_r4/threshold_960_refinement.json
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
import run_sweep_noise_transformer_armc as PA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')
R4_DIR = os.path.join(RESULTS_DIR, 'leakfree_r4')
os.makedirs(R4_DIR, exist_ok=True)

DS = 'FD001'
BACKBONE = 'Transformer'
N_TRIALS = V4.N_TRIALS  # 5
NEW_LEVELS_DB = [-1, -2, -3, -3.5, -4, -4.5]
ANCHOR_LEVELS_DB = [0, -5]  # existing bracket points, recomputed per-trial for a consistent curve


def picp_of_array(y, mu, sigma, z=C.Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    global_std = np.std(train_df_raw[feat_cols].values, axis=0)
    scalers_by_seed = PA.scalers_for_ds(DS)

    models_by_seed = {seed: E.load_models_for_seed(DS, BACKBONE, seed, device) for seed in C.SEEDS}

    with open(os.path.join(R3_DIR, 'D3_refined_grid_transformer_fd001_ensemble.json')) as f:
        d3 = json.load(f)
    picp_clean = d3['picp_clean']
    rel_threshold = d3['rel_threshold']
    print(f"picp_clean={picp_clean}  rel_threshold={rel_threshold}")

    all_levels = ANCHOR_LEVELS_DB + NEW_LEVELS_DB
    points = {}  # level_db -> {'feat_oob': float, 'picp_grand': float, 'picp_per_trial': [5 floats]}

    for level in all_levels:
        level_key = str(level)
        picp_per_trial = []
        feat_oob_per_trial = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(DS, BACKBONE, 'mainarm', 'global', level_key, t)))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)

            mu_mem, sigma_mem = [], []
            y_ref = None
            # 2026-09-21（本轮）：此前这里只用 C.SEEDS[0] 一个seed的scaler算
            # fo_this_trial（`if fo_this_trial is None` 只在第一个seed触发），
            # 且用 scale_and_package 返回的整段轨迹 scaled_feat——两个口径问题
            # 与主扫描脚本（sweep_engine.py等）此前的旧bug是同一类，只是这个
            # 脚本当时漏了没改。现在改成5个seed各自在窗口化X_test上算，取平均，
            # 与PICP自身的5模型汇总、以及本轮修的其它文件同一口径。
            fo_this_trial_per_seed = []
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                nll_model, _, _ = models_by_seed[seed]
                mu, ls = E.infer_nll(nll_model, X_t)
                sigma = np.exp(ls) * 125.0
                mu_mem.append(mu); sigma_mem.append(sigma)
                y_ref = y_test
                # R9-Part3: V4.feat_oob 全项目唯一实现，分母限定传感器列。
                fo_this_trial_per_seed.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
            fo_this_trial = float(np.mean(fo_this_trial_per_seed))

            mu_mem = np.stack(mu_mem); sigma_mem = np.stack(sigma_mem)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            picp_t = picp_of_array(y_ref, mu_ens, sigma_ens)
            picp_per_trial.append(picp_t)
            feat_oob_per_trial.append(fo_this_trial)

        points[level] = {
            'feat_oob_mean': float(np.mean(feat_oob_per_trial)),
            'feat_oob_per_trial': feat_oob_per_trial,
            'picp_grand_mean': float(np.mean(picp_per_trial)),
            'picp_per_trial': picp_per_trial,
        }
        print(f"  SNR={level:>5}dB  foob={points[level]['feat_oob_mean']*100:.4f}%  "
              f"PICP(grand-mean over 5 trials)={points[level]['picp_grand_mean']:.4f}  "
              f"per-trial={[f'{p:.4f}' for p in picp_per_trial]}")

    # ---- grand-mean curve crossover (uses feat_oob_mean, picp_grand_mean per point) ----
    grand_pts = sorted([(points[lv]['feat_oob_mean'], points[lv]['picp_grand_mean'], lv) for lv in all_levels])

    def interp_crossover(pts_sorted, threshold):
        for i in range(len(pts_sorted) - 1):
            fo0, p0, _ = pts_sorted[i]; fo1, p1, _ = pts_sorted[i + 1]
            if p0 >= threshold and p1 < threshold:
                frac = 0.0 if p1 == p0 else (threshold - p0) / (p1 - p0)
                return fo0 + frac * (fo1 - fo0), (pts_sorted[i], pts_sorted[i + 1])
        return None, None

    co_grand, bracket_grand = interp_crossover(grand_pts, rel_threshold)
    print(f"\nRefined grand-mean crossover: {co_grand*100:.4f}%  "
          f"(bracket: {bracket_grand[0][2]}dB@{bracket_grand[0][0]*100:.4f}% -- "
          f"{bracket_grand[1][2]}dB@{bracket_grand[1][0]*100:.4f}%)")

    # ---- per-trial curve crossover: 5 separate curves using that trial's own feat_oob/PICP ----
    per_trial_crossovers = []
    for t in range(N_TRIALS):
        pts_t = sorted([(points[lv]['feat_oob_per_trial'][t], points[lv]['picp_per_trial'][t], lv)
                         for lv in all_levels])
        co_t, bracket_t = interp_crossover(pts_t, rel_threshold)
        per_trial_crossovers.append(co_t)
        print(f"  trial {t}: crossover={co_t*100:.4f}%" if co_t is not None else f"  trial {t}: never crosses")

    valid = [c for c in per_trial_crossovers if c is not None]
    result = {
        'picp_clean': picp_clean, 'rel_threshold': rel_threshold,
        'points': {str(lv): points[lv] for lv in all_levels},
        'refined_crossover_grand_mean': co_grand,
        'per_trial_crossovers': per_trial_crossovers,
        'per_trial_crossover_range_pct': {
            'min': float(min(valid) * 100) if valid else None,
            'max': float(max(valid) * 100) if valid else None,
            'spread_pct_points': float((max(valid) - min(valid)) * 100) if valid else None,
        },
        'prior_D3_refined_crossover': d3['refined_crossover_feat_oob'],
    }

    out_path = os.path.join(R4_DIR, 'threshold_960_refinement.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
