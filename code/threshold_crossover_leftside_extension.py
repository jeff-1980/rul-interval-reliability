"""
R6-2：9.6%阈值加密，补左侧点。已有右侧点覆盖 [2.83%, 11.83%]（0dB..-5dB）；
这里在 clean(0%) 和 2.83%(0dB) 之间再加两档：SNR=2dB(~1.4%)、SNR=1dB(~1.9%)
——探测过，同一条主臂SNR轴，实际达到值不是正好1%/2%，如实报告真实值，不
硬凑整数。

逐 trial 报"从 clean 起首次交叉"：把 clean(0%,picp_clean) 也当成曲线起点，
和该 trial 自己的 7 个点（新增2个左侧 + 已有0dB..-4.5dB右侧，不含最右端
-5dB，避免跳过中间可能的多次穿越）按 feat_oob 从小到大排序，从 clean 起
向右扫描，找第一次跌破 rel_threshold 的位置线性插值。如果这条 trial 专属
曲线在扫到的所有新点上都还没跌破（即最左侧新点 SNR=1dB 那里 PICP 仍
>= threshold，跌破发生在 1dB 点和原来最近的 0dB/2.83% 点之间，或者更远），
按用户要求统一标"<=2.84%"（用原始最小已测点的百分比做上界标签，不用
插值猜测更靠左的具体数字）。
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
R6_DIR = os.path.join(RESULTS_DIR, 'leakfree_r6')
os.makedirs(R6_DIR, exist_ok=True)

DS = 'FD001'
BACKBONE = 'Transformer'
N_TRIALS = V4.N_TRIALS
NEW_LEFT_LEVELS_DB = [2, 1]
ORIGINAL_MIN_FOOB_PCT = 2.84  # 0dB point, the fallback upper-bound label


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

    with open(os.path.join(R4_DIR, 'threshold_960_refinement.json')) as f:
        right = json.load(f)
    picp_clean = right['picp_clean']
    rel_threshold = right['rel_threshold']
    print(f"picp_clean={picp_clean}  rel_threshold={rel_threshold}")

    # ---- new left-side points ----
    left_points = {}
    for level in NEW_LEFT_LEVELS_DB:
        level_key = str(level)
        picp_per_trial, feat_oob_per_trial = [], []
        for t in range(N_TRIALS):
            rng = np.random.RandomState(C.stable_seed(DS, BACKBONE, 'mainarm', 'global', level_key, t))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)
            mu_mem, sigma_mem = [], []
            y_ref = None
            # 2026-09-21（本轮）：见 threshold_crossover_refinement.py 同日
            # 同条注释——此前只用第一个seed的scaler+整段轨迹算feat_oob，现在改成
            # 5个seed各自在窗口化X_test上算，取平均。
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
            picp_per_trial.append(picp_of_array(y_ref, mu_ens, sigma_ens))
            feat_oob_per_trial.append(fo_this_trial)
        left_points[level] = {'feat_oob_per_trial': feat_oob_per_trial, 'picp_per_trial': picp_per_trial,
                               'feat_oob_mean': float(np.mean(feat_oob_per_trial)),
                               'picp_grand_mean': float(np.mean(picp_per_trial))}
        print(f"  SNR={level}dB  foob={left_points[level]['feat_oob_mean']*100:.4f}%  "
              f"PICP={left_points[level]['picp_grand_mean']:.4f}")

    # ---- per-trial "first crossing from clean" ----
    right_points = right['points']  # {'0': {...}, '-1': {...}, ...}
    RIGHT_LEVELS = [0, -1, -2, -3, -3.5, -4, -4.5, -5]

    def interp_crossover_from_clean(pts_sorted, threshold):
        # pts_sorted includes (0.0, picp_clean) as the first point
        for i in range(len(pts_sorted) - 1):
            fo0, p0 = pts_sorted[i]; fo1, p1 = pts_sorted[i + 1]
            if p0 >= threshold and p1 < threshold:
                frac = 0.0 if p1 == p0 else (threshold - p0) / (p1 - p0)
                return fo0 + frac * (fo1 - fo0)
        return None  # never crosses within the tested range -> caller applies fallback label

    per_trial_results = []
    for t in range(N_TRIALS):
        pts = [(0.0, picp_clean)]
        for lv in NEW_LEFT_LEVELS_DB:
            pts.append((left_points[lv]['feat_oob_per_trial'][t], left_points[lv]['picp_per_trial'][t]))
        for lv in RIGHT_LEVELS:
            rp = right_points[str(lv)]
            pts.append((rp['feat_oob_per_trial'][t], rp['picp_per_trial'][t]))
        pts.sort(key=lambda p: p[0])
        co = interp_crossover_from_clean(pts, rel_threshold)
        if co is None:
            label = f"<={ORIGINAL_MIN_FOOB_PCT}%"
        else:
            label = f"{co*100:.4f}%"
        per_trial_results.append({'trial': t, 'crossover_pct_label': label,
                                   'crossover_frac': co, 'points_used': pts})
        print(f"  trial {t}: first crossing from clean = {label}")

    result = {
        'picp_clean': picp_clean, 'rel_threshold': rel_threshold,
        'new_left_points_db_and_actual_pct': {str(lv): left_points[lv]['feat_oob_mean'] * 100
                                               for lv in NEW_LEFT_LEVELS_DB},
        'left_points_raw': {str(lv): left_points[lv] for lv in NEW_LEFT_LEVELS_DB},
        'per_trial_first_crossing_from_clean': per_trial_results,
    }
    out_path = os.path.join(R6_DIR, 'threshold_960_leftside_extension.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
