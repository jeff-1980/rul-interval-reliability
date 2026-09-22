"""
T2-A4：Transformer 骨干自己的臂C（固定%FS）扫描，4 数据集，供计算 Transformer
自身的 clamp_frac / 绝对+相对半衰 f_oob / 冻结σ̂分解——即 Part A 要回答的两个
问题("扰动下失效是否仍以μ̂为主、方差贡献符号是否与LSTM一致"、"相对半衰的
机制排序是否与LSTM一致")所需的 Transformer 侧原始数据。

范围说明（如实记录）：只用臂C（5档固定%FS，条件无关），不重复 LSTM 侧当年
额外做的主SNR臂（A/B，9+2档）。理由：(a) 臂C本身已经给出跨越
0.37%-2%feat_oob量级的5个点，半衰/clamp_frac/冻结σ̂分解在这个范围内已经
稳定收敛（FD003当初也是这样验证的）；(b) Part B的三类新增确定性劣化本来就
只要求臂C同一套绝对尺度，Part A自己的基线用同一套尺度可以和Part B的高斯
噪声基线直接放在同一张剂量-反应图上比较，不需要额外算一遍主SNR臂。
"""
import os
import json
import time

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
PCT_LEVELS = V4.PCT_LEVELS


def scalers_for_ds(ds):
    with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    scalers = {}
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        scalers[seed] = scaler
    return scalers


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  Part A armC sweep  DATASETS={DATASETS}")

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_armC_sweep_leakfree.json')
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_out = json.load(f)
        print(f"Resuming: found existing results for {list(all_out.keys())}")

    for ds in DATASETS:
        if ds in all_out:
            print(f"\n{'=' * 20} {ds} (already done, skip) {'=' * 20}")
            continue
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        t0 = time.time()
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
        full_scale = V4.sensor_only_scale(feat_cols, full_scale)  # R8-B1
        scalers_by_seed = scalers_for_ds(ds)

        result = E.run_df_perturb_sweep(
            ds, 'Transformer', 'armC_gaussian', V4.inject_noise_fixed_pct_raw, PCT_LEVELS,
            is_pct=True, device=device, scalers_by_seed=scalers_by_seed, full_scale=full_scale)

        all_out[ds] = result
        with open(out_path, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)
        print(f"  [{ds}] done in {time.time() - t0:.1f}s, saved -> {out_path}")

    print("\nT2-A4 (Transformer armC sweep) complete.")
