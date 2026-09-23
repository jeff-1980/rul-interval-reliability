"""
T2-B：三种确定性劣化（bias/drift/gain）× 两骨干（LSTM/Transformer）× 四数据集，
臂C绝对尺度（{0.1,0.5,1,2,5}% FS，工况无关，归一化前注入原始读数，模型不
重训，5模型×5trial共享流设计），产物落 leakfree_t2/。

LSTM 侧复用已有 checkpoints_leakfree/ 下的 checkpoint（不重训，只是新的
扰动类型下重新推理）；Transformer 侧复用 T2 训练产出的 checkpoint。

按 (backbone, degradation, dataset) 为最小单位增量保存 + 断点续跑，任何一次
中断（例如系统重启）只丢当前正在跑的那一个 (backbone,degradation,dataset)
组合，不影响已完成的。
"""
import os
import json
import time
import functools

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
import run_sweep_noise_transformer_armc as PA

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
DEGRADATIONS = ['bias', 'drift', 'gain']
PCT_LEVELS = V4.PCT_LEVELS

OUT_PATH = os.path.join(T2.TRANSFORMER_DIR, 't2_degradation_sweep_leakfree.json')


def load_all():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {}


def save_all(all_out):
    with open(OUT_PATH, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)


def run_one(backbone, degradation, ds, device):
    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
    full_scale = V4.sensor_only_scale(feat_cols, full_scale)  # R8-B1
    scalers_by_seed = PA.scalers_for_ds(ds)

    if degradation == 'bias':
        return E.run_df_perturb_sweep(ds, backbone, 'bias', V4.inject_bias_fixed_pct_raw, PCT_LEVELS,
                                       is_pct=True, device=device, scalers_by_seed=scalers_by_seed,
                                       full_scale=full_scale)
    elif degradation == 'gain':
        # R9-Part2: gain's multiplier doesn't depend on full_scale_range, so
        # sensor_only_scale() above has no effect on it -- bind sensor_mask
        # explicitly instead.
        gain_fn = functools.partial(V4.inject_gain_fixed_pct_raw, sensor_mask=V4.sensor_mask_for(feat_cols))
        return E.run_df_perturb_sweep(ds, backbone, 'gain', gain_fn, PCT_LEVELS,
                                       is_pct=True, device=device, scalers_by_seed=scalers_by_seed,
                                       full_scale=full_scale)
    elif degradation == 'drift':
        return E.run_drift_sweep(ds, backbone, PCT_LEVELS, device, scalers_by_seed, full_scale)
    else:
        raise ValueError(degradation)


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Part B: degradations={DEGRADATIONS}  backbones={BACKBONES}  datasets={DATASETS}")

    all_out = load_all()

    for backbone in BACKBONES:
        all_out.setdefault(backbone, {})
        for degradation in DEGRADATIONS:
            all_out[backbone].setdefault(degradation, {})
            for ds in DATASETS:
                if ds in all_out[backbone][degradation]:
                    print(f"[{backbone}/{degradation}/{ds}] already done, skip")
                    continue
                print(f"\n{'=' * 20} {backbone} / {degradation} / {ds} {'=' * 20}")
                t0 = time.time()
                result = run_one(backbone, degradation, ds, device)
                all_out[backbone][degradation][ds] = result
                save_all(all_out)
                print(f"  done in {time.time() - t0:.1f}s, saved -> {OUT_PATH}")

    print("\nT2-B (three deterministic degradations, both backbones) complete.")
