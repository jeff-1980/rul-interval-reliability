"""
R5-2：D1 固定终点乱序判别对照重新生成。原脚本已不在 code/ 里（推测是内联
一次性脚本，未存档——与 relative_half_life_feat_oob.json 同样的情况），按
`leakfree_r3/D1_fixed_endpoint_shuffle.json` 的既有结构（{backbone:{k:PICP}}，
k in {1,3,5}，FD002，5%FS，heteroscedastic/NLL 单一机制）用
`noise_injection.inject_drift_fixed_endpoint_shuffle_windows` 重新实现，
5 trial 共享噪声（新 stable_seed 标签 'r5_d1_shuffle'）。
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
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')

DS = 'FD002'
BACKBONES = ['LSTM', 'Transformer']
PCT = 5.0
K_VALUES = [1, 3, 5]
N_TRIALS = V4.N_TRIALS


def scalers_for(ds, canon):
    s = {}
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        s[seed] = scaler
    return s


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
    full_scale = V4.sensor_only_scale(feat_cols, full_scale)  # R8-B1
    X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')

    result = {}
    for backbone in BACKBONES:
        print(f"\n{'=' * 20} {backbone} {'=' * 20}")
        scalers_by_seed = scalers_for(DS, canon)
        nll_model_by_seed = {seed: T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, DS, seed), device)
                              for seed in C.SEEDS}
        result[backbone] = {}
        for k_end in K_VALUES:
            trial_picps = []
            for t in range(N_TRIALS):
                rng = np.random.RandomState(C.stable_seed(DS, backbone, 'r5_d1_shuffle', k_end, t))
                X_shuf = V4.inject_drift_fixed_endpoint_shuffle_windows(X_raw_clean, PCT, full_scale, k_end, rng)
                seed_mu, seed_sigma = [], []
                for seed in C.SEEDS:
                    scaler = scalers_by_seed[seed]
                    X_scaled = V4.scale_raw_windows(X_shuf, scaler)
                    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                    mu, ls = E.infer_nll(nll_model_by_seed[seed], X_t)
                    sigma = np.exp(ls) * 125.0
                    picp, _ = E.picp_mpiw(y_ref, mu, sigma)
                    trial_picps.append(picp)
            result[backbone][str(k_end)] = float(np.mean(trial_picps))
            print(f"  k_end={k_end}  PICP={result[backbone][str(k_end)]:.4f}")
        del nll_model_by_seed
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_path = os.path.join(R3_DIR, 'D1_fixed_endpoint_shuffle.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
