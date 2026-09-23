"""
T2-A4b (added later): Transformer main SNR-arm sweep. Only pooling this
together with arm C (run_sweep_noise_transformer_armc.py) gives the same
feat_oob coverage range as the LSTM side's half-life computation -- arm
C's own smallest level (0.1% FS) already has feat_oob around 5%-6%, while
the LSTM side's half-life mostly falls at <2% feat_oob, which arm C alone
cannot measure; this script was added after finding that gap.

FD001/FD003 (single operating condition): only runs scheme='global'
(A_percondition degenerates to B_pooled on these datasets, matching the
LSTM side's handling -- both labels point to the same result).
FD002/FD004 (multiple operating conditions): runs both A_percondition
(per_condition) and B_pooled (global) arms, matching the LSTM side exactly.
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
import run_sweep_noise_transformer_armc as PA

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]

OUT_PATH = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_mainarm_sweep_leakfree.json')

if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  Part A main-SNR-arm sweep (added later)  DATASETS={DATASETS}")

    all_out = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            all_out = json.load(f)
        print(f"Resuming: found existing results for {list(all_out.keys())}")

    for ds in DATASETS:
        if ds in all_out:
            print(f"\n{'=' * 20} {ds} (already done, skip) {'=' * 20}")
            continue
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        t0 = time.time()
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        scalers_by_seed = PA.scalers_for_ds(ds)

        if ds in ('FD001', 'FD003'):
            global_std = np.std(train_df_raw[feat_cols].values, axis=0)
            main_result = E.run_snr_sweep(ds, 'Transformer', 'global', SNR_LEVELS_ALL, device,
                                           scalers_by_seed, global_std=global_std)
            all_out[ds] = {'A_percondition': main_result, 'B_pooled': main_result,
                            'note': 'single condition; A_percondition degenerates to B_pooled'}
        else:
            km, cond_std, global_std = V4.fit_condition_model(train_df_raw, feat_cols)
            cond_std = {c: V4.sensor_only_scale(feat_cols, v) for c, v in cond_std.items()}
            global_std = V4.sensor_only_scale(feat_cols, global_std)
            arm_a = E.run_snr_sweep(ds, 'Transformer', 'per_condition', SNR_LEVELS_ALL, device,
                                     scalers_by_seed, km=km, cond_std=cond_std)
            arm_b = E.run_snr_sweep(ds, 'Transformer', 'global', SNR_LEVELS_ALL, device,
                                     scalers_by_seed, global_std=global_std)
            all_out[ds] = {'A_percondition': arm_a, 'B_pooled': arm_b}

        with open(OUT_PATH, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)
        print(f"  [{ds}] done in {time.time() - t0:.1f}s, saved -> {OUT_PATH}")

    print("\nT2-A4b (Transformer main-SNR-arm sweep) complete.")
