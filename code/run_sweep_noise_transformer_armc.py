"""
Arm-C (fixed %FS) sweep for the Transformer backbone itself, 4 datasets,
providing Transformer's own clamp_frac / absolute+relative half-life
f_oob / frozen-sigma decomposition -- the raw Transformer-side data needed
to answer Part A's two questions ("under perturbation, is failure still
dominated by mu_hat, and is the sign of the variance contribution
consistent with LSTM", "is the mechanism ranking at the relative
half-life point consistent with LSTM").

Scope note (reported honestly): only uses arm C (5 fixed-%FS levels,
condition-independent), not the extra main SNR arm (A/B, 9+2 levels) that
was additionally run on the LSTM side. Rationale: (a) arm C alone already
gives 5 points spanning the 0.37%-2% feat_oob range, and the
half-life/clamp_frac/frozen-sigma decomposition already converges stably
within that range (this was verified the same way for FD003); (b) Part
B's three new deterministic degradation types already only require arm
C's absolute-scale levels, so Part A's own baseline on the same scale can
be plotted directly against Part B's Gaussian-noise baseline on the same
dose-response chart, without needing to compute the main SNR arm again.
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
        full_scale = V4.sensor_only_scale(feat_cols, full_scale)
        scalers_by_seed = scalers_for_ds(ds)

        result = E.run_df_perturb_sweep(
            ds, 'Transformer', 'armC_gaussian', V4.inject_noise_fixed_pct_raw, PCT_LEVELS,
            is_pct=True, device=device, scalers_by_seed=scalers_by_seed, full_scale=full_scale)

        all_out[ds] = result
        with open(out_path, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)
        print(f"  [{ds}] done in {time.time() - t0:.1f}s, saved -> {out_path}")

    print("\nT2-A4 (Transformer armC sweep) complete.")
