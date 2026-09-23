"""
T2-B: three deterministic degradation types (bias/drift/gain) x two
backbones (LSTM/Transformer) x four datasets, arm-C absolute scale
({0.1,0.5,1,2,5}% FS, condition-independent, injected into the raw
readings before normalization, models not retrained, 5-model x 5-trial
shared-stream design), output goes to leakfree_t2/.

The LSTM side reuses existing checkpoints under checkpoints_leakfree/ (not
retrained, just re-running inference under the new perturbation type); the
Transformer side reuses checkpoints from T2 training.

Saves incrementally at the (backbone, degradation, dataset) granularity
with resume-on-restart support, so any interruption (e.g. a system
restart) only loses the one (backbone,degradation,dataset) combination
currently in progress, without affecting already-completed ones.
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
    full_scale = V4.sensor_only_scale(feat_cols, full_scale)
    scalers_by_seed = PA.scalers_for_ds(ds)

    if degradation == 'bias':
        return E.run_df_perturb_sweep(ds, backbone, 'bias', V4.inject_bias_fixed_pct_raw, PCT_LEVELS,
                                       is_pct=True, device=device, scalers_by_seed=scalers_by_seed,
                                       full_scale=full_scale)
    elif degradation == 'gain':
        # gain's multiplier doesn't depend on full_scale_range, so
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
