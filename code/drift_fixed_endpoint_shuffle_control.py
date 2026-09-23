"""
Regenerates the D1 fixed-endpoint-shuffle discriminability control. The
original script is no longer in code/ (presumably an inline one-off
script that was never archived -- the same situation as
relative_half_life_feat_oob.json); this reimplements it using
`noise_injection.inject_drift_fixed_endpoint_shuffle_windows`, matching
the existing structure of `intermediate/attribution_and_decision/D1_fixed_endpoint_shuffle.json`
({backbone:{k:PICP}}, k in {1,3,5}, FD002, 5% FS, heteroscedastic/NLL only),
with noise shared across the 5 trials (new stable_seed tag
'r5_d1_shuffle').
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
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')

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
    full_scale = V4.sensor_only_scale(feat_cols, full_scale)
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

    out_path = os.path.join(ATTR_DECISION_DIR, 'D1_fixed_endpoint_shuffle.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
