"""
Frozen decomposition 2x2. For NLL/CP-norm, both backbones, four datasets,
near the relative half-life point (the nearest level to rel_co in the
existing sweep grid, main-arm preferred over armC since most crossovers
fall in main-arm's low-feat_oob range), computes four combinations:
  C(mu0,sigma0): true prediction on clean input
  C(mu1,sigma0): perturbed mu, frozen clean sigma (the existing
                 "frozen-sigma" counterfactual)
  C(mu0,sigma1): frozen clean mu, perturbed sigma (new)
  C(mu1,sigma1): true prediction on perturbed input (the original "real" curve)
Two decomposition orders:
  Order A (freeze sigma first): mu effect=C(mu1,s0)-C(mu0,s0); sigma effect=C(mu1,s1)-C(mu1,s0)
  Order B (freeze mu first):    sigma effect=C(mu0,s1)-C(mu0,s0); mu effect=C(mu1,s1)-C(mu0,s1)
  Interaction = [C(mu1,s1)-C(mu1,s0)] - [C(mu0,s1)-C(mu0,s0)]
              = mu effect(order A) - mu effect(order B) (equivalent definitions)
"Mean dominance" (|mu effect| > |sigma effect|) is only reported as holding
if it holds under both orders.

LSTM's CP-norm reuses the NLL weights (verified bit-exact), so the
per-sample mu/sigma values are the same for NLL and CP-norm, differing
only in the interval-width formula (NLL uses z=1.645; CP-norm uses that
seed's own calibrated q_norm) -- this script does one inference pass,
shared by both methods.

Inference only, no retraining. Looks up the nearest grid point primarily
in the main-arm grid (SNR levels, compared via feat_oob) since most
models' relative half-life falls in main-arm's <2% feat_oob range, while
armC's own smallest level (about 5-6%) often overshoots it.
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
CALIB_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'calibration_and_controls')
os.makedirs(CALIB_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
METHODS = ['NLL', 'CP_norm']
SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]
PCT_LEVELS = V4.PCT_LEVELS


def picp_z(y_true, mu, sigma, z):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def get_rel_co(backbone, ds, method):
    if backbone == 'LSTM':
        with open(os.path.join(RESULTS_DIR, 'leakfree', 'relative_half_life_feat_oob.json')) as f:
            d = json.load(f)
        return d[ds][method]['rel_co']
    else:
        with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_half_life_feat_oob.json')) as f:
            d = json.load(f)
        return d[ds][method]['rel_co']


def nearest_grid_point(ds, backbone, target_fo):
    """Searches both main-arm schemes (A_percondition + B_pooled -- their
    feat_oob differs on multi-condition datasets, so both must be
    candidates, or the coverage gap between the two arms could wrongly
    pick 'inf'/clean as "nearest") plus armC (pct levels) for the level
    whose feat_oob is nearest target_fo. Returns
    ('mainarm', level, arm_label, actual_feat_oob) or
    ('armc', level, None, fo). arm_label in {'global','per_condition'}
    tells infer_pooled which injection scheme to use.
    """
    candidates = []
    if backbone == 'LSTM':
        if ds == 'FD003':
            with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_FD003.json')) as f:
                mainfile = json.load(f)['FD003']
        else:
            with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree.json')) as f:
                mainfile = json.load(f)[ds]
    else:
        with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_mainarm_sweep_leakfree.json')) as f:
            mainfile = json.load(f)[ds]

    arm_keys = [('A_percondition', 'per_condition'), ('B_pooled', 'global')] if ds in ('FD002', 'FD004') \
        else [('B_pooled', 'global')]  # FD001/FD003 single-condition: A=B, use B_pooled only (no km for per_condition)
    for arm_key, arm_label in arm_keys:
        for k, fo in mainfile[arm_key]['feat_oob'].items():
            candidates.append(('mainarm', k, arm_label, fo))

    if backbone == 'LSTM':
        if ds == 'FD003':
            with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
                armc = json.load(f)['FD003']
        else:
            with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC.json')) as f:
                armc = json.load(f)[ds]
    else:
        with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_armC_sweep_leakfree.json')) as f:
            armc = json.load(f)[ds]
    for k, fo in armc['feat_oob'].items():
        candidates.append(('armc', k, None, fo))

    # Drop candidates with feat_oob approx 0 (clean/inf) -- half-life is
    # defined as "some distance from clean", so a plain nearest-value
    # comparison should not accidentally pick clean itself
    candidates = [c for c in candidates if c[3] > 1e-6]
    best = min(candidates, key=lambda c: abs(c[3] - target_fo) if target_fo is not None else abs(c[3]))
    return best


def infer_pooled(backbone, ds, level_kind, level_key, device, scalers_by_seed, full_scale, global_std,
                  arm_label=None, km=None, cond_std=None):
    """For a given (level_kind, level_key), pooled 5-seed inference,
    returning (mu1, sigma1, y) -- mu/sigma under perturbed input (clean
    goes through the same path with level_key='inf' or the matching
    smallest level, just without noise injection). arm_label='per_condition'
    needs km/cond_std (arm A's scheme for multi-condition datasets)."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    all_mu, all_sigma, y_ref = [], [], None
    for seed in C.SEEDS:
        model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
        scaler = scalers_by_seed[seed]
        if level_kind == 'clean':
            raw_vals = test_df_raw[feat_cols].values.astype(np.float64)
        elif level_kind == 'mainarm':
            level = np.inf if level_key == 'inf' else float(level_key)
            scheme = arm_label or 'global'
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'r2_2x2_mainarm', scheme, level_key)))
            raw_vals = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, scheme,
                                            global_std=global_std, km=km, cond_std=cond_std)
        elif level_kind == 'armc':
            level = float(level_key)
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'r2_2x2_armc', level_key)))
            raw_vals = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, level, rng, full_scale)
        else:
            raise ValueError(level_kind)
        df_noisy, _ = V4.scale_and_package(test_df_raw, feat_cols, raw_vals, scaler)
        X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        mu, ls = E.infer_nll(model, X_t)
        sigma = np.exp(ls) * 125.0
        all_mu.append(mu); all_sigma.append(sigma)
        y_ref = y_test
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.array(all_mu), np.array(all_sigma), y_ref  # (5, n_test), (5, n_test), (n_test,)


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    def scalers_for(ds):
        s = {}
        for seed in C.SEEDS:
            fit_units = canon[ds][str(seed)]['fit_units']
            _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
            s[seed] = scaler
        return s

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
            full_scale = V4.sensor_only_scale(feat_cols, full_scale)
            global_std = np.std(train_df_raw[feat_cols].values, axis=0)
            scalers_by_seed = scalers_for(ds)
            if ds in ('FD002', 'FD004'):
                km, cond_std, _ = V4.fit_condition_model(train_df_raw, feat_cols)
                cond_std = {c: V4.sensor_only_scale(feat_cols, v) for c, v in cond_std.items()}
                global_std = V4.sensor_only_scale(feat_cols, global_std)
            else:
                km, cond_std = None, None

            mu0, sigma0, y0 = infer_pooled(backbone, ds, 'clean', None, device, scalers_by_seed, full_scale, global_std)

            if backbone == 'LSTM' and ds == 'FD003':
                with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
                    cp_list = json.load(f)
            elif backbone == 'LSTM':
                with open(os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')) as f:
                    cp_list = json.load(f)[ds]
            else:
                with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')) as f:
                    cp_list = json.load(f)[ds]
            cp_by_seed = {str(r['seed']): r for r in cp_list}

            result[backbone][ds] = {}
            for method in METHODS:
                rel_co = get_rel_co(backbone, ds, method)
                if rel_co is None:
                    result[backbone][ds][method] = {'note': 'rel_co unavailable (never crossed in measured range)'}
                    print(f"  [{method}] rel_co unavailable, skip")
                    continue
                kind, level_key, arm_label, actual_fo = nearest_grid_point(ds, backbone, rel_co)
                mu1, sigma1, y1 = infer_pooled(backbone, ds, kind, level_key, device, scalers_by_seed,
                                                full_scale, global_std, arm_label=arm_label,
                                                km=km, cond_std=cond_std)
                assert np.allclose(y0, y1)
                y = y0

                picps = {'00': [], '10': [], '01': [], '11': []}
                for i, seed in enumerate(C.SEEDS):
                    if method == 'NLL':
                        z = C.Z_SCORE
                        picps['00'].append(picp_z(y, mu0[i], sigma0[i], z))
                        picps['10'].append(picp_z(y, mu1[i], sigma0[i], z))
                        picps['01'].append(picp_z(y, mu0[i], sigma1[i], z))
                        picps['11'].append(picp_z(y, mu1[i], sigma1[i], z))
                    else:  # CP_norm
                        q = cp_by_seed[str(seed)]['cp_norm']['q']
                        picps['00'].append(picp_z(y, mu0[i], sigma0[i], q))
                        picps['10'].append(picp_z(y, mu1[i], sigma0[i], q))
                        picps['01'].append(picp_z(y, mu0[i], sigma1[i], q))
                        picps['11'].append(picp_z(y, mu1[i], sigma1[i], q))
                c00 = float(np.mean(picps['00'])); c10 = float(np.mean(picps['10']))
                c01 = float(np.mean(picps['01'])); c11 = float(np.mean(picps['11']))

                mu_effect_A = c10 - c00       # order A: freeze sigma first, mu effect
                sigma_effect_A = c11 - c10    # order A: then release sigma
                sigma_effect_B = c01 - c00    # order B: freeze mu first, sigma effect
                mu_effect_B = c11 - c01       # order B: then release mu
                interaction = mu_effect_A - mu_effect_B  # equals -(sigma_effect_A - sigma_effect_B)
                total = c11 - c00

                mean_dominant_A = abs(mu_effect_A) > abs(sigma_effect_A)
                mean_dominant_B = abs(mu_effect_B) > abs(sigma_effect_B)
                mean_dominant_both = mean_dominant_A and mean_dominant_B

                result[backbone][ds][method] = {
                    'grid_point_used': {'kind': kind, 'level_key': level_key, 'target_rel_co': rel_co,
                                         'actual_feat_oob': actual_fo},
                    'C_mu0_sigma0': c00, 'C_mu1_sigma0': c10, 'C_mu0_sigma1': c01, 'C_mu1_sigma1': c11,
                    'total_delta': total,
                    'order_A_freeze_sigma_first': {'mu_effect': mu_effect_A, 'sigma_effect': sigma_effect_A},
                    'order_B_freeze_mu_first': {'sigma_effect': sigma_effect_B, 'mu_effect': mu_effect_B},
                    'interaction': interaction,
                    'mean_dominant_order_A': mean_dominant_A,
                    'mean_dominant_order_B': mean_dominant_B,
                    'mean_dominant_both_orders': mean_dominant_both,
                }
                print(f"  [{method}] grid={kind}/{level_key} (target_rel_co={rel_co:.5f}, actual_fo={actual_fo:.5f})")
                print(f"    C00={c00:.4f} C10={c10:.4f} C01={c01:.4f} C11={c11:.4f}  total={total:+.4f}")
                print(f"    order A: mu={mu_effect_A:+.4f} sigma={sigma_effect_A:+.4f}  "
                      f"order B: sigma={sigma_effect_B:+.4f} mu={mu_effect_B:+.4f}  interaction={interaction:+.4f}")
                print(f"    mean-dominant: orderA={mean_dominant_A} orderB={mean_dominant_B} "
                      f"BOTH={mean_dominant_both}")

    out_path = os.path.join(CALIB_DIR, 'frozen_decomposition_2x2.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
