"""
Raw four-combination attribution values, for supplementary_attribution.tex.
Reuses each cell's stored (kind, level_key, arm_label) from the existing
frozen_decomposition_2x2.json as-is (guaranteeing exact agreement with the
published numbers, not re-locating the crossover); the only addition is
recording C00/C10/C01/C11 per seed, so the cross-seed std can be computed
(the earlier script only saved the aggregated mean, not each seed's own
value).
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
from interval_score import get_cp_q

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CALIB_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'calibration_and_controls')
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')
os.makedirs(ATTR_DECISION_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
METHODS = ['NLL', 'CP_norm']


def picp_z(y_true, mu, sigma, z):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def infer_pooled_per_seed(backbone, ds, level_kind, level_key, arm_label, device, scalers_by_seed,
                           full_scale, global_std, km=None, cond_std=None):
    """Identical to attribution_frozen_2x2_nearest_grid.py's infer_pooled verbatim,
    the only difference is returning a per-seed (mu, sigma) list instead of a
    stacked array (to make it easy to compute C values per seed)."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    per_seed = {}
    y_ref = None
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
        per_seed[seed] = (mu, sigma)
        y_ref = y_test
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return per_seed, y_ref


def resolve_arm_label(backbone, ds, kind, level_key, actual_fo):
    """grid_point_used never stored arm_label (an omission in the earlier script);
    this looks up whether A_percondition(per_condition) or B_pooled(global) was
    hit by reverse-matching the feat_oob value -- FD001/FD003's single operating
    condition is always global; the armc kind does not need an arm_label."""
    if kind != 'mainarm' or ds not in ('FD002', 'FD004'):
        return 'global'
    if backbone == 'LSTM':
        mainfile = json.load(open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree.json')))[ds]
    else:
        mainfile = json.load(open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_mainarm_sweep_leakfree.json')))[ds]
    fo_a = mainfile['A_percondition']['feat_oob'].get(level_key)
    fo_b = mainfile['B_pooled']['feat_oob'].get(level_key)
    if fo_a is not None and abs(fo_a - actual_fo) < 1e-9:
        return 'per_condition'
    if fo_b is not None and abs(fo_b - actual_fo) < 1e-9:
        return 'global'
    raise ValueError(f"Could not resolve arm_label for {backbone}/{ds}/{kind}/{level_key}/{actual_fo}")


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(CALIB_DIR, 'frozen_decomposition_2x2.json')) as f:
        existing = json.load(f)
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

            clean_per_seed, y0 = infer_pooled_per_seed(backbone, ds, 'clean', None, None, device,
                                                        scalers_by_seed, full_scale, global_std)

            if backbone == 'LSTM' and ds == 'FD003':
                cp_list = json.load(open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')))
            elif backbone == 'LSTM':
                cp_list = json.load(open(os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')))[ds]
            else:
                cp_list = json.load(open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')))[ds]
            cp_by_seed = {str(r['seed']): r for r in cp_list}

            result[backbone][ds] = {}
            for method in METHODS:
                cell = existing[backbone][ds][method]
                if 'note' in cell:
                    result[backbone][ds][method] = {'note': cell['note']}
                    continue
                g = cell['grid_point_used']
                arm_label = resolve_arm_label(backbone, ds, g['kind'], g['level_key'], g['actual_feat_oob'])
                pert_per_seed, y1 = infer_pooled_per_seed(backbone, ds, g['kind'], g['level_key'], arm_label,
                                                           device, scalers_by_seed, full_scale, global_std,
                                                           km=km, cond_std=cond_std)
                assert np.allclose(y0, y1)
                y = y0

                c00_s, c10_s, c01_s, c11_s = [], [], [], []
                for seed in C.SEEDS:
                    mu0, sigma0 = clean_per_seed[seed]
                    mu1, sigma1 = pert_per_seed[seed]
                    if method == 'NLL':
                        z = C.Z_SCORE
                    else:
                        z = cp_by_seed[str(seed)]['cp_norm']['q']
                    c00_s.append(picp_z(y, mu0, sigma0, z))
                    c10_s.append(picp_z(y, mu1, sigma0, z))
                    c01_s.append(picp_z(y, mu0, sigma1, z))
                    c11_s.append(picp_z(y, mu1, sigma1, z))

                c00_s = np.array(c00_s); c10_s = np.array(c10_s); c01_s = np.array(c01_s); c11_s = np.array(c11_s)
                result[backbone][ds][method] = {
                    'grid_point_used': {**g, 'resolved_arm_label': arm_label},
                    'C00_per_seed': c00_s.tolist(), 'C10_per_seed': c10_s.tolist(),
                    'C01_per_seed': c01_s.tolist(), 'C11_per_seed': c11_s.tolist(),
                    'C00_mean': float(c00_s.mean()), 'C00_std': float(c00_s.std(ddof=1)),
                    'C10_mean': float(c10_s.mean()), 'C10_std': float(c10_s.std(ddof=1)),
                    'C01_mean': float(c01_s.mean()), 'C01_std': float(c01_s.std(ddof=1)),
                    'C11_mean': float(c11_s.mean()), 'C11_std': float(c11_s.std(ddof=1)),
                    'percentage_computation': 'aggregate-then-difference: C00/C10/C01/C11 are seed-averaged '
                                               'PICP first, effects/interaction computed from those averaged '
                                               'values (not per-seed ratio averaged after).',
                }
                cc = result[backbone][ds][method]
                print(f"  [{method}] C00={cc['C00_mean']:.4f}(std={cc['C00_std']:.4f}) "
                      f"C10={cc['C10_mean']:.4f}(std={cc['C10_std']:.4f}) "
                      f"C01={cc['C01_mean']:.4f}(std={cc['C01_std']:.4f}) "
                      f"C11={cc['C11_mean']:.4f}(std={cc['C11_std']:.4f})")

    out_path = os.path.join(ATTR_DECISION_DIR, 'B_attribution_raw_per_seed.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
