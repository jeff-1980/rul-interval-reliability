"""
Transformer four-combination attribution, switched to the exact interpolated
crossover point ("same method as LSTM": total loss exactly equals
picp_clean - 0.10) instead of the "nearest measured grid point" that
frozen_decomposition_2x2.json used before.

C00 (mu_clean,sigma_clean) and C10 (mu_pert,sigma_clean, the frozen-sigma
curve) already have continuous curves across the full grid (NLL/CP_norm and
NLL_frozen_sigma/CP_norm_frozen_sigma in
t2_transformer_mainarm_sweep_leakfree.json +
t2_transformer_armC_sweep_leakfree.json), so interpolation needs no new
inference.

C01 (mu_clean,sigma_pert, the frozen-mu curve) never had a continuous curve
computed before (the old frozen_decomposition_2x2.json only ran inference
once, at a single nearest grid point). This script only runs new inference
for frozen-mu PICP at the two "measured grid points" that bracket the exact
crossover, then linearly interpolates with C00/C10 using the same
interpolation weight to get the exact four-combination values at the
crossover.

The only "new computation" added is C01; C00/C10/crossover location all
reuse existing data, no retraining. Read-only output:
intermediate/attribution_and_decision/G_transformer_exact_interp_attribution.json. Does not modify
main.tex or write any tex.
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
TRANSFORMER_DIR = os.path.join(RESULTS_DIR, 'leakfree_t2')
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['NLL', 'CP_norm']
N_TRIALS = V4.N_TRIALS
SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]
PCT_LEVELS = V4.PCT_LEVELS


def picp_grand(cell):
    return cell['picp']['grand_mean']


def collect_pts(ds, mainarm_json, armc_json):
    """returns dict method -> list of (fo, picp, kind, level_key, scheme)"""
    out = {m: [] for m in ['NLL', 'CP_norm', 'NLL_frozen_sigma', 'CP_norm_frozen_sigma']}
    main_ds = mainarm_json[ds]
    schemes = [('A_percondition', 'per_condition'), ('B_pooled', 'global')]
    seen_fo_by_method = {m: set() for m in out}
    single_condition = main_ds.get('note', '').startswith('single condition')
    for arm_key, scheme in schemes:
        if arm_key not in main_ds:
            continue
        if single_condition and scheme == 'global':
            continue  # avoid double-count of the degenerate duplicate
        arm = main_ds[arm_key]
        actual_scheme = 'global' if single_condition else scheme
        for level_key, fo in arm['feat_oob'].items():
            for m in out:
                if level_key not in arm[m]:
                    continue
                picp = picp_grand(arm[m][level_key])
                out[m].append((fo, picp, 'mainarm', level_key, actual_scheme))
    armc = armc_json[ds]
    for level_key, fo in armc['feat_oob'].items():
        for m in out:
            if level_key not in armc[m]:
                continue
            picp = picp_grand(armc[m][level_key])
            out[m].append((fo, picp, 'armc', level_key, None))
    for m in out:
        out[m] = sorted(out[m], key=lambda p: p[0])
    return out


def find_crossover_bracket(real_pts, rel_threshold):
    """Mirrors frozen_sigma_relative.py's crossover(): first (fo0,p0)->(fo1,p1) with p0>=thr>p1."""
    for i in range(len(real_pts) - 1):
        fo0, p0 = real_pts[i][0], real_pts[i][1]
        fo1, p1 = real_pts[i + 1][0], real_pts[i + 1][1]
        if p0 >= rel_threshold and p1 < rel_threshold:
            frac = 0.0 if p1 == p0 else (rel_threshold - p0) / (p1 - p0)
            co = fo0 + frac * (fo1 - fo0)
            return co, frac, real_pts[i], real_pts[i + 1]
    if real_pts and real_pts[0][1] < rel_threshold:
        return real_pts[0][0], 0.0, real_pts[0], real_pts[0]
    return None, None, None, None


def interp_value(vA, vB, frac):
    return vA + frac * (vB - vA)


def clean_mu_sigma_by_seed(ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed, models_by_seed):
    mu_out, sigma_out = {}, {}
    for seed in C.SEEDS:
        nll_model, _, _ = models_by_seed[seed]
        scaler = scalers_by_seed[seed]
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        mu, ls = E.infer_nll(nll_model, X_t)
        mu_out[seed] = mu
        sigma_out[seed] = np.exp(ls) * 125.0
    return mu_out, sigma_out


def clean_mu_by_seed(ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed, models_by_seed):
    mu_out, _ = clean_mu_sigma_by_seed(ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed,
                                        models_by_seed)
    return mu_out


def mu_sigma_pert_replicates(point, ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed,
                              models_by_seed, full_scale, global_std, km, cond_std):
    """Extends sigma_pert_replicates: same rng derivation, but also keeps
    each seed's each trial's mu (previously only sigma was kept, for
    frozen-mu's C01; now C10/C11's per-seed values also need the mu from
    the same batch of perturbed inference), avoiding a second inference
    pass just to get mu (inference is fully deterministic for the same
    point/trial/seed, so the same forward pass is reused).
    Returns dict seed -> list of N_TRIALS mu arrays, dict seed -> list of N_TRIALS
    sigma arrays (both perturbed), y_ref."""
    fo, picp, kind, level_key, scheme = point
    mus_by_seed = {seed: [] for seed in C.SEEDS}
    sigmas_by_seed = {seed: [] for seed in C.SEEDS}
    y_ref = None
    for t in range(N_TRIALS):
        if kind == 'mainarm':
            level = np.inf if level_key == 'inf' else float(level_key)
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'mainarm', scheme, level_key, t)))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, scheme,
                                             global_std=global_std, km=km, cond_std=cond_std)
        else:
            level = float(level_key)
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'armC_gaussian', level_key, t)))
            raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, level, rng, full_scale)
        for seed in C.SEEDS:
            scaler = scalers_by_seed[seed]
            df_noisy, _ = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
            X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            nll_model, _, _ = models_by_seed[seed]
            mu, ls = E.infer_nll(nll_model, X_t)
            sigma = np.exp(ls) * 125.0
            mus_by_seed[seed].append(mu)
            sigmas_by_seed[seed].append(sigma)
            y_ref = y_test
    return mus_by_seed, sigmas_by_seed, y_ref


def sigma_pert_replicates(point, ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed,
                           models_by_seed, full_scale, global_std, km, cond_std):
    """Returns dict seed -> list of N_TRIALS sigma arrays (perturbed), y_ref, using the exact
    same rng derivation as sweep_engine.run_snr_sweep / run_df_perturb_sweep (armC_gaussian)."""
    _, sigmas_by_seed, y_ref = mu_sigma_pert_replicates(point, ds, backbone, device, test_df_raw, feat_cols,
                                                          true_ruls, scalers_by_seed, models_by_seed, full_scale,
                                                          global_std, km, cond_std)
    return sigmas_by_seed, y_ref


def picp_frozen_mu_grand(mu_clean_by_seed, sigmas_by_seed, y_ref, z_by_seed):
    """grand mean over 5 seeds x N_TRIALS replicates, matching grand_and_marginal_stats convention."""
    vals = []
    for seed in C.SEEDS:
        mu0 = mu_clean_by_seed[seed]
        z = z_by_seed[seed]
        for sigma1 in sigmas_by_seed[seed]:
            lo = mu0 - z * sigma1; hi = mu0 + z * sigma1
            vals.append(float(np.mean((y_ref >= lo) & (y_ref <= hi))))
    return float(np.mean(vals))


def picp1(y_true, mu, sigma, z):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def per_seed_C_at_point(mu_clean_by_seed, sigma_clean_by_seed, mus_pert_by_seed, sigmas_pert_by_seed, y_ref,
                         z_by_seed):
    """At a single bracket point, computes C00/C10/C01/C11 separately for
    each seed (C00 is constant, independent of the perturbed point; C10/C01/
    C11 are that seed's mean PICP over the N_TRIALS perturbation replicates
    -- matching grand_and_marginal_stats's model_marginal definition:
    average over trials first, giving one value per seed)."""
    out = {}
    for seed in C.SEEDS:
        mu0, sigma0 = mu_clean_by_seed[seed], sigma_clean_by_seed[seed]
        z = z_by_seed[seed]
        c00 = picp1(y_ref, mu0, sigma0, z)
        c10_t, c01_t, c11_t = [], [], []
        for mu1, sigma1 in zip(mus_pert_by_seed[seed], sigmas_pert_by_seed[seed]):
            c10_t.append(picp1(y_ref, mu1, sigma0, z))
            c01_t.append(picp1(y_ref, mu0, sigma1, z))
            c11_t.append(picp1(y_ref, mu1, sigma1, z))
        out[seed] = {'C00': c00, 'C10': float(np.mean(c10_t)), 'C01': float(np.mean(c01_t)),
                     'C11': float(np.mean(c11_t))}
    return out


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(TRANSFORMER_DIR, 't2_transformer_mainarm_sweep_leakfree.json')) as f:
        mainarm_json = json.load(f)
    with open(os.path.join(TRANSFORMER_DIR, 't2_transformer_armC_sweep_leakfree.json')) as f:
        armc_json = json.load(f)
    with open(os.path.join(TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')) as f:
        cp_all = json.load(f)
    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'intermediate', 'calibration_and_controls', 'frozen_decomposition_2x2.json')) as f:
        old_2x2 = json.load(f)

    backbone = 'Transformer'
    result = {}

    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
        full_scale = V4.sensor_only_scale(feat_cols, full_scale)
        scalers_by_seed = PA.scalers_for_ds(ds)
        if ds in ('FD002', 'FD004'):
            km, cond_std, global_std = V4.fit_condition_model(train_df_raw, feat_cols)
            cond_std = {c: V4.sensor_only_scale(feat_cols, v) for c, v in cond_std.items()}
            global_std = V4.sensor_only_scale(feat_cols, global_std)
        else:
            km, cond_std = None, None
            global_std = np.std(train_df_raw[feat_cols].values, axis=0)

        models_by_seed = {seed: E.load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
        mu_clean, sigma_clean = clean_mu_sigma_by_seed(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                         scalers_by_seed, models_by_seed)
        cp_by_seed = {str(r['seed']): r for r in cp_all[ds]}
        z_nll = {seed: C.Z_SCORE for seed in C.SEEDS}
        z_cp = {seed: cp_by_seed[str(seed)]['cp_norm']['q'] for seed in C.SEEDS}

        pts = collect_pts(ds, mainarm_json, armc_json)

        result[ds] = {}
        for method in METHODS:
            real_key = method
            frozen_key = f"{method}_frozen_sigma"
            real_pts = pts[real_key]
            frozen_pts = pts[frozen_key]
            picp_clean = real_pts[0][1]
            rel_threshold = picp_clean - 0.10
            co, frac, ptA, ptB = find_crossover_bracket(real_pts, rel_threshold)
            if co is None:
                result[ds][method] = {'note': 'never crosses relative threshold in measured range'}
                continue

            # C10: frozen-sigma curve, interpolate at fo using the frozen_pts curve directly
            # (find bracket in frozen_pts by fo, not reusing ptA/ptB indices since curves
            # may have different point order per method -- but same grid, so match by fo)
            frozen_sorted = sorted(frozen_pts, key=lambda p: p[0])
            fx = [p[0] for p in frozen_sorted]; fy = [p[1] for p in frozen_sorted]
            if co <= fx[0]:
                c10 = fy[0]
            elif co >= fx[-1]:
                c10 = fy[-1]
            else:
                c10 = None
                for i in range(len(fx) - 1):
                    if fx[i] <= co <= fx[i + 1]:
                        w = 0.0 if fx[i + 1] == fx[i] else (co - fx[i]) / (fx[i + 1] - fx[i])
                        c10 = interp_value(fy[i], fy[i + 1], w)
                        break

            c00 = picp_clean
            c11 = rel_threshold  # exact by construction

            # C01 grand-mean (frozen-mu curve) + per-seed C00/C10/C01/C11 (item 2): NEW inference
            # only at ptA, ptB (bracketing the crossover) -- one inference pass per point,
            # reused for both the grand-mean C01 and the per-seed four-way breakdown.
            z_by_seed = z_nll if method == 'NLL' else z_cp
            muA, sigA, yA = mu_sigma_pert_replicates(ptA, ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                       scalers_by_seed, models_by_seed, full_scale, global_std, km,
                                                       cond_std)
            c01_A = picp_frozen_mu_grand(mu_clean, sigA, yA, z_by_seed)
            perseed_A = per_seed_C_at_point(mu_clean, sigma_clean, muA, sigA, yA, z_by_seed)
            same_point = ptB[3] == ptA[3] and ptB[2] == ptA[2] and ptB[4] == ptA[4]
            if same_point:
                c01_B = c01_A
                perseed_B = perseed_A
            else:
                muB, sigB, yB = mu_sigma_pert_replicates(ptB, ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                           scalers_by_seed, models_by_seed, full_scale, global_std,
                                                           km, cond_std)
                c01_B = picp_frozen_mu_grand(mu_clean, sigB, yB, z_by_seed)
                perseed_B = per_seed_C_at_point(mu_clean, sigma_clean, muB, sigB, yB, z_by_seed)
            c01 = interp_value(c01_A, c01_B, frac)

            total_delta = c11 - c00
            orderA_mu = c10 - c00
            orderA_sigma = c11 - c10
            orderB_sigma = c01 - c00
            orderB_mu = c11 - c01
            interaction = orderA_mu - orderB_mu  # == orderB_sigma - orderA_sigma, up to sign convention check below

            dom_A = abs(orderA_mu) > abs(orderA_sigma)
            dom_B = abs(orderB_mu) > abs(orderB_sigma)

            # interpolate C00/C10/C01/C11 per seed between A and B (frac,
            # same weight as the grand-mean crossover) -- C00 is point-independent so A==B there.
            c00_ps, c10_ps, c01_ps, c11_ps = [], [], [], []
            for seed in C.SEEDS:
                a, b = perseed_A[seed], perseed_B[seed]
                c00_ps.append(interp_value(a['C00'], b['C00'], frac))
                c10_ps.append(interp_value(a['C10'], b['C10'], frac))
                c01_ps.append(interp_value(a['C01'], b['C01'], frac))
                c11_ps.append(interp_value(a['C11'], b['C11'], frac))
            c00_ps = np.array(c00_ps); c10_ps = np.array(c10_ps)
            c01_ps = np.array(c01_ps); c11_ps = np.array(c11_ps)
            # sanity check: mean-of-per-seed vs grand-mean should be close (different aggregation
            # order -- mean-of-means vs a single pooled mean -- small numerical drift is expected,
            # not a bug; report if it drifts more than a coarse tolerance instead of hiding it).
            perseed_vs_grand_gap = {
                'C00': float(c00_ps.mean() - c00), 'C10': float(c10_ps.mean() - c10),
                'C01': float(c01_ps.mean() - c01), 'C11': float(c11_ps.mean() - c11),
            }
            max_gap = max(abs(v) for v in perseed_vs_grand_gap.values())
            if max_gap > 0.01:
                print(f"  [WARN] {ds}/{method}: per-seed-mean vs grand-mean gap up to {max_gap:.4f} "
                      f"(expected small from aggregation-order only, this looks large)")

            old_cell = old_2x2[backbone][ds][method]
            old_g = old_cell.get('grid_point_used', {})

            result[ds][method] = {
                'crossover_feat_oob_exact': co,
                'bracket_points': {
                    'A': {'kind': ptA[2], 'level_key': ptA[3], 'scheme': ptA[4], 'feat_oob': ptA[0], 'picp': ptA[1]},
                    'B': {'kind': ptB[2], 'level_key': ptB[3], 'scheme': ptB[4], 'feat_oob': ptB[0], 'picp': ptB[1]},
                    'interp_frac_toward_B': frac,
                },
                'C00': c00, 'C10': c10, 'C01': c01, 'C11': c11,
                'total_delta': total_delta,
                'order_A_freeze_sigma_first': {'mu_effect': orderA_mu, 'sigma_effect': orderA_sigma},
                'order_B_freeze_mu_first': {'sigma_effect': orderB_sigma, 'mu_effect': orderB_mu},
                'interaction': interaction,
                'mean_dominant_order_A': dom_A,
                'mean_dominant_order_B': dom_B,
                'mean_dominant_both_orders': dom_A and dom_B,
                'C00_per_seed': c00_ps.tolist(), 'C10_per_seed': c10_ps.tolist(),
                'C01_per_seed': c01_ps.tolist(), 'C11_per_seed': c11_ps.tolist(),
                'per_seed_seed_order': list(C.SEEDS),
                'per_seed_vs_grand_mean_gap': perseed_vs_grand_gap,
                'per_seed_definition': "C_ij interpolated per seed between bracket points A and B using the same "
                                        "frac as the grand-mean crossover; C10/C01/C11 at each bracket point are "
                                        "the seed's own mean over N_TRIALS perturbation replicates (model_marginal "
                                        "convention, same as grand_and_marginal_stats); C00 is point-independent.",
                'old_nearest_grid_point_version': {
                    'actual_feat_oob': old_g.get('actual_feat_oob'),
                    'C00': old_cell.get('C_mu0_sigma0'), 'C10': old_cell.get('C_mu1_sigma0'),
                    'C01': old_cell.get('C_mu0_sigma1'), 'C11': old_cell.get('C_mu1_sigma1'),
                    'mean_dominant_both_orders': old_cell.get('mean_dominant_both_orders'),
                },
            }
            r = result[ds][method]
            print(f"  [{method}] co={co*100:.4f}%  C00={c00:.4f} C10={c10:.4f} C01={c01:.4f} C11={c11:.4f}  "
                  f"domA={dom_A} domB={dom_B}  (old domBoth={old_cell.get('mean_dominant_both_orders')})")

        del models_by_seed
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_path = os.path.join(ATTR_DECISION_DIR, 'G_transformer_exact_interp_attribution.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
