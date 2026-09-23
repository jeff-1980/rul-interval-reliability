"""
STEP 5j: feat_oob dose-response summary for the leakfree data + frozen-sigma
counterfactual decomposition.

Supersedes an earlier pooled dose-response script (not included;
superseded) (pooled points + half-life crossover) and an earlier
MSE-proxy decomposition script (not included; superseded, see
results/superseded/) (MSE-proxy decomposition, rejected by the user).

The decomposition no longer relies on a second model (MSE) as a "pure
mu_hat" proxy -- it uses the same NLL/CP-norm model's PICP curve directly
under the frozen-sigma counterfactual (`NLL_frozen_sigma` /
`CP_norm_frozen_sigma`, already computed by STEP5's full leakfree rerun,
paired sample-for-sample; see that script's docstring), because the real
curve and the counterfactual curve share exactly the same feat_oob grid
points (the same batch of trials/levels), so the interpolation is close
to an exact match rather than a cross-model approximation.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
FIG_DIR = os.path.join(PROJ_DIR, 'figures', 'generated')
os.makedirs(FIG_DIR, exist_ok=True)

METHODS = ['NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm', 'MSE_fixed']
METHOD_LABELS = {'NLL': 'NLL', 'MC_Dropout_fixed': 'MC-Dropout', 'Deep_Ensemble': 'Deep Ensemble', 'CP_norm': 'CP-norm',
                  'MSE_fixed': 'MSE (fixed sigma)'}
DECOMPOSE_METHODS = ['NLL', 'CP_norm']


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def sigma_of(cell, method):
    return cell['sigma_mean']['mean'] if method == 'Deep_Ensemble' else cell['sigma_mean']['grand_mean']


def clamp_of(cell):
    return None


def collect_points(ds_block, method, level_keys):
    pts = []
    for k in level_keys:
        if k not in ds_block['feat_oob'] or k not in ds_block[method]:
            continue
        fo = ds_block['feat_oob'][k]
        picp = picp_of(ds_block[method][k], method)
        sig = sigma_of(ds_block[method][k], method)
        pts.append((fo, picp, sig))
    return pts


def crossover_feat_oob(points, threshold=0.80):
    pts = sorted(points, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i][0], pts[i][1]
        fo1, p1 = pts[i + 1][0], pts[i + 1][1]
        if p0 >= threshold and p1 < threshold:
            if p1 == p0:
                return fo0
            frac = (threshold - p0) / (p1 - p0)
            return fo0 + frac * (fo1 - fo0)
    if pts and pts[0][1] < threshold:
        return pts[0][0]
    return None


def interp_picp(sorted_pts, x):
    xs = [p[0] for p in sorted_pts]; ys = [p[1] for p in sorted_pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            frac = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]


if __name__ == '__main__':
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree.json')) as f:
        main_lf = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'noise_sensitivity_leakfree_armC.json')) as f:
        armC_lf = json.load(f)

    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = [str(p) for p in [0.1, 0.5, 1, 2, 5]]

    ALL_METHODS = METHODS + ['NLL_frozen_sigma', 'CP_norm_frozen_sigma']
    dose_response = {}
    for ds in C.DATASETS:
        dose_response[ds] = {}
        for method in ALL_METHODS:
            pts = []
            arm_a = main_lf[ds]['A_percondition']
            pts += collect_points(arm_a, method, snr_keys_all)
            if ds != 'FD001':
                arm_b = main_lf[ds]['B_pooled']
                pts += collect_points(arm_b, method, snr_keys_all)
            pts += collect_points(armC_lf[ds], method, pct_keys)

            seen = set(); uniq = []
            for p in pts:
                key = (round(p[0], 6), round(p[1], 6))
                if key not in seen:
                    seen.add(key); uniq.append(p)

            co = crossover_feat_oob(uniq, threshold=0.80)
            dose_response[ds][method] = {'points_feat_oob_picp_sigma': uniq, 'crossover_feat_oob_at_picp_080': co}
            print(f"{ds:6}{method:22} n_points={len(uniq):3}  crossover@0.80: {co if co is None else f'{co:.4f}'}")

    # ---- Frozen-sigma decomposition ----
    decomposition = {}
    for ds in C.DATASETS:
        decomposition[ds] = {}
        for method in DECOMPOSE_METHODS:
            frozen_key = f"{method}_frozen_sigma"
            real_pts = sorted(dose_response[ds][method]['points_feat_oob_picp_sigma'], key=lambda p: p[0])
            frozen_pts = sorted(dose_response[ds][frozen_key]['points_feat_oob_picp_sigma'], key=lambda p: p[0])
            picp_clean_real = min(real_pts, key=lambda p: p[0])[1]
            picp_clean_frozen = min(frozen_pts, key=lambda p: p[0])[1]
            co = dose_response[ds][method]['crossover_feat_oob_at_picp_080']
            if co is None:
                decomposition[ds][method] = {'note': 'never crosses PICP=0.80 in measured range'}
                continue
            picp_frozen_at_co = interp_picp(frozen_pts, co)
            delta_total = picp_clean_real - 0.80
            delta_mu_only = picp_clean_frozen - picp_frozen_at_co
            delta_sigma = delta_total - delta_mu_only
            sigma_fraction = delta_sigma / delta_total if delta_total != 0 else None
            decomposition[ds][method] = {
                'crossover_feat_oob': co,
                'picp_clean_real': picp_clean_real,
                'picp_clean_frozen_sigma_counterfactual': picp_clean_frozen,
                'picp_frozen_sigma_counterfactual_at_crossover': picp_frozen_at_co,
                'delta_picp_total': delta_total,
                'delta_picp_mu_only_frozen_sigma_counterfactual': delta_mu_only,
                'delta_picp_sigma_contribution': delta_sigma,
                'sigma_contribution_fraction': sigma_fraction,
            }
            print(f"\n{ds} {method}: crossover={co:.4f}  ΔPICP_total={delta_total:.4f}  "
                  f"ΔPICP_mu_only(frozen-σ counterfactual)={delta_mu_only:.4f}  "
                  f"ΔPICP_sigma={delta_sigma:.4f}  sigma_contribution_fraction={sigma_fraction:.3f}")

    out_path = os.path.join(RESULTS_DIR, 'dose_response_feat_oob_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(dose_response, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    out_path2 = os.path.join(RESULTS_DIR, 'frozen_sigma_decomposition_leakfree.json')
    with open(out_path2, 'w') as fp:
        json.dump(decomposition, fp, indent=2, default=float)
    print(f"Saved -> {out_path2}")

    print('dose_response_frozen_sigma complete. (Figure generation removed from this release; '
          'not needed to reproduce the JSON results above.)')
