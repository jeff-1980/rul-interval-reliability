"""
Computes the clamp_frac baseline, absolute/relative half-life f_oob, and
NLL/CP-norm frozen-sigma decomposition -- all three are core data needed
for Part A's two questions, matching the LSTM side's
dose_response_frozen_sigma.py / clamp_frac.py logic.

Correction: half-life was originally computed using only arm C (5 levels,
feat_oob range 5.6%-16%), but the LSTM side computes half-life by pooling
the main SNR arm (9-11 levels, feat_oob as low as <0.1%) with arm C before
taking the crossover -- arm C's own smallest level (feat_oob around 5.6%)
already exceeds the <2% range where most of the LSTM side's half-life
points fall, so arm C alone can't measure it and isn't the same feat_oob
coverage as LSTM. After finding this, the main SNR arm was added
(run_sweep_noise_transformer_mainarm.py), and this script now pools the
main arm (A_percondition+B_pooled) with arm C's points before computing
the crossover, matching the LSTM side's convention.

clamp_frac's definition matches the LSTM side: under clean (approximately
clean, using the main arm's 'inf' level, genuinely noise-free) input, the
fraction of samples whose log sigma_hat lands near the architecture floor
(log_sigma_min=-3.0) (<= min+CLAMP_EPS).
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
CLAMP_EPS = V4.CLAMP_EPS


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def sigma_of(cell, method):
    return cell['sigma_mean']['mean'] if method == 'Deep_Ensemble' else cell['sigma_mean']['grand_mean']


def crossover(pts, threshold):
    """pts: [(feat_oob, picp), ...]; piecewise-linear interpolation to find
    the feat_oob at which picp first drops below threshold."""
    pts = sorted(pts, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i]
        fo1, p1 = pts[i + 1]
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


METHODS = ['NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm', 'MSE_fixed']
DECOMPOSE_METHODS = ['NLL', 'CP_norm']


def collect_points(block, method, level_keys):
    pts = []
    for k in level_keys:
        if k not in block.get('feat_oob', {}) or k not in block.get(method, {}):
            continue
        pts.append((block['feat_oob'][k], picp_of(block[method][k], method)))
    return pts


def pooled_points(ds, method, armc_block, mainarm_ds_block):
    """Pools points from three sources -- main SNR arm (A_percondition +
    B_pooled) and arm C -- deduplicated (by (feat_oob,picp) rounded to 6
    decimals), matching the LSTM side's dose_response_frozen_sigma.py
    collect_points/dedup logic."""
    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = ['0.1', '0.5', '1', '2', '5']
    pts = []
    pts += collect_points(mainarm_ds_block['A_percondition'], method, snr_keys_all)
    pts += collect_points(mainarm_ds_block['B_pooled'], method, snr_keys_all)
    pts += collect_points(armc_block, method, pct_keys)
    seen = set(); uniq = []
    for p in pts:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq


def compute_half_life(ds, armc_block, mainarm_ds_block):
    """Returns {method: {picp_clean, abs_co, rel_co}}: abs threshold fixed
    at 0.80, rel threshold = picp_clean - 0.10 (matching
    relative_half_life_feat_oob.json's definition), computed on the pooled
    main-arm + arm-C points, the same coverage range as the LSTM side."""
    out = {}
    for method in METHODS:
        pts = sorted(pooled_points(ds, method, armc_block, mainarm_ds_block), key=lambda p: p[0])
        picp_clean = pts[0][1]
        abs_co = crossover(pts, 0.80)
        rel_co = crossover(pts, picp_clean - 0.10)
        out[method] = {'picp_clean': picp_clean, 'abs_co': abs_co, 'rel_co': rel_co}
    return out


def compute_clamp_frac(ds, device):
    """Under arm C's smallest level (0.1% FS, approximately clean), the
    fraction of NLL/CP-norm samples whose log sigma_hat lands near the
    architecture floor, pooled over 5 seeds. Transformer log_sigma_min =
    T2.T2_LOG_SIGMA_MIN."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)
    all_logsigma_nll = []
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_clean_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)
        with torch.no_grad():
            _, ls = model(X_clean_t)
        all_logsigma_nll.append(ls.cpu().numpy().flatten())
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    all_logsigma_nll = np.concatenate(all_logsigma_nll)
    clamp_frac = float(np.mean(all_logsigma_nll <= (T2.T2_LOG_SIGMA_MIN + CLAMP_EPS)))
    return clamp_frac


def compute_frozen_sigma_decomposition(ds, armc_block, mainarm_ds_block, half_life):
    decomposition = {}
    for method in DECOMPOSE_METHODS:
        frozen_key = f"{method}_frozen_sigma"
        real_pts = sorted(pooled_points(ds, method, armc_block, mainarm_ds_block), key=lambda p: p[0])
        frozen_pts = sorted(pooled_points(ds, frozen_key, armc_block, mainarm_ds_block), key=lambda p: p[0])
        co = half_life[method]['rel_co']
        if co is None:
            decomposition[method] = {'note': 'never crosses relative threshold in measured range'}
            continue
        picp_clean_real = real_pts[0][1]
        picp_clean_frozen = frozen_pts[0][1]
        picp_frozen_at_co = interp_picp(frozen_pts, co)
        threshold = half_life[method]['picp_clean'] - 0.10
        delta_total = picp_clean_real - threshold
        delta_mu_only = picp_clean_frozen - picp_frozen_at_co
        delta_sigma = delta_total - delta_mu_only
        sigma_fraction = delta_sigma / delta_total if delta_total != 0 else None
        decomposition[method] = {
            'crossover_feat_oob_relative': co,
            'picp_clean_real': picp_clean_real,
            'picp_clean_frozen_sigma_counterfactual': picp_clean_frozen,
            'picp_frozen_sigma_counterfactual_at_crossover': picp_frozen_at_co,
            'delta_picp_total': delta_total,
            'delta_picp_mu_only_frozen_sigma_counterfactual': delta_mu_only,
            'delta_picp_sigma_contribution': delta_sigma,
            'sigma_contribution_fraction': sigma_fraction,
        }
    return decomposition


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  mechanism extraction (half-life/clamp_frac/frozen-sigma)")

    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_armC_sweep_leakfree.json')) as f:
        armc = json.load(f)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_mainarm_sweep_leakfree.json')) as f:
        mainarm = json.load(f)

    half_life_all, clamp_frac_all, decomposition_all = {}, {}, {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        half_life_all[ds] = compute_half_life(ds, armc[ds], mainarm[ds])
        for m, v in half_life_all[ds].items():
            print(f"  {m:20s} picp_clean={v['picp_clean']:.4f}  abs_co={v['abs_co']}  rel_co={v['rel_co']}")
        clamp_frac_all[ds] = compute_clamp_frac(ds, device)
        print(f"  baseline clamp_frac (NLL, ~clean) = {clamp_frac_all[ds]:.4f}")
        decomposition_all[ds] = compute_frozen_sigma_decomposition(ds, armc[ds], mainarm[ds], half_life_all[ds])
        for m, v in decomposition_all[ds].items():
            if 'sigma_contribution_fraction' in v:
                print(f"  [{m}] sigma_contribution_fraction={v['sigma_contribution_fraction']}")

    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_half_life_feat_oob.json'), 'w') as fp:
        json.dump(half_life_all, fp, indent=2, default=float)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_clamp_frac.json'), 'w') as fp:
        json.dump(clamp_frac_all, fp, indent=2, default=float)
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_frozen_sigma_decomposition.json'), 'w') as fp:
        json.dump(decomposition_all, fp, indent=2, default=float)
    print("\nSaved -> t2_transformer_half_life_feat_oob.json, t2_transformer_clamp_frac.json, "
          "t2_transformer_frozen_sigma_decomposition.json")
    print("T2-A6 complete.")
