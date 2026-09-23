"""
Densified verification of the 9.6% crossover threshold. Transformer /
FD001 / Deep_Ensemble's relative half-life crossover point (linear
interpolation between the two existing points 0dB->2.846% and
-5dB->11.86% gives 9.607%; a sparse grid check already confirmed this is
not an interpolation artifact) -- this adds 6 new main-arm Gaussian-SNR
levels {-1,-2,-3,-3.5,-4,-4.5} dB inside the [2.85%, 11.86%] bracket
(the armC fixed-%FS scheme only reaches this feat_oob range at tens of
%FS, which is not densifying "the same axis", so it was dropped in favor
of levels on the same main-arm SNR axis as the two existing bracket
points -- a genuine "densification within the bracket"), with the same 5
trials sharing noise (identical rng tag ('mainarm','global',level_key,t)
as the main-arm run_snr_sweep, exactly reproducing the two existing
points' historical values).

Also done: beyond the grand-mean PICP, per-trial Ensemble PICP is tracked
separately (the 5-seed moment-matching ensemble, one PICP-vs-feat_oob
curve per trial), and interpolating each of the 5 trial-specific curves
at the same rel_threshold=picp_clean-0.10 gives 5 crossover points, whose
range is reported -- this measures the sensitivity of the "9.6%" figure
to the noise realization itself, not just to grid density (which was
already verified separately).

Inference only, no retraining. Output:
results/generated/intermediate/partition_and_rul_correction/threshold_960_refinement.json
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
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')
PARTITION_RUL_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'partition_and_rul_correction')
os.makedirs(PARTITION_RUL_DIR, exist_ok=True)

DS = 'FD001'
BACKBONE = 'Transformer'
N_TRIALS = V4.N_TRIALS  # 5
NEW_LEVELS_DB = [-1, -2, -3, -3.5, -4, -4.5]
ANCHOR_LEVELS_DB = [0, -5]  # existing bracket points, recomputed per-trial for a consistent curve


def picp_of_array(y, mu, sigma, z=C.Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    global_std = np.std(train_df_raw[feat_cols].values, axis=0)
    scalers_by_seed = PA.scalers_for_ds(DS)

    models_by_seed = {seed: E.load_models_for_seed(DS, BACKBONE, seed, device) for seed in C.SEEDS}

    with open(os.path.join(ATTR_DECISION_DIR, 'D3_refined_grid_transformer_fd001_ensemble.json')) as f:
        d3 = json.load(f)
    picp_clean = d3['picp_clean']
    rel_threshold = d3['rel_threshold']
    print(f"picp_clean={picp_clean}  rel_threshold={rel_threshold}")

    all_levels = ANCHOR_LEVELS_DB + NEW_LEVELS_DB
    points = {}  # level_db -> {'feat_oob': float, 'picp_grand': float, 'picp_per_trial': [5 floats]}

    for level in all_levels:
        level_key = str(level)
        picp_per_trial = []
        feat_oob_per_trial = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(DS, BACKBONE, 'mainarm', 'global', level_key, t)))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)

            mu_mem, sigma_mem = [], []
            y_ref = None
            # Previously this used only C.SEEDS[0]'s scaler to compute
            # fo_this_trial, on the whole-trajectory scaled_feat returned
            # by scale_and_package -- the same class of bug fixed in the
            # main sweep scripts (sweep_engine.py etc.), just missed here.
            # Fixed to compute per-seed on the windowed X_test and average,
            # matching PICP's own 5-model aggregation convention.
            fo_this_trial_per_seed = []
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                nll_model, _, _ = models_by_seed[seed]
                mu, ls = E.infer_nll(nll_model, X_t)
                sigma = np.exp(ls) * 125.0
                mu_mem.append(mu); sigma_mem.append(sigma)
                y_ref = y_test
                # V4.feat_oob is the project's single implementation; denominator restricted to sensor columns.
                fo_this_trial_per_seed.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
            fo_this_trial = float(np.mean(fo_this_trial_per_seed))

            mu_mem = np.stack(mu_mem); sigma_mem = np.stack(sigma_mem)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            picp_t = picp_of_array(y_ref, mu_ens, sigma_ens)
            picp_per_trial.append(picp_t)
            feat_oob_per_trial.append(fo_this_trial)

        points[level] = {
            'feat_oob_mean': float(np.mean(feat_oob_per_trial)),
            'feat_oob_per_trial': feat_oob_per_trial,
            'picp_grand_mean': float(np.mean(picp_per_trial)),
            'picp_per_trial': picp_per_trial,
        }
        print(f"  SNR={level:>5}dB  foob={points[level]['feat_oob_mean']*100:.4f}%  "
              f"PICP(grand-mean over 5 trials)={points[level]['picp_grand_mean']:.4f}  "
              f"per-trial={[f'{p:.4f}' for p in picp_per_trial]}")

    # ---- grand-mean curve crossover (uses feat_oob_mean, picp_grand_mean per point) ----
    grand_pts = sorted([(points[lv]['feat_oob_mean'], points[lv]['picp_grand_mean'], lv) for lv in all_levels])

    def interp_crossover(pts_sorted, threshold):
        for i in range(len(pts_sorted) - 1):
            fo0, p0, _ = pts_sorted[i]; fo1, p1, _ = pts_sorted[i + 1]
            if p0 >= threshold and p1 < threshold:
                frac = 0.0 if p1 == p0 else (threshold - p0) / (p1 - p0)
                return fo0 + frac * (fo1 - fo0), (pts_sorted[i], pts_sorted[i + 1])
        return None, None

    co_grand, bracket_grand = interp_crossover(grand_pts, rel_threshold)
    print(f"\nRefined grand-mean crossover: {co_grand*100:.4f}%  "
          f"(bracket: {bracket_grand[0][2]}dB@{bracket_grand[0][0]*100:.4f}% -- "
          f"{bracket_grand[1][2]}dB@{bracket_grand[1][0]*100:.4f}%)")

    # ---- per-trial curve crossover: 5 separate curves using that trial's own feat_oob/PICP ----
    per_trial_crossovers = []
    for t in range(N_TRIALS):
        pts_t = sorted([(points[lv]['feat_oob_per_trial'][t], points[lv]['picp_per_trial'][t], lv)
                         for lv in all_levels])
        co_t, bracket_t = interp_crossover(pts_t, rel_threshold)
        per_trial_crossovers.append(co_t)
        print(f"  trial {t}: crossover={co_t*100:.4f}%" if co_t is not None else f"  trial {t}: never crosses")

    valid = [c for c in per_trial_crossovers if c is not None]
    result = {
        'picp_clean': picp_clean, 'rel_threshold': rel_threshold,
        'points': {str(lv): points[lv] for lv in all_levels},
        'refined_crossover_grand_mean': co_grand,
        'per_trial_crossovers': per_trial_crossovers,
        'per_trial_crossover_range_pct': {
            'min': float(min(valid) * 100) if valid else None,
            'max': float(max(valid) * 100) if valid else None,
            'spread_pct_points': float((max(valid) - min(valid)) * 100) if valid else None,
        },
        'prior_D3_refined_crossover': d3['refined_crossover_feat_oob'],
    }

    out_path = os.path.join(PARTITION_RUL_DIR, 'threshold_960_refinement.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
