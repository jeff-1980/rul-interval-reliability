"""
Densifies the 9.6% threshold crossover by adding left-side points. Existing
right-side points cover [2.83%, 11.83%] (0dB..-5dB); this adds two more
levels between clean(0%) and 2.83%(0dB): SNR=2dB(~1.4%), SNR=1dB(~1.9%)
-- probed empirically, since on the same main-arm SNR axis the actually
achieved values are not exactly 1%/2%; the true values are reported as-is
rather than rounded to convenient integers.

Reports "first crossover from clean" per trial: treats clean(0%,picp_clean)
as the curve's starting point, sorts it together with that trial's own 7
points (the 2 new left-side points plus the existing 0dB..-4.5dB right-side
points, excluding the rightmost -5dB to avoid skipping a possible multiple
crossing in between) by feat_oob ascending, and scans rightward from clean
to linearly interpolate the first point where it drops below
rel_threshold. If this trial's own curve has not yet dropped below the
threshold at any of the scanned new points (i.e. PICP is still >=
threshold at the leftmost new point, SNR=1dB, meaning the crossing happens
somewhere between the 1dB point and the previously nearest 0dB/2.83% point,
or further left), it is uniformly labeled "<=2.84%" (using the smallest
originally measured point's percentage as an upper-bound label, rather
than guessing a more specific number further left by extrapolation).
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
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')
R4_DIR = os.path.join(RESULTS_DIR, 'leakfree_r4')
R6_DIR = os.path.join(RESULTS_DIR, 'leakfree_r6')
os.makedirs(R6_DIR, exist_ok=True)

DS = 'FD001'
BACKBONE = 'Transformer'
N_TRIALS = V4.N_TRIALS
NEW_LEFT_LEVELS_DB = [2, 1]
ORIGINAL_MIN_FOOB_PCT = 2.84  # 0dB point, the fallback upper-bound label


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

    with open(os.path.join(R4_DIR, 'threshold_960_refinement.json')) as f:
        right = json.load(f)
    picp_clean = right['picp_clean']
    rel_threshold = right['rel_threshold']
    print(f"picp_clean={picp_clean}  rel_threshold={rel_threshold}")

    # ---- new left-side points ----
    left_points = {}
    for level in NEW_LEFT_LEVELS_DB:
        level_key = str(level)
        picp_per_trial, feat_oob_per_trial = [], []
        for t in range(N_TRIALS):
            rng = np.random.RandomState(C.stable_seed(DS, BACKBONE, 'mainarm', 'global', level_key, t))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)
            mu_mem, sigma_mem = [], []
            y_ref = None
            # see threshold_crossover_refinement.py's matching comment --
            # previously feat_oob was computed once using the first seed's
            # scaler + the full trajectory; now each of the 5 seeds computes
            # it on its own windowed X_test and the results are averaged.
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
                # V4.feat_oob is the project's only implementation, denominator restricted to sensor columns.
                fo_this_trial_per_seed.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
            fo_this_trial = float(np.mean(fo_this_trial_per_seed))
            mu_mem = np.stack(mu_mem); sigma_mem = np.stack(sigma_mem)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            picp_per_trial.append(picp_of_array(y_ref, mu_ens, sigma_ens))
            feat_oob_per_trial.append(fo_this_trial)
        left_points[level] = {'feat_oob_per_trial': feat_oob_per_trial, 'picp_per_trial': picp_per_trial,
                               'feat_oob_mean': float(np.mean(feat_oob_per_trial)),
                               'picp_grand_mean': float(np.mean(picp_per_trial))}
        print(f"  SNR={level}dB  foob={left_points[level]['feat_oob_mean']*100:.4f}%  "
              f"PICP={left_points[level]['picp_grand_mean']:.4f}")

    # ---- per-trial "first crossing from clean" ----
    right_points = right['points']  # {'0': {...}, '-1': {...}, ...}
    RIGHT_LEVELS = [0, -1, -2, -3, -3.5, -4, -4.5, -5]

    def interp_crossover_from_clean(pts_sorted, threshold):
        # pts_sorted includes (0.0, picp_clean) as the first point
        for i in range(len(pts_sorted) - 1):
            fo0, p0 = pts_sorted[i]; fo1, p1 = pts_sorted[i + 1]
            if p0 >= threshold and p1 < threshold:
                frac = 0.0 if p1 == p0 else (threshold - p0) / (p1 - p0)
                return fo0 + frac * (fo1 - fo0)
        return None  # never crosses within the tested range -> caller applies fallback label

    per_trial_results = []
    for t in range(N_TRIALS):
        pts = [(0.0, picp_clean)]
        for lv in NEW_LEFT_LEVELS_DB:
            pts.append((left_points[lv]['feat_oob_per_trial'][t], left_points[lv]['picp_per_trial'][t]))
        for lv in RIGHT_LEVELS:
            rp = right_points[str(lv)]
            pts.append((rp['feat_oob_per_trial'][t], rp['picp_per_trial'][t]))
        pts.sort(key=lambda p: p[0])
        co = interp_crossover_from_clean(pts, rel_threshold)
        if co is None:
            label = f"<={ORIGINAL_MIN_FOOB_PCT}%"
        else:
            label = f"{co*100:.4f}%"
        per_trial_results.append({'trial': t, 'crossover_pct_label': label,
                                   'crossover_frac': co, 'points_used': pts})
        print(f"  trial {t}: first crossing from clean = {label}")

    result = {
        'picp_clean': picp_clean, 'rel_threshold': rel_threshold,
        'new_left_points_db_and_actual_pct': {str(lv): left_points[lv]['feat_oob_mean'] * 100
                                               for lv in NEW_LEFT_LEVELS_DB},
        'left_points_raw': {str(lv): left_points[lv] for lv in NEW_LEFT_LEVELS_DB},
        'per_trial_first_crossing_from_clean': per_trial_results,
    }
    out_path = os.path.join(R6_DIR, 'threshold_960_leftside_extension.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
