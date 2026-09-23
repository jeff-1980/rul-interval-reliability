"""
Maintenance decision case-study analysis, inference level.

Rule: maintenance is triggered when the 90% PI lower bound
mu - z*sigma <= L, for L in {10,20,30} cycles. End-window convention (one
decision per test engine, same convention as the main table, verified
against the project's split-0 assertion).

Conditions: clean / Gaussian 1%FS (arm C) / drift 5%FS / bias 5%FS.
Mechanisms: NLL, MSE_fixed (calib-based sigma), MC_Dropout_fixed
(calib-based sigma), Deep_Ensemble, CP_norm. MSE/MC-Dropout use
calibration-set residual variance (same convention as the "fair
calibration" main table, not the training-set residual version).

Each (backbone,ds,condition,method) computes 5 seeds' (gaussian/bias also
have 5 trials) independent decisions; late/early rate is the average
across (seed,trial) cells -- matching the project-wide grand_mean
statistical convention, not averaging mu/sigma first and then deciding.

late_rate  = mean over engines[ true_RUL<=L and not triggered ]
early_rate = mean over engines[ triggered and true_RUL>L+20 ]
wasted_life = mean over "early triggered" engines of (true_RUL - L)

Cost = r * late_rate + 1 * early_rate, r = c_late/c_early in {5,20,100}.
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

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
METHODS = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
CONDITIONS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']
L_LEVELS = [10, 20, 30]
COST_RATIOS = [5, 20, 100]
N_TRIALS = V4.N_TRIALS


def sequences_for_units(df, feature_cols, unit_set):
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def batched_forward(model, X_t, batch=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            outs.append(model(X_t[i:i + batch]).cpu().numpy())
    return np.concatenate(outs).flatten()


def calib_sigma_fixed(ds, backbone, seed, canon, device):
    fit_units = canon[ds][str(seed)]['fit_units']
    calib_units = canon[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    yhat_calib = np.clip(batched_forward(mc_model, X_calib_t) * 125.0, 0, 125)
    aleatory_var = float(np.var(y_calib - yhat_calib, ddof=1))
    return aleatory_var, mc_model


def get_raw_test_window(ds, condition, trial, full_scale, rng_seed_tag):
    """Returns the raw (unscaled) end-window (n_engines,T,F) for a given
    condition/trial; None means clean (no trial-dependent variation)."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    if condition == 'clean':
        X_raw, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
        return X_raw, y_ref, feat_cols, test_df_raw, true_ruls
    X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
    if condition == 'drift5pct':
        X_raw = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, full_scale)
        return X_raw, y_ref, feat_cols, test_df_raw, true_ruls
    elif condition == 'bias5pct':
        rng = np.random.RandomState((C.stable_seed(ds, 'r2_maint_bias', trial)))
        raw_df_noisy = V4.inject_bias_fixed_pct_raw(test_df_raw, feat_cols, 5.0, rng, full_scale)
        df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_df_noisy
        X_raw, y_ref2, _ = V4.extract_raw_windows(df_tmp, feat_cols, true_ruls, mode='test')
        return X_raw, y_ref, feat_cols, test_df_raw, true_ruls
    elif condition == 'gaussian1pct':
        rng = np.random.RandomState((C.stable_seed(ds, 'r2_maint_gauss', trial)))
        raw_df_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, 1.0, rng, full_scale)
        df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_df_noisy
        X_raw, y_ref2, _ = V4.extract_raw_windows(df_tmp, feat_cols, true_ruls, mode='test')
        return X_raw, y_ref, feat_cols, test_df_raw, true_ruls
    else:
        raise ValueError(condition)


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            _, _, _, feat_cols_probe, _ = V4.load_raw_train_test_and_scaler(ds)
            train_df_raw_probe, _, _, _, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw_probe, feat_cols_probe)
            full_scale = V4.sensor_only_scale(feat_cols_probe, full_scale)

            scalers_by_seed = {}
            aleatory_var_by_seed = {}
            mc_model_by_seed = {}
            nll_model_by_seed = {}
            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                scalers_by_seed[seed] = scaler
                av, mc_model = calib_sigma_fixed(ds, backbone, seed, canon, device)
                aleatory_var_by_seed[seed] = av
                mc_model_by_seed[seed] = mc_model
                nll_model_by_seed[seed] = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)

            result[backbone][ds] = {}
            for condition in CONDITIONS:
                trials = range(N_TRIALS) if condition in ('gaussian1pct', 'bias5pct') else [0]
                # per-method accumulation across (seed, trial) cells
                cell_late = {m: {L: [] for L in L_LEVELS} for m in METHODS}
                cell_early = {m: {L: [] for L in L_LEVELS} for m in METHODS}
                cell_wasted = {m: {L: [] for L in L_LEVELS} for m in METHODS}

                nll_mu_by_seed_trial = {}
                nll_sigma_by_seed_trial = {}

                for trial in trials:
                    X_raw, y_ref, feat_cols, test_df_raw, true_ruls = get_raw_test_window(
                        ds, condition, trial, full_scale, None)

                    for seed in C.SEEDS:
                        scaler = scalers_by_seed[seed]
                        X_scaled = V4.scale_raw_windows(X_raw, scaler)
                        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                        mu_n, ls_n = E.infer_nll(nll_model_by_seed[seed], X_t)
                        sigma_n = np.exp(ls_n) * 125.0
                        nll_mu_by_seed_trial[(seed, trial)] = mu_n
                        nll_sigma_by_seed_trial[(seed, trial)] = sigma_n

                        aleatory_var = aleatory_var_by_seed[seed]
                        mc_seed = C.stable_seed(ds, backbone, condition, trial, seed, 'mc_dropout')
                        mu_mc, sigma_mc = E.infer_mc_dropout(mc_model_by_seed[seed], X_t, T=50,
                                                              aleatory_var=aleatory_var, seed=mc_seed)
                        mu_mse, sigma_mse = E.infer_mse_fixed(mc_model_by_seed[seed], X_t,
                                                               sigma_fixed=float(np.sqrt(aleatory_var)))
                        q_norm = get_cp_q(backbone, ds, seed)

                        preds = {
                            'NLL': (mu_n, sigma_n, C.Z_SCORE),
                            'MSE_fixed': (mu_mse, sigma_mse, C.Z_SCORE),
                            'MC_Dropout_fixed': (mu_mc, sigma_mc, C.Z_SCORE),
                            'CP_norm': (mu_n, sigma_n, q_norm),
                        }
                        for m, (mu, sigma, z) in preds.items():
                            lower = mu - z * sigma
                            for L in L_LEVELS:
                                triggered = lower <= L
                                late = (y_ref <= L) & (~triggered)
                                early = triggered & (y_ref > L + 20)
                                cell_late[m][L].append(float(np.mean(late)))
                                cell_early[m][L].append(float(np.mean(early)))
                                if np.any(early):
                                    cell_wasted[m][L].append(float(np.mean(y_ref[early] - L)))

                # Deep_Ensemble: combine the 5 seeds' NLL mu/sigma per trial
                for trial in trials:
                    mu_mem = np.stack([nll_mu_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    sigma_mem = np.stack([nll_sigma_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    mu_ens = mu_mem.mean(0)
                    sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
                    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
                    mu_ens = np.clip(mu_ens, 0, 125)
                    _, y_ref, _, _, _ = get_raw_test_window(ds, condition, trial, full_scale, None)
                    lower = mu_ens - C.Z_SCORE * sigma_ens
                    for L in L_LEVELS:
                        triggered = lower <= L
                        late = (y_ref <= L) & (~triggered)
                        early = triggered & (y_ref > L + 20)
                        cell_late['Deep_Ensemble'][L].append(float(np.mean(late)))
                        cell_early['Deep_Ensemble'][L].append(float(np.mean(early)))
                        if np.any(early):
                            cell_wasted['Deep_Ensemble'][L].append(float(np.mean(y_ref[early] - L)))

                result[backbone][ds][condition] = {}
                for m in METHODS:
                    result[backbone][ds][condition][m] = {}
                    for L in L_LEVELS:
                        late_rate = float(np.mean(cell_late[m][L]))
                        early_rate = float(np.mean(cell_early[m][L]))
                        wasted = float(np.mean(cell_wasted[m][L])) if cell_wasted[m][L] else None
                        costs = {str(r): r * late_rate + 1.0 * early_rate for r in COST_RATIOS}
                        result[backbone][ds][condition][m][str(L)] = {
                            'late_rate': late_rate, 'early_rate': early_rate,
                            'wasted_life_mean': wasted, 'cost_by_ratio': costs,
                        }
                print(f"  [{condition}] " + "  ".join(
                    f"{m}:L20late={result[backbone][ds][condition][m]['20']['late_rate']:.3f}" for m in METHODS))

            for seed in C.SEEDS:
                del mc_model_by_seed[seed], nll_model_by_seed[seed]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(CALIB_DIR, 'maintenance_decision.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
