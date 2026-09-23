"""
Fair-calibration main table. MSE-fixed and MC-Dropout's sigma switches to
calibration-set (calib_units, the same engines as Split-CP) residual
estimation, regenerating clean-condition PICP/MPIW/ECE/per-engine + IS/WIS
for both backbones x four datasets, alongside the training-residual
version. NLL/Deep_Ensemble/CP-norm are unaffected (sigma doesn't come from
train-fit residuals), reusing interval_score.py's existing results as-is.

The main table now uses this (calibration-set) version as the reference;
the training-residual version is kept for comparison, not deleted.
"""
import os
import json

import numpy as np
import torch

import common as C
import transformer_common as T2
import sweep_engine as E
from interval_score import interval_score, wis, get_cp_q

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
Z_SCORE = C.Z_SCORE


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


def per_engine_picp(covered, units):
    return {int(u): float(np.mean(covered[units == u])) for u in np.unique(units)}


def summarize_compliance(engine_picp_dict):
    vals = np.array(list(engine_picp_dict.values()))
    return float(np.mean(vals >= 0.90))


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
            picp_all, mpiw_all, ece_all, is_all, wis_all = {'MSE_fixed': [], 'MC_Dropout_fixed': []}, \
                {'MSE_fixed': [], 'MC_Dropout_fixed': []}, {'MSE_fixed': [], 'MC_Dropout_fixed': []}, \
                {'MSE_fixed': [], 'MC_Dropout_fixed': []}, {'MSE_fixed': [], 'MC_Dropout_fixed': []}
            engine_picp_mse_per_seed, engine_picp_mc_per_seed = [], []

            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                calib_units = canon[ds][str(seed)]['calib_units']
                train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)

                # calib-based aleatory var (same as fair_calibration_sigma_fixed.py)
                X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)],
                                                             feat_cols, calib_units)
                y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
                X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
                mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
                yhat_calib = np.clip(batched_forward(mc_model, X_calib_t) * 125.0, 0, 125)
                aleatory_var_calib = float(np.var(y_calib - yhat_calib, ddof=1))
                sigma_fixed_calib = float(np.sqrt(aleatory_var_calib))

                # clean test-set eval (single end window)
                X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

                mc_seed = C.stable_seed(ds, backbone, seed, 'r8b5_mc_dropout_terminal')
                mu_mc, sigma_mc = E.infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var_calib, seed=mc_seed)
                p_mc, w_mc = E.picp_mpiw(y_test, mu_mc, sigma_mc, Z_SCORE)
                ece_mc = C.compute_ece(mu_mc, sigma_mc, y_test)
                picp_all['MC_Dropout_fixed'].append(p_mc); mpiw_all['MC_Dropout_fixed'].append(w_mc)
                ece_all['MC_Dropout_fixed'].append(ece_mc)
                is_all['MC_Dropout_fixed'].append(interval_score(y_test, mu_mc, sigma_mc, Z_SCORE))
                wis_all['MC_Dropout_fixed'].append(wis(y_test, mu_mc, sigma_mc, Z_SCORE))

                mu_mse, sigma_mse = E.infer_mse_fixed(mc_model, X_t, sigma_fixed=sigma_fixed_calib)
                p_mse, w_mse = E.picp_mpiw(y_test, mu_mse, sigma_mse, Z_SCORE)
                ece_mse = C.compute_ece(mu_mse, sigma_mse, y_test)
                picp_all['MSE_fixed'].append(p_mse); mpiw_all['MSE_fixed'].append(w_mse)
                ece_all['MSE_fixed'].append(ece_mse)
                is_all['MSE_fixed'].append(interval_score(y_test, mu_mse, sigma_mse, Z_SCORE))
                wis_all['MSE_fixed'].append(wis(y_test, mu_mse, sigma_mse, Z_SCORE))

                # per-engine (full trajectory), calib-based sigma
                X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df, feat_cols, true_ruls)
                X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)
                mc_seed_f = C.stable_seed(ds, backbone, seed, 'r8b5_mc_dropout_full_traj')
                mu_mc_f, sigma_mc_f = E.infer_mc_dropout(mc_model, X_full_t, T=50, aleatory_var=aleatory_var_calib, seed=mc_seed_f)
                covered_mc_f = (y_full >= mu_mc_f - Z_SCORE * sigma_mc_f) & (y_full <= mu_mc_f + Z_SCORE * sigma_mc_f)
                engine_picp_mc_per_seed.append(per_engine_picp(covered_mc_f, u_full))

                mu_mse_f, sigma_mse_f = E.infer_mse_fixed(mc_model, X_full_t, sigma_fixed=sigma_fixed_calib)
                covered_mse_f = (y_full >= mu_mse_f - Z_SCORE * sigma_mse_f) & (y_full <= mu_mse_f + Z_SCORE * sigma_mse_f)
                engine_picp_mse_per_seed.append(per_engine_picp(covered_mse_f, u_full))

                del mc_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            units = sorted(engine_picp_mse_per_seed[0].keys())
            engine_picp_mse = {u: float(np.mean([d[u] for d in engine_picp_mse_per_seed])) for u in units}
            engine_picp_mc = {u: float(np.mean([d[u] for d in engine_picp_mc_per_seed])) for u in units}

            result[backbone][ds] = {}
            for m in ['MSE_fixed', 'MC_Dropout_fixed']:
                compliance = summarize_compliance(engine_picp_mse if m == 'MSE_fixed' else engine_picp_mc)
                result[backbone][ds][m] = {
                    'picp_mean': float(np.mean(picp_all[m])), 'picp_std': float(np.std(picp_all[m], ddof=1)),
                    'mpiw_mean': float(np.mean(mpiw_all[m])), 'mpiw_std': float(np.std(mpiw_all[m], ddof=1)),
                    'ece_mean': float(np.mean(ece_all[m])), 'ece_std': float(np.std(ece_all[m], ddof=1)),
                    'per_engine_compliance': compliance,
                    'interval_score_mean': float(np.mean(np.concatenate(is_all[m]))),
                    'wis_mean': float(np.mean(np.concatenate(wis_all[m]))),
                }
                print(f"  [{m}] PICP={result[backbone][ds][m]['picp_mean']:.3f}  "
                      f"MPIW={result[backbone][ds][m]['mpiw_mean']:.2f}  "
                      f"ECE={result[backbone][ds][m]['ece_mean']:.4f}  "
                      f"per_engine={compliance:.3f}  IS={result[backbone][ds][m]['interval_score_mean']:.3f}  "
                      f"WIS={result[backbone][ds][m]['wis_mean']:.3f}")

    out_path = os.path.join(R2_DIR, 'fair_calibration_main_table.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
