"""
FD003 completion (5/6): per-engine coverage distribution, same protocol
as eval_lstm_per_engine_coverage.py, scoped to FD003.
"""
import os
import json

import numpy as np
import torch
from scipy import stats

import common as C
import mc_dropout_model as S1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
Z_SCORE = 1.645
DS = 'FD003'

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)
with open(os.path.join(RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json')) as f:
    MC_JSON = {str(r['seed']): r for r in json.load(f)}
with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
    CP_JSON = {str(r['seed']): r for r in json.load(f)}


def per_engine_picp(covered, units):
    return {int(u): float(np.mean(covered[units == u])) for u in np.unique(units)}


def summarize(engine_picp_dict, engine_length_dict):
    vals = np.array(list(engine_picp_dict.values()))
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    units_sorted = sorted(engine_picp_dict.keys())
    picps = np.array([engine_picp_dict[u] for u in units_sorted])
    lengths = np.array([engine_length_dict[u] for u in units_sorted])
    rho, p = stats.spearmanr(picps, lengths)
    return {'median': float(med), 'iqr': float(q3 - q1), 'q1': float(q1), 'q3': float(q3),
            'compliance_rate_ge_090': float(np.mean(vals >= 0.90)), 'worst_5pct': float(np.percentile(vals, 5)),
            'spearman_picp_vs_lifelength': {'rho': float(rho), 'p': float(p)},
            'n_engines': len(vals), 'per_engine_picp': engine_picp_dict}


def infer_nll_full(model, X_t, batch=8192):
    mus, sigmas = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mu, log_sigma = model(X_t[i:i + batch])
            mus.append(mu.cpu().numpy().flatten())
            sigmas.append(torch.exp(log_sigma).cpu().numpy().flatten())
    mu = np.concatenate(mus) * 125.0
    sigma = np.concatenate(sigmas) * 125.0
    return np.clip(mu, 0, 125), sigma


def infer_mc_dropout_full(mc_model, X_t, T, aleatory_var, batch=4096, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    all_samples = []
    mc_model.train()
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            xb = X_t[i:i + batch]
            samples = [mc_model(xb).cpu().numpy().flatten() * 125.0 for _ in range(T)]
            all_samples.append(np.stack(samples))
    samples = np.concatenate(all_samples, axis=1)
    mu = np.clip(samples.mean(0), 0, 125)
    sigma = np.sqrt(aleatory_var + samples.var(0))
    return mu, sigma


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    method_results = {}
    engine_length = None
    units = None
    picp_nll_per_seed, picp_mc_per_seed, picp_cp_per_seed, picp_mse_per_seed = [], [], [], []
    ens_mu_members, ens_sigma_members = [], []

    for si, seed in enumerate(C.SEEDS):
        fit_units = CANON[DS][str(seed)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(DS, fit_units)
        X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df, feat_cols, true_ruls)
        X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)
        if engine_length is None:
            engine_length = {int(u): int(np.sum(u_full == u)) for u in np.unique(u_full)}
            units = sorted(engine_length.keys())
            print(f"full-trajectory test windows: {len(y_full)}  engines: {len(units)}")

        model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt"), device)
        mu, sigma = infer_nll_full(model, X_full_t)
        covered = (y_full >= mu - Z_SCORE * sigma) & (y_full <= mu + Z_SCORE * sigma)
        picp_nll_per_seed.append(per_engine_picp(covered, u_full))
        ens_mu_members.append(mu); ens_sigma_members.append(sigma)
        del model

        ckpt = torch.load(os.path.join(CKPT_DIR, f"{DS}_MCDropoutMSE_seed{seed}.pt"), map_location=device, weights_only=False)
        mc_model = S1.MC_LSTM(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout']).to(device)
        mc_model.load_state_dict(ckpt['state_dict'])
        aleatory_var = MC_JSON[str(seed)]['T50']['kendall_gal_full']['aleatory_var']
        mc_seed = C.stable_seed(DS, seed, 'r8b5_mc_dropout_full_traj')
        mu_mc, sigma_mc = infer_mc_dropout_full(mc_model, X_full_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
        covered_mc = (y_full >= mu_mc - Z_SCORE * sigma_mc) & (y_full <= mu_mc + Z_SCORE * sigma_mc)
        picp_mc_per_seed.append(per_engine_picp(covered_mc, u_full))

        # MSE row: same checkpoint, eval() mode (dropout off), fixed sigma=sqrt(aleatory_var)
        mc_model.eval()
        with torch.no_grad():
            mu_mse_list = []
            for i in range(0, X_full_t.shape[0], 8192):
                mu_mse_list.append(mc_model(X_full_t[i:i + 8192]).cpu().numpy().flatten())
        mu_mse = np.clip(np.concatenate(mu_mse_list) * 125.0, 0, 125)
        sigma_fixed = float(np.sqrt(aleatory_var))
        sigma_mse = np.full_like(mu_mse, sigma_fixed)
        covered_mse = (y_full >= mu_mse - Z_SCORE * sigma_mse) & (y_full <= mu_mse + Z_SCORE * sigma_mse)
        picp_mse_per_seed.append(per_engine_picp(covered_mse, u_full))
        del mc_model

        cp_model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_SplitCP_seed{seed}.pt"), device)
        mu_cp, sigma_cp = infer_nll_full(cp_model, X_full_t)
        q_norm = CP_JSON[str(seed)]['cp_norm']['q']
        covered_cp = (y_full >= mu_cp - q_norm * sigma_cp) & (y_full <= mu_cp + q_norm * sigma_cp)
        picp_cp_per_seed.append(per_engine_picp(covered_cp, u_full))
        del cp_model

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    engine_picp_nll = {u: float(np.mean([d[u] for d in picp_nll_per_seed])) for u in units}
    method_results['NLL'] = summarize(engine_picp_nll, engine_length)
    print(f"[NLL] median={method_results['NLL']['median']:.3f} compliance={method_results['NLL']['compliance_rate_ge_090']:.3f}")

    engine_picp_mc = {u: float(np.mean([d[u] for d in picp_mc_per_seed])) for u in units}
    method_results['MC_Dropout_fixed'] = summarize(engine_picp_mc, engine_length)
    print(f"[MC_Dropout_fixed] median={method_results['MC_Dropout_fixed']['median']:.3f} compliance={method_results['MC_Dropout_fixed']['compliance_rate_ge_090']:.3f}")

    mu_members = np.stack(ens_mu_members); sigma_members = np.stack(ens_sigma_members)
    mu_ens = mu_members.mean(0)
    sigma2_ens = (sigma_members ** 2 + mu_members ** 2).mean(0) - mu_ens ** 2
    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
    mu_ens = np.clip(mu_ens, 0, 125)
    covered_ens = (y_full >= mu_ens - Z_SCORE * sigma_ens) & (y_full <= mu_ens + Z_SCORE * sigma_ens)
    method_results['Deep_Ensemble'] = summarize(per_engine_picp(covered_ens, u_full), engine_length)
    print(f"[Deep_Ensemble] median={method_results['Deep_Ensemble']['median']:.3f} compliance={method_results['Deep_Ensemble']['compliance_rate_ge_090']:.3f}")

    engine_picp_cp = {u: float(np.mean([d[u] for d in picp_cp_per_seed])) for u in units}
    method_results['CP_norm'] = summarize(engine_picp_cp, engine_length)
    print(f"[CP_norm] median={method_results['CP_norm']['median']:.3f} compliance={method_results['CP_norm']['compliance_rate_ge_090']:.3f}")

    engine_picp_mse = {u: float(np.mean([d[u] for d in picp_mse_per_seed])) for u in units}
    method_results['MSE_fixed'] = summarize(engine_picp_mse, engine_length)
    print(f"[MSE_fixed] median={method_results['MSE_fixed']['median']:.3f} compliance={method_results['MSE_fixed']['compliance_rate_ge_090']:.3f}")

    out_path = os.path.join(RESULTS_DIR, 'stepFD003_per_engine_coverage_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump({DS: method_results}, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("FD003 per-engine complete.")
