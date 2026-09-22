"""
T2-A5：Transformer 骨干 per-engine 覆盖率分布，4 数据集，与
eval_lstm_fd003_per_engine_coverage.py 同一协议（全滑窗 full-trajectory
per-engine PICP，5 seed 平均），Split-CP 复用 NLL checkpoint（同
sweep_engine.py 的方法论差异说明）。
"""
import os
import json

import numpy as np
import torch
from scipy import stats

import common as C
import transformer_common as T2
import sweep_engine as E

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
Z_SCORE = 1.645

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


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


def run_dataset(ds, device):
    mc_by_seed, cp_by_seed = E.load_mc_cp_for_seed(ds, 'Transformer')

    engine_length = None
    units = None
    picp_nll_per_seed, picp_mc_per_seed, picp_cp_per_seed, picp_mse_per_seed = [], [], [], []
    ens_mu_members, ens_sigma_members = [], []

    for seed in C.SEEDS:
        fit_units = CANON[ds][str(seed)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
        X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df, feat_cols, true_ruls)
        X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)
        if engine_length is None:
            engine_length = {int(u): int(np.sum(u_full == u)) for u in np.unique(u_full)}
            units = sorted(engine_length.keys())
            print(f"  [{ds}] full-trajectory test windows: {len(y_full)}  engines: {len(units)}")

        nll_model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)
        mu, log_sigma = E.infer_nll(nll_model, X_full_t)
        sigma = np.exp(log_sigma) * 125.0
        covered = (y_full >= mu - Z_SCORE * sigma) & (y_full <= mu + Z_SCORE * sigma)
        picp_nll_per_seed.append(per_engine_picp(covered, u_full))
        ens_mu_members.append(mu); ens_sigma_members.append(sigma)

        mc_model = T2.load_checkpoint_mc_model_t2('Transformer', T2.mc_ckpt_path('Transformer', ds, seed), device)
        aleatory_var = mc_by_seed[str(seed)]['T50']['kendall_gal_full']['aleatory_var']
        mc_seed = C.stable_seed(ds, 'transformer', seed, 'r8b5_mc_dropout_full_traj')
        mu_mc, sigma_mc = E.infer_mc_dropout(mc_model, X_full_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
        covered_mc = (y_full >= mu_mc - Z_SCORE * sigma_mc) & (y_full <= mu_mc + Z_SCORE * sigma_mc)
        picp_mc_per_seed.append(per_engine_picp(covered_mc, u_full))

        mu_mse, sigma_mse = E.infer_mse_fixed(mc_model, X_full_t, sigma_fixed=float(np.sqrt(aleatory_var)))
        covered_mse = (y_full >= mu_mse - Z_SCORE * sigma_mse) & (y_full <= mu_mse + Z_SCORE * sigma_mse)
        picp_mse_per_seed.append(per_engine_picp(covered_mse, u_full))

        # CP: Transformer 复用 NLL 模型（同 sweep_engine.py 说明，不重训 SplitCP clone）
        q_norm = cp_by_seed[str(seed)]['cp_norm']['q']
        covered_cp = (y_full >= mu - q_norm * sigma) & (y_full <= mu + q_norm * sigma)
        picp_cp_per_seed.append(per_engine_picp(covered_cp, u_full))

        del nll_model, mc_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    method_results = {}
    engine_picp_nll = {u: float(np.mean([d[u] for d in picp_nll_per_seed])) for u in units}
    method_results['NLL'] = summarize(engine_picp_nll, engine_length)

    engine_picp_mc = {u: float(np.mean([d[u] for d in picp_mc_per_seed])) for u in units}
    method_results['MC_Dropout_fixed'] = summarize(engine_picp_mc, engine_length)

    mu_members = np.stack(ens_mu_members); sigma_members = np.stack(ens_sigma_members)
    mu_ens = mu_members.mean(0)
    sigma2_ens = (sigma_members ** 2 + mu_members ** 2).mean(0) - mu_ens ** 2
    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
    mu_ens = np.clip(mu_ens, 0, 125)
    covered_ens = (y_full >= mu_ens - Z_SCORE * sigma_ens) & (y_full <= mu_ens + Z_SCORE * sigma_ens)
    method_results['Deep_Ensemble'] = summarize(per_engine_picp(covered_ens, u_full), engine_length)

    engine_picp_cp = {u: float(np.mean([d[u] for d in picp_cp_per_seed])) for u in units}
    method_results['CP_norm'] = summarize(engine_picp_cp, engine_length)

    engine_picp_mse = {u: float(np.mean([d[u] for d in picp_mse_per_seed])) for u in units}
    method_results['MSE_fixed'] = summarize(engine_picp_mse, engine_length)

    for m in ['NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm', 'MSE_fixed']:
        print(f"  [{ds}][{m}] median={method_results[m]['median']:.3f} "
              f"compliance={method_results[m]['compliance_rate_ge_090']:.3f}")
    return method_results


if __name__ == '__main__':
    C.require_fixed_hashseed()  # R8-B5: root-caused run1-vs-run2 MD5 mismatch to missing cuDNN determinism here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  per-engine coverage")

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_per_engine_coverage_leakfree.json')
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_out = json.load(f)

    for ds in DATASETS:
        if ds in all_out:
            print(f"[{ds}] already done, skip")
            continue
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        all_out[ds] = run_dataset(ds, device)
        with open(out_path, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)

    print(f"\nSaved -> {out_path}")
    print("T2-A5 (Transformer per-engine) complete.")
