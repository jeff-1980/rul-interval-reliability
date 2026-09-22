"""
R2-2：区间评分（Winkler interval score, alpha=0.1）与 WIS，全部方法×骨干×
数据集，clean 条件（对应主 Table 3 的干净条件行）。定义：

  IS_alpha(l,u,y) = (u-l) + (2/alpha)(l-y) if y<l
                  = (u-l) + (2/alpha)(y-u) if y>u
                  = (u-l)                  otherwise

  WIS（Bracher et al. 2021，单档区间 K=1，中位数用 mu 代替，对称正态假设下
  中位数=均值）：
    WIS = (1/(K+0.5)) * [0.5*|y-mu| + (alpha/2)*IS_alpha(l,u,y)]
        = (1/1.5) * [0.5*|y-mu| + 0.05*IS_0.1(l,u,y)]

只做推理，不重训。MC-Dropout/Ensemble/MSE 的 mu,sigma 构造与既有代价表
逐字一致（复用 sweep_engine.py 的 infer_* 函数）。
"""
import os
import json

import numpy as np
import torch

import common as C
import transformer_common as T2
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
ALPHA = 0.10


def interval_score(y, mu, sigma, z):
    lo = mu - z * sigma
    hi = mu + z * sigma
    width = hi - lo
    below = y < lo
    above = y > hi
    penalty = np.zeros_like(y)
    penalty[below] = (2.0 / ALPHA) * (lo[below] - y[below])
    penalty[above] = (2.0 / ALPHA) * (y[above] - hi[above])
    return width + penalty


def wis(y, mu, sigma, z):
    is_alpha = interval_score(y, mu, sigma, z)
    return (1.0 / 1.5) * (0.5 * np.abs(y - mu) + (ALPHA / 2.0) * is_alpha)


def get_cp_q(backbone, ds, seed):
    if backbone == 'LSTM' and ds == 'FD003':
        with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')) as f:
            lst = json.load(f)
    elif backbone == 'LSTM':
        with open(os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')) as f:
            lst = json.load(f)[ds]
    else:
        with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')) as f:
            lst = json.load(f)[ds]
    by_seed = {str(r['seed']): r for r in lst}
    return by_seed[str(seed)]['cp_norm']['q']


def get_aleatory_var(backbone, ds, seed):
    if backbone == 'LSTM' and ds == 'FD003':
        with open(os.path.join(RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json')) as f:
            lst = json.load(f)
    elif backbone == 'LSTM':
        with open(os.path.join(RESULTS_DIR, 'mcdropout_fixed_leakfree.json')) as f:
            lst = json.load(f)[ds]
    else:
        with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json')) as f:
            lst = json.load(f)[ds]
    by_seed = {str(r['seed']): r for r in lst}
    return by_seed[str(seed)]['T50']['kendall_gal_full']['aleatory_var']


if __name__ == '__main__':
    C.require_fixed_hashseed()  # R8-B5: root-caused run1-vs-run2 MD5 mismatch to missing cuDNN determinism here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)

            is_all = {m: [] for m in ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']}
            wis_all = {m: [] for m in is_all}
            nll_mu_members, nll_sigma_members = [], []
            y_ref = None

            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                train_df_lf, test_df_lf, true_ruls_lf, feat_cols_lf, scaler = C.load_and_process_leakfree(ds, fit_units)
                X_test, y_test = C.create_sequences(test_df_lf, feat_cols_lf, mode='test', true_ruls=true_ruls_lf)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_ref = y_test

                nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                mu_n, ls_n = E.infer_nll(nll_model, X_t)
                sigma_n = np.exp(ls_n) * 125.0
                nll_mu_members.append(mu_n); nll_sigma_members.append(sigma_n)
                is_all['NLL'].append(interval_score(y_test, mu_n, sigma_n, C.Z_SCORE))
                wis_all['NLL'].append(wis(y_test, mu_n, sigma_n, C.Z_SCORE))

                q_norm = get_cp_q(backbone, ds, seed)
                is_all['CP_norm'].append(interval_score(y_test, mu_n, sigma_n, q_norm))
                wis_all['CP_norm'].append(wis(y_test, mu_n, sigma_n, q_norm))

                mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
                aleatory_var = get_aleatory_var(backbone, ds, seed)
                mc_seed = C.stable_seed(ds, backbone, seed, 'r8b5_mc_dropout_terminal')
                mu_mc, sigma_mc = E.infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
                is_all['MC_Dropout_fixed'].append(interval_score(y_test, mu_mc, sigma_mc, C.Z_SCORE))
                wis_all['MC_Dropout_fixed'].append(wis(y_test, mu_mc, sigma_mc, C.Z_SCORE))

                mu_mse, sigma_mse = E.infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
                is_all['MSE_fixed'].append(interval_score(y_test, mu_mse, sigma_mse, C.Z_SCORE))
                wis_all['MSE_fixed'].append(wis(y_test, mu_mse, sigma_mse, C.Z_SCORE))

                del nll_model, mc_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            mu_mem = np.stack(nll_mu_members); sigma_mem = np.stack(nll_sigma_members)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            is_all['Deep_Ensemble'] = [interval_score(y_ref, mu_ens, sigma_ens, C.Z_SCORE)]
            wis_all['Deep_Ensemble'] = [wis(y_ref, mu_ens, sigma_ens, C.Z_SCORE)]

            result[backbone][ds] = {}
            for m in is_all:
                is_flat = np.concatenate(is_all[m])
                wis_flat = np.concatenate(wis_all[m])
                result[backbone][ds][m] = {
                    'interval_score_mean': float(is_flat.mean()), 'interval_score_std': float(is_flat.std(ddof=1)),
                    'wis_mean': float(wis_flat.mean()), 'wis_std': float(wis_flat.std(ddof=1)),
                }
                print(f"  {m:20s} IS={result[backbone][ds][m]['interval_score_mean']:.3f}  "
                      f"WIS={result[backbone][ds][m]['wis_mean']:.3f}")

    out_path = os.path.join(R2_DIR, 'interval_score_wis_clean.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
