"""
R8-B1（推理级，不重训）：重新评估同划分集成对照（Table III / samesplit_
ensemble.json）。5 个成员 checkpoint 全部复用已有权重（init_seed=42 的从
checkpoints_leakfree*/ 复用，其余4个从 checkpoints_leakfree_r4_samesplit/
加载）——不调用 samesplit_ensemble_control.py 的 train_lstm/
train_transformer（那两个函数会无条件重训4个新模型，不符合"不重训"）。

直接 import samesplit_ensemble_control 复用其 ckpt_path_r4 /
load_model_generic / gate_check_split 等函数（import 本身不触发训练，训练
代码全在 if __name__=='__main__': 内）。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
import samesplit_ensemble_control as M
from interval_score import interval_score, wis

if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    result = {}
    for backbone in M.BACKBONES:
        result[backbone] = {}
        for ds in M.DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            split = M.CANON[ds][str(M.SPLIT_SEED)]
            fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']
            M.gate_check_split(ds, fit_units, val_units, calib_units)

            ckpt_paths = {}
            for init_seed in M.INIT_SEEDS:
                if init_seed == M.SPLIT_SEED:
                    ckpt_paths[init_seed] = T2.nll_ckpt_path(backbone, ds, M.SPLIT_SEED)
                else:
                    ckpt_paths[init_seed] = M.ckpt_path_r4(backbone, ds, init_seed)
                assert os.path.exists(ckpt_paths[init_seed]), f"missing checkpoint: {ckpt_paths[init_seed]}"

            _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
            train_df_lf, test_df_lf, true_ruls_lf, feat_cols_lf, _ = C.load_and_process_leakfree(ds, fit_units)
            X_test, y_test = C.create_sequences(test_df_lf, feat_cols_lf, mode='test', true_ruls=true_ruls_lf)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df_lf, feat_cols_lf, true_ruls_lf)
            X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)

            mu_members, sigma_members = [], []
            mu_full_members, sigma_full_members = [], []
            single_model_rows = []
            for init_seed in M.INIT_SEEDS:
                model = M.load_model_generic(backbone, ckpt_paths[init_seed], device)
                mu, ls = E.infer_nll(model, X_test_t)
                sigma = np.exp(ls) * 125.0
                mu_members.append(mu); sigma_members.append(sigma)

                mu_f, ls_f = E.infer_nll(model, X_full_t)
                sigma_f = np.exp(ls_f) * 125.0
                mu_full_members.append(mu_f); sigma_full_members.append(sigma_f)

                picp_i, mpiw_i = C.picp_mpiw(y_test, mu, sigma, z=C.Z_SCORE)
                ece_i = C.compute_ece(mu, sigma, y_test, M.CONF_LEVELS)
                is_i = interval_score(y_test, mu, sigma, C.Z_SCORE)
                wis_i = wis(y_test, mu, sigma, C.Z_SCORE)
                single_model_rows.append({
                    'init_seed': init_seed, 'picp': picp_i, 'mpiw': mpiw_i, 'ece': ece_i,
                    'is_mean': float(is_i.mean()), 'is_std': float(is_i.std(ddof=1)),
                    'wis_mean': float(wis_i.mean()),
                })
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            is_means_across_models = np.array([r['is_mean'] for r in single_model_rows])

            mu_mem = np.stack(mu_members); sigma_mem = np.stack(sigma_members)
            mu_ens = mu_mem.mean(0)
            sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
            sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
            mu_ens = np.clip(mu_ens, 0, 125)
            picp_ens, mpiw_ens = C.picp_mpiw(y_test, mu_ens, sigma_ens, z=C.Z_SCORE)
            ece_ens = C.compute_ece(mu_ens, sigma_ens, y_test, M.CONF_LEVELS)
            is_ens = interval_score(y_test, mu_ens, sigma_ens, C.Z_SCORE)
            wis_ens = wis(y_test, mu_ens, sigma_ens, C.Z_SCORE)

            mu_full_mem = np.stack(mu_full_members); sigma_full_mem = np.stack(sigma_full_members)
            mu_full_ens = mu_full_mem.mean(0)
            sigma2_full_ens = (sigma_full_mem ** 2 + mu_full_mem ** 2).mean(0) - mu_full_ens ** 2
            sigma_full_ens = np.sqrt(np.clip(sigma2_full_ens, 0, None))
            mu_full_ens = np.clip(mu_full_ens, 0, 125)
            covered_full = (y_full >= mu_full_ens - C.Z_SCORE * sigma_full_ens) & \
                           (y_full <= mu_full_ens + C.Z_SCORE * sigma_full_ens)
            per_engine_picp = {int(u): float(np.mean(covered_full[u_full == u])) for u in np.unique(u_full)}
            compliance_rate_ge_090 = float(np.mean([v >= 0.90 for v in per_engine_picp.values()]))

            result[backbone][ds] = {
                'split_seed': M.SPLIT_SEED, 'init_seeds': M.INIT_SEEDS,
                'gate_check': 'PASS: fit/val/calib pairwise disjoint (runtime-asserted), '
                               'checkpoint selection + early stop both on val_units only, '
                               'scaler fit only on fit_units (split_seed=42)',
                'single_models': single_model_rows,
                'single_model_is_mean_across_inits': float(is_means_across_models.mean()),
                'single_model_is_std_across_inits': float(is_means_across_models.std(ddof=1)),
                'ensemble_same_split': {
                    'picp': picp_ens, 'mpiw': mpiw_ens, 'ece': ece_ens,
                    'is_mean': float(is_ens.mean()), 'is_std': float(is_ens.std(ddof=1)),
                    'wis_mean': float(wis_ens.mean()),
                    'per_engine_compliance_rate_ge_090': compliance_rate_ge_090,
                    'n_engines': len(per_engine_picp),
                },
            }
            print(f"  [{backbone}/{ds}] same-split ensemble: PICP={picp_ens:.4f} MPIW={mpiw_ens:.2f} "
                  f"ECE={ece_ens:.4f} IS={is_ens.mean():.3f} per_engine={compliance_rate_ge_090:.3f}")

    out_path = os.path.join(M.R4_DIR, 'samesplit_ensemble.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("R8-B1 samesplit ensemble re-eval complete (no retraining).")
