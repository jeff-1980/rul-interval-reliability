"""
STEP 6b (leakfree)：代价表 MSE 行，用 checkpoints_leakfree 的 MCDropoutMSE
checkpoint（eval模式=标准MSE推理），逐 seed 用自己的 canonical fit_units
scaler。固定 sigma 复用 mcdropout_fixed_leakfree.json 的 aleatory_var。
"""
import os
import json

import numpy as np
import torch

import common as C
import mc_dropout_model as S1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
Z_SCORE = 1.645

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def load_mse_model(ds, seed, device):
    ck = torch.load(os.path.join(CKPT_DIR, f"{ds}_MCDropoutMSE_seed{seed}.pt"),
                     map_location=device, weights_only=False)
    m = S1.MC_LSTM(ck['input_dim'], ck['hidden_dim'], ck['dropout']).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m


def infer_mu(model, X_t, batch=8192):
    mus = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mus.append(model(X_t[i:i + batch]).cpu().numpy().flatten())
    return np.clip(np.concatenate(mus) * 125.0, 0, 125)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(RESULTS_DIR, 'mcdropout_fixed_leakfree.json')) as f:
        mc_json = json.load(f)

    results = {}
    for ds in C.DATASETS:
        print(f"\n{'='*20} {ds} {'='*20}")
        seed_rows = []
        engine_picp_per_seed = []
        for i, seed in enumerate(C.SEEDS):
            fit_units = CANON[ds][str(seed)]['fit_units']
            train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
            X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            X_full, y_full, u_full = C.create_full_trajectory_test_windows(test_df, feat_cols, true_ruls)
            X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)

            model = load_mse_model(ds, seed, device)
            aleatory_var = mc_json[ds][i]['T50']['kendall_gal_full']['aleatory_var']
            sigma_fixed = float(np.sqrt(aleatory_var))

            mu = infer_mu(model, X_test_t)
            sigma_arr = np.full_like(mu, sigma_fixed)
            rmse, score = C.rmse_score(y_test, mu)
            picp, mpiw = C.picp_mpiw(y_test, mu, sigma_arr, z=Z_SCORE)
            ece = C.compute_ece(mu, sigma_arr, y_test)
            seed_rows.append({'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp,
                               'mpiw': mpiw, 'ece': ece, 'sigma_fixed': sigma_fixed})
            print(f"  seed={seed}: RMSE={rmse:.3f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

            mu_full = infer_mu(model, X_full_t)
            sigma_full = np.full_like(mu_full, sigma_fixed)
            covered = (y_full >= mu_full - Z_SCORE * sigma_full) & (y_full <= mu_full + Z_SCORE * sigma_full)
            eng_picp = {int(u): float(np.mean(covered[u_full == u])) for u in np.unique(u_full)}
            engine_picp_per_seed.append(eng_picp)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        units = sorted(engine_picp_per_seed[0].keys())
        engine_picp_mean = {u: float(np.mean([d[u] for d in engine_picp_per_seed])) for u in units}
        vals = np.array(list(engine_picp_mean.values()))
        compliance_rate = float(np.mean(vals >= 0.90))

        rmse_arr = np.array([r['rmse'] for r in seed_rows])
        picp_arr = np.array([r['picp'] for r in seed_rows])
        mpiw_arr = np.array([r['mpiw'] for r in seed_rows])
        ece_arr = np.array([r['ece'] for r in seed_rows])

        results[ds] = {
            'per_seed': seed_rows,
            'rmse_mean': float(rmse_arr.mean()), 'rmse_std': float(rmse_arr.std(ddof=1)),
            'picp_mean': float(picp_arr.mean()), 'picp_std': float(picp_arr.std(ddof=1)),
            'mpiw_mean': float(mpiw_arr.mean()), 'mpiw_std': float(mpiw_arr.std(ddof=1)),
            'ece_mean': float(ece_arr.mean()), 'ece_std': float(ece_arr.std(ddof=1)),
            'per_engine_compliance_rate_ge_090': compliance_rate,
            'per_engine_picp_median': float(np.median(vals)),
            'params': int(sum(p.numel() for p in S1.MC_LSTM(len(feat_cols), C.HIDDEN_DIM, 0.2).parameters())),
        }
        print(f"  -> RMSE={results[ds]['rmse_mean']:.3f}±{results[ds]['rmse_std']:.3f}  "
              f"PICP={results[ds]['picp_mean']:.3f}±{results[ds]['picp_std']:.3f}  "
              f"per-engine={compliance_rate:.3f}")

    out_path = os.path.join(RESULTS_DIR, 'mse_row_recomputed_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP6b (leakfree) complete.")
