"""
T2-A7：Transformer 骨干 Deep Ensemble 在真正 clean（无扰动）测试集上的
RMSE/PICP/MPIW/ECE，与 eval_lstm_fd003_deep_ensemble.py 同一协议（复用
NLL 5 seeds checkpoint，逐成员用自己的 canonical fit_units scaler），4 数据集。
供 cost table 主表行使用（区别于臂C扫描里 0.1%FS 那一档，这里是严格 clean）。
"""
import os
import json
import time

import numpy as np
import torch

import common as C
import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def infer(model, X_t, batch=8192):
    mus, sigmas = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mu, log_sigma = model(X_t[i:i + batch])
            mus.append(mu.cpu().numpy().flatten())
            sigmas.append(torch.exp(log_sigma).cpu().numpy().flatten())
    mu = np.concatenate(mus) * 125.0
    sigma = np.concatenate(sigmas) * 125.0
    return np.clip(mu, 0, 125), sigma


def run_dataset(ds, device):
    mu_members, sigma_members = [], []
    y_test_ref = None
    per_model_infer_times = []
    for seed in C.SEEDS:
        fit_units = CANON[ds][str(seed)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
        X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        if y_test_ref is None:
            y_test_ref = y_test
        else:
            assert np.allclose(y_test, y_test_ref)

        model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)
        t0 = time.time()
        mu_m, sigma_m = infer(model, X_test_t)
        per_model_infer_times.append((time.time() - t0) / len(y_test) * 1000.0)
        mu_members.append(mu_m); sigma_members.append(sigma_m)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    y_test = y_test_ref
    mu_members = np.stack(mu_members); sigma_members = np.stack(sigma_members)
    M = mu_members.shape[0]
    mu_ens = mu_members.mean(0)
    sigma2_ens = (sigma_members ** 2 + mu_members ** 2).mean(0) - mu_ens ** 2
    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
    mu_ens_clip = np.clip(mu_ens, 0, 125)

    rmse, score = C.rmse_score(y_test, mu_ens_clip)
    picp, mpiw = C.picp_mpiw(y_test, mu_ens_clip, sigma_ens)
    ece = C.compute_ece(mu_ens_clip, sigma_ens, y_test)

    print(f"[{ds}] Deep Ensemble (M={M}, Transformer): RMSE={rmse:.3f}  PICP={picp:.3f}  MPIW={mpiw:.2f}  ECE={ece:.4f}")
    return {
        'method': 'deep_ensemble', 'M': M, 'variant': 'option_A_single_ensemble_no_seed_variance_leakfree_T2',
        'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
        'latency_ms_per_sample': sum(per_model_infer_times), 'member_seeds': C.SEEDS,
    }


if __name__ == '__main__':
    C.require_fixed_hashseed()  # R8-B5: root-caused run1-vs-run2 MD5 mismatch to missing cuDNN determinism here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  Deep Ensemble (clean)")

    result = {}
    for ds in DATASETS:
        result[ds] = run_dataset(ds, device)

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_ensemble_clean_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("T2-A7 (Transformer Deep Ensemble, clean) complete.")
