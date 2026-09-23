"""
STEP 2c: recomputes Deep Ensemble using STEP0c's new checkpoints
(inference only, no retraining).

Key detail: under the leakage-free protocol, each seed's scaler is fit
only on that seed's own fit_units (60%, different per seed), not a
single project-wide shared scaler -- the 5 members each expect a
different input normalization (each saw a different [-1,1] mapping
during its own training). So this script rebuilds each member's own
scaler **individually** from that seed's
`canonical_splits.json[...]['fit_units']`, and transforms the test set
with that member's own scaler, rather than letting all 5 members share a
single `C.load_and_process(ds)` output as the old version did.

The combination formula matches STEP2 exactly (Lakshminarayanan 2017
Gaussian mixture):
  mu_ens = mean_m(mu_m)
  sigma_ens^2 = mean_m(sigma_m^2 + mu_m^2) - mu_ens^2
"""
import os
import json
import time

import numpy as np
import torch

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
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


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  CKPT_DIR={CKPT_DIR}")

    results = {}
    per_model_infer_times = []

    for ds in C.DATASETS:
        print(f"\n{'='*20} {ds} {'='*20}")
        mu_members, sigma_members = [], []
        y_test_ref = None
        for seed in C.SEEDS:
            fit_units = CANON[ds][str(seed)]['fit_units']
            train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
            X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            if y_test_ref is None:
                y_test_ref = y_test
            else:
                assert np.allclose(y_test, y_test_ref), "y_test must be identical across members (true RUL doesn't depend on scaler)"

            ckpt_path = os.path.join(CKPT_DIR, f"{ds}_LSTM_seed{seed}.pt")
            model = C.load_checkpoint_model(ckpt_path, device)
            t0 = time.time()
            mu_m, sigma_m = infer(model, X_test_t)
            per_model_infer_times.append((time.time() - t0) / len(y_test) * 1000.0)
            mu_members.append(mu_m)
            sigma_members.append(sigma_m)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        y_test = y_test_ref
        mu_members = np.stack(mu_members)
        sigma_members = np.stack(sigma_members)
        M = mu_members.shape[0]

        mu_ens = mu_members.mean(0)
        sigma2_ens = (sigma_members ** 2 + mu_members ** 2).mean(0) - mu_ens ** 2
        sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
        mu_ens_clip = np.clip(mu_ens, 0, 125)

        rmse, score = C.rmse_score(y_test, mu_ens_clip)
        picp, mpiw = C.picp_mpiw(y_test, mu_ens_clip, sigma_ens)
        ece = C.compute_ece(mu_ens_clip, sigma_ens, y_test)
        lat_ms_per_sample = sum(per_model_infer_times[-M:])

        print(f"[{ds}] Deep Ensemble (M={M}): RMSE={rmse:.3f}  Score={score:.1f}  "
              f"PICP={picp:.3f}  MPIW={mpiw:.2f}  ECE={ece:.4f}  lat={lat_ms_per_sample:.4f}ms/smp")

        results[ds] = {
            'method': 'deep_ensemble', 'M': M,
            'variant': 'option_A_single_ensemble_no_seed_variance_leakfree',
            'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'latency_ms_per_sample': lat_ms_per_sample, 'member_seeds': C.SEEDS,
            'note': 'each member scaled by its OWN seed-specific fit_units scaler (leakfree protocol)',
        }

    out_path = os.path.join(RESULTS_DIR, 'deep_ensemble_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP2c complete.")
