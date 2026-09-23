"""
STEP 2d: ensemble-size sweep M in {2,3,5,10}, leakfree protocol, 15-seed
pool (5 original + 10 extra).

Same disjoint-grouping / arm-A noise-injection logic as this script's
earlier (non-leakfree-protocol) version; the only difference: checkpoints
come from checkpoints_leakfree/, and each member within a group uses
**its own** canonical fit_units scaler (under the leakfree protocol the
scaler differs per seed, so a group can't share one scaler). The noise
itself (the raw, unstandardized perturbation) is shared across all
members within a group for a given snr/trial; only the standardization
step differs per member.
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
os.makedirs(LEAKFREE_DIR, exist_ok=True)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)

ORIG_SEEDS = [42, 2024, 7, 888, 123]
EXTRA_SEEDS = [8601, 16713, 19906, 21852, 38466, 40016, 40967, 51778, 52287, 90270]
ALL_15_SEEDS = ORIG_SEEDS + EXTRA_SEEDS
M_LEVELS = [2, 3, 5, 10]
SNR_LEVELS = V4.SNR_LEVELS


def ckpt_path_for_seed(ds, seed):
    if seed in ORIG_SEEDS:
        return os.path.join(CKPT_DIR, f"{ds}_LSTM_seed{seed}.pt")
    return os.path.join(CKPT_DIR, f"{ds}_LSTM_extraseed{seed}.pt")


def disjoint_groups(seeds, M, shuffle_seed):
    rng = np.random.RandomState(shuffle_seed)
    perm = rng.permutation(seeds).tolist()
    n_groups = len(perm) // M
    groups = [perm[i * M:(i + 1) * M] for i in range(n_groups)]
    dropped = perm[n_groups * M:]
    return groups, dropped


def infer_full_batch(model, X_t, batch=8192):
    mus, sigmas = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mu, log_sigma = model(X_t[i:i + batch])
            mus.append(mu.cpu().numpy().flatten())
            sigmas.append(torch.exp(log_sigma).cpu().numpy().flatten())
    mu = np.concatenate(mus) * 125.0
    sigma = np.concatenate(sigmas) * 125.0
    return np.clip(mu, 0, 125), sigma


def ensemble_predict(mu_members, sigma_members):
    mu_ens = mu_members.mean(0)
    sigma2_ens = (sigma_members ** 2 + mu_members ** 2).mean(0) - mu_ens ** 2
    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
    return np.clip(mu_ens, 0, 125), sigma_ens


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  M_LEVELS={M_LEVELS}  total seeds={len(ALL_15_SEEDS)}  arm=A_percondition only")

    results = {}
    for ds in C.DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        if ds == 'FD001':
            global_std = np.std(train_df_raw[feat_cols].values, axis=0)
            km, cond_std = None, None
            scheme = 'global'
        else:
            km, cond_std, global_std = V4.fit_condition_model(train_df_raw, feat_cols)
            cond_std = {c: V4.sensor_only_scale(feat_cols, v) for c, v in cond_std.items()}
            global_std = V4.sensor_only_scale(feat_cols, global_std)
            scheme = 'per_condition'

        # per-seed leakfree scaler for all 15 seeds
        scalers_by_seed = {}
        for seed in ALL_15_SEEDS:
            fit_units = CANON[ds][str(seed)]['fit_units']
            _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
            scalers_by_seed[seed] = scaler

        ds_out = {}
        for M in M_LEVELS:
            shuffle_seed = C.stable_seed(ds, M, 'leakfree')
            groups, dropped = disjoint_groups(ALL_15_SEEDS, M, shuffle_seed)
            print(f"\n  M={M}: {len(groups)} disjoint groups (dropped {len(dropped)} seeds: {dropped})")

            group_results = []
            for gi, group in enumerate(groups):
                models = [C.load_checkpoint_model(ckpt_path_for_seed(ds, s), device) for s in group]

                snr_out = {}
                for snr in SNR_LEVELS:
                    snr_key = 'inf' if np.isinf(snr) else str(snr)
                    noise_rng = np.random.RandomState((C.stable_seed(ds, M, gi, snr_key, 'leakfree')))
                    raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, snr, noise_rng, scheme,
                                                     global_std=global_std, km=km, cond_std=cond_std)

                    mu_members, sigma_members = [], []
                    for seed, model in zip(group, models):
                        df_noisy, _ = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scalers_by_seed[seed])
                        X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                        mu_m, sigma_m = infer_full_batch(model, X_t)
                        mu_members.append(mu_m); sigma_members.append(sigma_m)
                        del X_t
                    mu_ens, sigma_ens = ensemble_predict(np.stack(mu_members), np.stack(sigma_members))

                    rmse, score = C.rmse_score(y_test, mu_ens)
                    picp, mpiw = C.picp_mpiw(y_test, mu_ens, sigma_ens)
                    snr_out[snr_key] = {'rmse': rmse, 'picp': picp, 'mpiw': mpiw,
                                         'sigma_mean': float(sigma_ens.mean())}
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                group_results.append({'group_seeds': group, 'by_snr': snr_out})
                print(f"    group {gi} seeds={group}: clean PICP={snr_out['inf']['picp']:.3f}  "
                      f"0dB PICP={snr_out['0']['picp']:.3f}  0dB sigma={snr_out['0']['sigma_mean']:.2f}")
                for m in models:
                    del m
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            picp_clean_arr = np.array([g['by_snr']['inf']['picp'] for g in group_results])
            picp_0db_arr = np.array([g['by_snr']['0']['picp'] for g in group_results])
            ds_out[str(M)] = {
                'n_groups': len(groups), 'dropped_seeds': dropped, 'groups': group_results,
                'clean_picp_mean': float(picp_clean_arr.mean()),
                'clean_picp_std': float(picp_clean_arr.std(ddof=1)) if len(picp_clean_arr) > 1 else 'n/a (single group)',
                '0dB_picp_mean': float(picp_0db_arr.mean()),
                '0dB_picp_std': float(picp_0db_arr.std(ddof=1)) if len(picp_0db_arr) > 1 else 'n/a (single group)',
            }

        results[ds] = ds_out

    out_path = os.path.join(LEAKFREE_DIR, 'ensemble_scale_sweep_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("STEP2d complete.")
