"""
Two parts:
Part 1: FD004's M-sweep (M in {2,3,5,10}) -- computes IS to check against
     an earlier draft's Table 7 PICP(0.910->0.937), to see whether the
     "no benefit" finding still holds and whether the metric changed.
Part 2: 3 disjoint M=5 ensembles grouped from the 15-seed pool (LSTM, 4
     datasets), reporting each ensemble's IS/WIS and the paired-difference
     bootstrap interval against the single-model mechanisms
     (NLL/MSE-fixed) (resampled by engine, since the end-window convention
     gives one sample per engine, so "resample by engine" is the standard
     per-sample bootstrap).

Reuses the disjoint group_seeds already recorded in
leakfree/ensemble_scale_sweep_leakfree.json (FD001/2/4) +
leakfree/FD003_ensemble_scale_sweep_leakfree.json (guaranteeing the same
grouping as the already-published PICP numbers, no re-randomizing the
grouping), only re-running inference to get per-engine mu/sigma/y for
computing IS/WIS.
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
R3_DIR = os.path.join(RESULTS_DIR, 'leakfree_r3')
os.makedirs(R3_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
N_BOOT = 2000
RNG = np.random.RandomState(0)


def load_groups(ds):
    if ds == 'FD003':
        with open(os.path.join(RESULTS_DIR, 'leakfree', 'FD003_ensemble_scale_sweep_leakfree.json')) as f:
            d = json.load(f)['FD003']
    else:
        with open(os.path.join(RESULTS_DIR, 'leakfree', 'ensemble_scale_sweep_leakfree.json')) as f:
            d = json.load(f)[ds]
    return d


def nll_ckpt_for_seed(ds, seed):
    orig_seeds = [42, 2024, 7, 888, 123]
    tag = 'seed' if seed in orig_seeds else 'extraseed'
    return os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_LSTM_{tag}{seed}.pt")


def infer_group_mu_sigma(ds, group_seeds, device):
    """Under the clean condition, per-seed NLL inference, returns (mu_members, sigma_members, y)."""
    canon = json.load(open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')))
    mu_mem, sigma_mem, y_ref = [], [], None
    for seed in group_seeds:
        fit_units = canon[ds][str(seed)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
        X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        model = C.load_checkpoint_model(nll_ckpt_for_seed(ds, seed), device)
        mu, ls = E.infer_nll(model, X_t)
        sigma = np.exp(ls) * 125.0
        mu_mem.append(mu); sigma_mem.append(sigma)
        y_ref = y_test
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.stack(mu_mem), np.stack(sigma_mem), y_ref


def combine_ensemble(mu_mem, sigma_mem):
    mu_ens = mu_mem.mean(0)
    sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
    mu_ens = np.clip(mu_ens, 0, 125)
    return mu_ens, sigma_ens


def bootstrap_ci(a, b, n_boot=N_BOOT, rng=RNG):
    """a,b: per-engine arrays of the SAME metric for two mechanisms, same
    engines (paired). Returns (mean_diff, ci_lo, ci_hi) for mean(a-b),
    resampling engines with replacement."""
    n = len(a)
    diffs = a - b
    boot_means = np.array([diffs[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- A.2: FD004 M-sweep with IS added ----
    print("\n" + "=" * 20 + " A.2: FD004 M-sweep " + "=" * 20)
    fd004_groups = load_groups('FD004')
    a2_out = {}
    for M in ['2', '3', '5', '10']:
        groups = fd004_groups[M]['groups']
        picps, iss = [], []
        for g in groups:
            mu_mem, sigma_mem, y_ref = infer_group_mu_sigma('FD004', g['group_seeds'], device)
            mu_ens, sigma_ens = combine_ensemble(mu_mem, sigma_mem)
            picp, _ = E.picp_mpiw(y_ref, mu_ens, sigma_ens, C.Z_SCORE)
            is_arr = interval_score(y_ref, mu_ens, sigma_ens, C.Z_SCORE)
            picps.append(picp); iss.append(float(is_arr.mean()))
        a2_out[M] = {'n_groups': len(groups), 'picp_mean': float(np.mean(picps)),
                     'picp_per_group': picps, 'is_mean': float(np.mean(iss)), 'is_per_group': iss}
        print(f"  M={M}: n_groups={len(groups)}  PICP={a2_out[M]['picp_mean']:.4f}  IS={a2_out[M]['is_mean']:.2f}  "
              f"(per-group IS: {[round(v,1) for v in iss]})")

    with open(os.path.join(R3_DIR, 'A2_fd004_msweep_IS.json'), 'w') as fp:
        json.dump(a2_out, fp, indent=2, default=float)

    # ---- E: 3 disjoint M=5 ensembles, 4 datasets, IS/WIS + bootstrap vs single-model mechanisms ----
    print("\n" + "=" * 20 + " E: independent M=5 ensemble replication " + "=" * 20)
    e_out = {}
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        groups5 = load_groups(ds)['5']['groups']
        canon = json.load(open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')))

        # single-model mechanisms for comparison: use the ORIGINAL 5-seed NLL model (seed=42) as
        # a representative single model, and MSE_fixed (seed=42, calib-based sigma) -- these are
        # the standard "one trained instance" comparators referenced by the main table.
        seed0 = 42
        fit_units = canon[ds][str(seed0)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
        X_test, y_test_single = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        nll_model0 = C.load_checkpoint_model(nll_ckpt_for_seed(ds, seed0), device)
        mu0, ls0 = E.infer_nll(nll_model0, X_t)
        sigma0 = np.exp(ls0) * 125.0
        is_single_nll = interval_score(y_test_single, mu0, sigma0, C.Z_SCORE)
        del nll_model0

        ds_out = {'groups': [], 'single_model_seed42_IS_mean': float(is_single_nll.mean())}
        group_is_arrays = []
        for gi, g in enumerate(groups5):
            mu_mem, sigma_mem, y_ref = infer_group_mu_sigma(ds, g['group_seeds'], device)
            mu_ens, sigma_ens = combine_ensemble(mu_mem, sigma_mem)
            picp, mpiw = E.picp_mpiw(y_ref, mu_ens, sigma_ens, C.Z_SCORE)
            is_arr = interval_score(y_ref, mu_ens, sigma_ens, C.Z_SCORE)
            wis_arr = wis(y_ref, mu_ens, sigma_ens, C.Z_SCORE)
            group_is_arrays.append((is_arr, y_ref))
            ds_out['groups'].append({
                'group_seeds': g['group_seeds'], 'picp': picp, 'mpiw': mpiw,
                'is_mean': float(is_arr.mean()), 'wis_mean': float(wis_arr.mean()),
            })
            print(f"  group {gi} seeds={g['group_seeds']}: PICP={picp:.3f} IS={is_arr.mean():.2f} WIS={wis_arr.mean():.2f}")

        is_means = [g['is_mean'] for g in ds_out['groups']]
        ds_out['is_across_groups_mean'] = float(np.mean(is_means))
        ds_out['is_across_groups_std'] = float(np.std(is_means, ddof=1))

        # bootstrap: ensemble group 0 vs single-model NLL (paired by engine; assumes same
        # engine order/count, true for end-window terminal unit within one sub-dataset)
        is_ens0, y_ens0 = group_is_arrays[0]
        assert len(is_ens0) == len(is_single_nll)
        mean_diff, lo, hi = bootstrap_ci(is_single_nll, is_ens0, rng=np.random.RandomState(0))
        ds_out['bootstrap_single_nll_minus_ensemble_group0'] = {
            'mean_diff': mean_diff, 'ci95_lo': lo, 'ci95_hi': hi,
            'note': 'positive means single-model NLL has HIGHER (worse) IS than the ensemble'}
        print(f"  bootstrap (single NLL - ensemble[group0]) IS diff: {mean_diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")

        e_out[ds] = ds_out

    with open(os.path.join(R3_DIR, 'E_ensemble_independent_replication.json'), 'w') as fp:
        json.dump(e_out, fp, indent=2, default=float)
    print(f"\nSaved -> A2_fd004_msweep_IS.json, E_ensemble_independent_replication.json")
