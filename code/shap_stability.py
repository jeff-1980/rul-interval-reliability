"""
SHAP cross-seed stability, leakage-free checkpoints + KernelSHAP (switched
from an earlier GradientExplainer version to KernelSHAP; a deliberate
methodology change, recorded explicitly).

Differences from the earlier version:
  1. Uses the leakage-free checkpoints' 5 seeds (no retraining), each seed
     with its own canonical fit_units scaler.
  2. Explainer switched from GradientExplainer to KernelExplainer --
     feature granularity is per sensor channel (14 channels, not the
     "per-timestep x per-sensor" 420-dim version): for each explained
     sample, a coalition mask decides which sensor channels get replaced
     with the background trajectory (other timesteps keep that sample's
     original value), then a forward pass gives mu. Closer to the standard
     "grouped SHAP" approach than the earlier "420-dim gradient
     attribution aggregated by timestep", and model-agnostic (no gradient
     dependency).
  3. nsamples=200 (KernelSHAP's coalition-sampling count for coefficient
     estimation, set explicitly rather than 'auto''s ~2076, to finish 5
     seeds x 50 explained samples in reasonable time -- a deliberate
     methodology parameter change, not a shortcut: the earlier
     GradientExplainer computed exact analytic gradients and has no
     "sample count" concept at all, so the two methods' "precision" isn't
     directly comparable).
  4. Number of explained samples N_EXP: matches the earlier version's own
     formula, `n_exp = min(200, len(X_te))` (for FD001's 100-window test
     set, this explains all 100) -- background (n_bg=200) and test-sample
     count now align with the earlier version exactly; only the explainer
     itself (KernelSHAP vs. GradientExplainer) and nsamples (a
     KernelSHAP-specific parameter) are the deliberate methodology change.
  5. Convergence check: nsamples takes a command-line argument
     (`python3 shap_stability.py 800`), to compare baseline(200) vs.
     4x(800)'s cross-seed Spearman for convergence (small rho difference
     -> converged; large difference -> 200 is too small, sampling noise
     dominates rank instability). If neither setting reaches 0.7, the
     whole SHAP appendix is dropped rather than keeping a weak result.

Output: leakfree/shap_ranking_leakfree{_ns<N>}.csv (Table 4, filename
suffixed when N=800) + leakfree/shap_spearman_leakfree{_ns<N>}.json (5x5
cross-seed correlation matrix) + a rank-difference comparison against the
earlier baseline_shap_ranking.csv.
"""
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
import shap

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
os.makedirs(LEAKFREE_DIR, exist_ok=True)

DATASET = 'FD001'
N_BG = 200
N_EXP = 200  # matches the earlier version's own formula: n_exp = min(200, len(X_te)); for FD001 test set (100 windows) -> 100
KERNEL_NSAMPLES = 200  # baseline; convergence check reruns with this x4 = 800

import sys
if len(sys.argv) > 1:
    KERNEL_NSAMPLES = int(sys.argv[1])
OUT_SUFFIX = f"_ns{KERNEL_NSAMPLES}" if len(sys.argv) > 1 else ''

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


class _MuWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        mu, _ = self.m(x)
        return mu


def make_mask_predict_fn(wrapper_cpu, X_sample, bg_mean_traj):
    """X_sample: (30,14) the sample being explained; bg_mean_traj: (30,14)
    background mean trajectory. mask: (n_coalitions, 14) binary, 1=keep
    that sensor channel's original value, 0=replace with background.
    """
    n_feat = X_sample.shape[1]

    def f(mask):
        n = mask.shape[0]
        batch = np.tile(X_sample[None, :, :], (n, 1, 1)).astype(np.float32).copy()
        for i in range(n):
            for j in range(n_feat):
                if mask[i, j] == 0:
                    batch[i, :, j] = bg_mean_traj[:, j]
        with torch.no_grad():
            out = wrapper_cpu(torch.tensor(batch, dtype=torch.float32))
        return out.numpy().flatten()

    return f


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  (SHAP itself runs on CPU per-instance, model loaded on {device} then moved)")

    feat_cols = C.get_feature_names(DATASET)
    n_feat = len(feat_cols)

    shap_per_seed = []
    rankings = []
    seed_rank_dfs = {}

    for seed in C.SEEDS:
        print(f"\n--- Seed {seed} ---")
        fit_units = CANON[DATASET][str(seed)]['fit_units']
        train_df, test_df, true_ruls, fc, scaler = C.load_and_process_leakfree(DATASET, fit_units)
        assert fc == feat_cols

        X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
        X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)

        model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DATASET}_LSTM_seed{seed}.pt"), device)
        wrapper = _MuWrapper(model).cpu()
        wrapper.eval()

        rng = np.random.RandomState(seed)
        idx_bg = rng.choice(len(X_fit), min(N_BG, len(X_fit)), replace=False)
        bg_mean_traj = X_fit[idx_bg].mean(axis=0)  # (30, n_feat)

        n_exp = min(N_EXP, len(X_test))
        idx_ex = rng.choice(len(X_test), n_exp, replace=False)

        shap_vals_all = np.zeros((n_exp, n_feat))
        for si, i in enumerate(idx_ex):
            X_sample = X_test[i]  # (30, n_feat)
            f = make_mask_predict_fn(wrapper, X_sample, bg_mean_traj)
            explainer = shap.KernelExplainer(f, np.zeros((1, n_feat)))
            sv = explainer.shap_values(np.ones((1, n_feat)), nsamples=KERNEL_NSAMPLES, silent=True)
            sv = np.array(sv).flatten()
            shap_vals_all[si] = np.abs(sv)
            if (si + 1) % 10 == 0:
                print(f"    explained {si + 1}/{n_exp} samples")

        mean_shap = shap_vals_all.mean(axis=0)
        shap_per_seed.append(mean_shap)
        rank = stats.rankdata(-mean_shap)
        rankings.append(rank)
        print(f"  Top-3: {[feat_cols[i] for i in np.argsort(mean_shap)[::-1][:3]]}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    shap_mean_all = np.stack(shap_per_seed)
    rankings = np.stack(rankings)

    corr_matrix = np.ones((5, 5))
    for i in range(5):
        for j in range(5):
            rho, _ = stats.spearmanr(rankings[i], rankings[j])
            corr_matrix[i, j] = rho
    print(f"\nSpearman correlation matrix (leakfree):\n{np.round(corr_matrix, 3)}")
    upper = corr_matrix[np.triu_indices(5, k=1)]
    print(f"Mean off-diagonal Spearman r = {upper.mean():.3f} ± {upper.std():.3f}")

    mean_shap_avg = shap_mean_all.mean(0)
    std_shap_avg = shap_mean_all.std(0)
    order = np.argsort(mean_shap_avg)[::-1]

    PHYS = {
        's_2': 'Total temperature fan inlet (°R)', 's_3': 'Total temperature LPC outlet (°R)',
        's_4': 'Total temperature HPC outlet (°R)', 's_7': 'Total pressure HPC outlet (psia)',
        's_8': 'Physical fan speed (rpm)', 's_9': 'Physical core speed (rpm)',
        's_11': 'Static pressure HPC outlet (psia)', 's_12': 'Fuel flow ratio (pps/psi)',
        's_13': 'Corrected fan speed (rpm)', 's_14': 'Corrected core speed (rpm)',
        's_15': 'Bypass ratio', 's_17': 'Bleed enthalpy',
        's_20': 'HPT cool air flow', 's_21': 'LPT cool air flow',
    }
    rows = []
    for rank_i, feat_i in enumerate(order):
        feat = feat_cols[feat_i]
        rows.append({'Rank': rank_i + 1, 'Sensor': feat,
                      'Mean |SHAP|': round(float(mean_shap_avg[feat_i]), 6),
                      'Std |SHAP|': round(float(std_shap_avg[feat_i]), 6),
                      'Physical Meaning': PHYS.get(feat, 'N/A')})
    df_rank = pd.DataFrame(rows)
    rank_path = os.path.join(LEAKFREE_DIR, f'shap_ranking_leakfree{OUT_SUFFIX}.csv')
    df_rank.to_csv(rank_path, index=False)
    print(f"\nSaved -> {rank_path}")
    print(df_rank.to_string(index=False))

    with open(os.path.join(LEAKFREE_DIR, f'shap_spearman_leakfree{OUT_SUFFIX}.json'), 'w') as fp:
        json.dump({'corr_matrix': corr_matrix.tolist(), 'seeds': C.SEEDS,
                    'mean_offdiag_spearman': float(upper.mean()), 'std_offdiag_spearman': float(upper.std()),
                    'method': f'KernelExplainer, sensor-channel-grouped masking, nsamples={KERNEL_NSAMPLES}, n_exp={n_exp}, n_bg={N_BG}'},
                  fp, indent=2)

    # ---- Compare against an earlier (pre-audit) ranking, if available ----
    earlier_ranking_path = os.path.join(PROJ_DIR, 'results', 'baseline_shap_ranking.csv')
    if os.path.exists(earlier_ranking_path):
        earlier_df = pd.read_csv(earlier_ranking_path)
        earlier_rank = dict(zip(earlier_df['Sensor'], earlier_df['Rank']))
        new_rank = dict(zip(df_rank['Sensor'], df_rank['Rank']))
        common = sorted(set(earlier_rank) & set(new_rank))
        earlier_r = [earlier_rank[s] for s in common]
        new_r = [new_rank[s] for s in common]
        rho, pval = stats.spearmanr(earlier_r, new_r)
        print(f"\n=== earlier (GradientExplainer, pre-audit checkpoint) vs current (KernelSHAP) ranking ===")
        diff_rows = []
        for s in common:
            diff_rows.append({'Sensor': s, 'earlier_rank': earlier_rank[s], 'leakfree_rank': new_rank[s],
                               'rank_change': new_rank[s] - earlier_rank[s]})
        diff_df = pd.DataFrame(diff_rows).sort_values('Sensor')
        print(diff_df.to_string(index=False))
        print(f"\nSpearman(earlier_rank, leakfree_rank) = {rho:.3f} (p={pval:.4f})")
        diff_path = os.path.join(LEAKFREE_DIR, f'shap_ranking_diff{OUT_SUFFIX}.csv')
        diff_df.to_csv(diff_path, index=False)
        with open(os.path.join(LEAKFREE_DIR, f'shap_spearman_vs_earlier{OUT_SUFFIX}.json'), 'w') as fp:
            json.dump({'spearman_rho': float(rho), 'spearman_p': float(pval),
                        'note': 'NOT apples-to-apples: the earlier ranking used GradientExplainer on a pre-audit '
                                '(leaky) checkpoint; the current ranking uses KernelExplainer (sensor-grouped) '
                                'on the leakage-audited checkpoint -- both the weights AND the explainer '
                                'method changed, this rho conflates both effects'},
                      fp, indent=2)
        print(f"Saved -> {diff_path}")
    else:
        print(f"\n⚠️ earlier ranking file not found at {earlier_ranking_path}, skipping comparison")

    print("\nE5b complete.")
