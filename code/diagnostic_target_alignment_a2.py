"""
Diagnostic, inference-level, no retraining: target-instant alignment.

Background: the training RUL label = (max_cycles - time_cycles).clip(125)
-- since training engines run to actual failure (run-to-failure), the last
row has time_cycles=max_cycles, RUL=0, i.e. "this row itself is the
failure instant". The test-side convention (V4.extract_raw_windows
mode='test') directly uses the official RUL_FD00X.txt value (clipped to
125) as the last window's ground truth: min(official_RUL, 125) --
official_RUL is NASA's stated "cycles remaining after the test file's last
row"; whether that shares the same counting origin as the training label's
"cycles remaining as of this row (inclusive)" depends on official_RUL's
exact definition -- if it measures "after the last row" rather than "as of
the last row", it's off by one cycle from the training label. This
diagnostic does exactly one thing: switches the test ground truth to
min(official_RUL - 1, 125) (matching the training label's counting
convention) and reports how much RMSE/PICP/IS/L=20 premature-trigger-rate
change -- no conclusion about which is "correct", just an honest report of
the magnitude of the difference.

Model output (mu, sigma) is identical under both conventions (the RUL
label is only used for evaluation, not as model input), so one inference
pass suffices, computing metrics against both y_true sets.

All 4 datasets x both backbones x NLL x 5 seeds, clean and drift 5%
conditions.
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
DIAG_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'protocol_diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
CONDITIONS = ['clean', 'drift5pct']
L = 20
Z = C.Z_SCORE
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


def metrics_for(y, mu, sigma):
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    lo, hi = mu - Z * sigma, mu + Z * sigma
    picp = float(np.mean((y >= lo) & (y <= hi)))
    is_mean = float(interval_score(y, mu, sigma, Z).mean())
    triggered = lo <= L
    at_risk = y <= L
    premature = triggered & (y > L + 20)
    return {'rmse': rmse, 'picp': picp, 'interval_score_mean': is_mean,
            'premature_rate_L20': float(premature.mean()), 'n_at_risk': int(at_risk.sum())}


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for ds in DATASETS:
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
        # extract_raw_windows(mode='test') itself already returns
        # min(official_RUL-1,125) (see noise_injection.py) -- this script
        # used to take its y_list directly as y_official, which meant
        # y_official and y_alt computed the same thing (both already
        # minus one), collapsing this whole contrast to nothing. Now
        # y_official / y_alt are both built independently from the raw
        # official terminal label, only borrowing extract_raw_windows'
        # X_raw_clean/u_ids (window layout and unit numbering, unrelated
        # to the RUL convention).
        X_raw_clean, _y_ignored, u_ids = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
        official_rul_raw = true_ruls.iloc[u_ids - 1]['RUL'].values.astype(np.float64)
        y_official = np.minimum(official_rul_raw, C.MAX_RUL)          # current-cycle convention: official RUL as-is
        y_alt = np.minimum(official_rul_raw - 1.0, C.MAX_RUL)          # next-cycle convention: matches training label
        assert not np.array_equal(y_official, y_alt), \
            f"{ds}: y_official and y_alt are identical -- the two conventions collapsed to the same array"
        X_raw_drift = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, full_scale)

        result[ds] = {}
        for backbone in BACKBONES:
            print(f"\n{'=' * 20} {ds} / {backbone} {'=' * 20}")
            result[ds][backbone] = {}
            for condition in CONDITIONS:
                X_raw = X_raw_clean if condition == 'clean' else X_raw_drift
                official_cells, alt_cells = [], []
                for seed in C.SEEDS:
                    fit_units = canon[ds][str(seed)]['fit_units']
                    _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                    X_scaled = V4.scale_raw_windows(X_raw, scaler)
                    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                    mu, ls = E.infer_nll(nll_model, X_t)
                    sigma = np.exp(ls) * 125.0
                    del nll_model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    official_cells.append(metrics_for(y_official, mu, sigma))
                    alt_cells.append(metrics_for(y_alt, mu, sigma))

                def agg(cells):
                    return {k: float(np.mean([c[k] for c in cells])) for k in
                            ['rmse', 'picp', 'interval_score_mean', 'premature_rate_L20']}

                result[ds][backbone][condition] = {
                    'official_RUL_current': agg(official_cells),
                    'RULminus1_train_convention': agg(alt_cells),
                }
                a, b = result[ds][backbone][condition]['official_RUL_current'], \
                    result[ds][backbone][condition]['RULminus1_train_convention']
                print(f"  [{condition:10s}] official: RMSE={a['rmse']:.3f} PICP={a['picp']:.4f} "
                      f"IS={a['interval_score_mean']:.2f} prem@L20={a['premature_rate_L20']:.4f}   |   "
                      f"RUL-1: RMSE={b['rmse']:.3f} PICP={b['picp']:.4f} "
                      f"IS={b['interval_score_mean']:.2f} prem@L20={b['premature_rate_L20']:.4f}")

    out_path = os.path.join(DIAG_DIR, 'A2_target_alignment_diagnostic.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
