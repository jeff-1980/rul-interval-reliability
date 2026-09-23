"""
Fair-calibration control. Fixed-variance (MSE-fixed) and MC-Dropout's
residual scale (aleatory_var) had always been estimated from fit_units
(training set) residuals; this switches to calib_units (the same engines
as Split-CP) residuals instead, reporting both versions side by side to
show the difference. Inference only, no retraining -- uses the existing
MC/MSE model checkpoints, just computes the residual variance on a
different data subset.

Inference on calib_units reuses train_transformer_split_cp.py's full
sliding-window sequences_for_units logic (label = training-style RUL, the
same data/labels CP uses).
"""
import os
import json

import numpy as np
import torch

import common as C
import transformer_common as T2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']


def sequences_for_units(df, feature_cols, unit_set):
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def batched_forward(model, X_t, batch=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            outs.append(model(X_t[i:i + batch]).cpu().numpy())
    return np.concatenate(outs).flatten()


def get_train_aleatory_var(backbone, ds, seed):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            per_seed = []
            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                calib_units = canon[ds][str(seed)]['calib_units']
                train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)

                X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)],
                                                             feat_cols, calib_units)
                y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
                X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)

                mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
                yhat_calib = np.clip(batched_forward(mc_model, X_calib_t) * 125.0, 0, 125)
                resid_calib = y_calib - yhat_calib
                aleatory_var_calib = float(np.var(resid_calib, ddof=1))

                aleatory_var_train = get_train_aleatory_var(backbone, ds, seed)

                per_seed.append({
                    'seed': seed, 'n_calib_windows': int(len(y_calib)),
                    'aleatory_var_train_fit': aleatory_var_train,
                    'aleatory_var_calib': aleatory_var_calib,
                    'sigma_fixed_train_fit': float(np.sqrt(aleatory_var_train)),
                    'sigma_fixed_calib': float(np.sqrt(aleatory_var_calib)),
                })
                print(f"  seed={seed}: sigma_fixed(train)={np.sqrt(aleatory_var_train):.3f}  "
                      f"sigma_fixed(calib)={np.sqrt(aleatory_var_calib):.3f}  "
                      f"ratio={np.sqrt(aleatory_var_calib)/np.sqrt(aleatory_var_train):.3f}")
                del mc_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            mean_train = float(np.mean([r['sigma_fixed_train_fit'] for r in per_seed]))
            mean_calib = float(np.mean([r['sigma_fixed_calib'] for r in per_seed]))
            result[backbone][ds] = {'per_seed': per_seed, 'mean_sigma_fixed_train_fit': mean_train,
                                     'mean_sigma_fixed_calib': mean_calib,
                                     'mean_ratio_calib_over_train': mean_calib / mean_train}
            print(f"  MEAN: train={mean_train:.3f}  calib={mean_calib:.3f}  ratio={mean_calib/mean_train:.3f}")

    out_path = os.path.join(R2_DIR, 'fair_calibration_sigma_fixed.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
