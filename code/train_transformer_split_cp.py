"""
T2-A3: Transformer-backbone Split-CP (norm), leakage-free protocol, 4
datasets x 5 seeds.

Methodology note (a deliberate difference from the LSTM precedent,
recorded honestly): on the LSTM side (train_lstm_split_cp_
split_cp_leakfree.py), a separate NLL model was cloned and trained just
for Split-CP, purely to get inference results on calib_units -- the
architecture/data/loss are identical to the main NLL model
(train_lstm), just retrained. The scope of this experiment explicitly caps
Transformer training at "two heads (NLL, MSE) = 40 models", with no third
head, so no clone-training is done here -- instead this reuses the NLL
Transformer checkpoint already trained for head 1
(train_transformer_nll.py) as-is, and only adds the calib_units forward
pass it had never done, from which the CP-abs/CP-norm quantiles are
computed. This is a deliberate methodology difference from the LSTM
precedent, not an oversight.

CP-abs/CP-norm computation logic, the 20-bin ECE convention, and the
compliance check are identical to train_lstm_split_cp.
"""
import os
import json

import numpy as np
import torch

import common as C
import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
ALPHA_MAIN = 0.10
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def sequences_for_units(df, feature_cols, unit_set):
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def conformal_quantile(scores, alpha, n):
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    level = k / n
    return float(np.quantile(scores, level, method='higher'))


def picp_mpiw(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper))), float(np.mean(upper - lower))


def run_one_seed(ds_name, seed, device):
    split = CANON[ds_name][str(seed)]
    fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)

    ckpt_path = T2.nll_ckpt_path('Transformer', ds_name, seed)
    model = T2.load_checkpoint_model_t2('Transformer', ckpt_path, device)

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu_c, ls_c = model(X_calib_t)
    mu_calib = np.clip(mu_c.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_calib = torch.exp(ls_c).cpu().numpy().flatten() * 125.0

    with torch.no_grad():
        mu_t, ls_t = model(X_test_t)
    mu_test = np.clip(mu_t.cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_test = torch.exp(ls_t).cpu().numpy().flatten() * 125.0

    rmse, score = C.rmse_score(y_test, mu_test)

    n_calib = len(y_calib)
    s_abs = np.abs(y_calib - mu_calib)
    s_norm = np.abs(y_calib - mu_calib) / np.clip(sigma_calib, 1e-6, None)

    q_abs = conformal_quantile(s_abs, ALPHA_MAIN, n_calib)
    q_norm = conformal_quantile(s_norm, ALPHA_MAIN, n_calib)
    picp_abs, mpiw_abs = picp_mpiw(y_test, mu_test - q_abs, mu_test + q_abs)
    picp_norm, mpiw_norm = picp_mpiw(y_test, mu_test - q_norm * sigma_test, mu_test + q_norm * sigma_test)

    def ece_for_variant(is_norm):
        emp = []; q_by_level = {}
        for p in CONF_LEVELS:
            a = 1 - p
            scores = s_norm if is_norm else s_abs
            q = conformal_quantile(scores, a, n_calib)
            q_by_level[f"{p:.2f}"] = q
            if is_norm:
                lo, hi = mu_test - q * sigma_test, mu_test + q * sigma_test
            else:
                lo, hi = mu_test - q, mu_test + q
            emp.append(np.mean((y_test >= lo) & (y_test <= hi)))
        emp = np.array(emp)
        return float(np.mean(np.abs(emp - CONF_LEVELS))), q_by_level

    ece_abs, q_abs_by_level = ece_for_variant(False)
    ece_norm, q_norm_by_level = ece_for_variant(True)
    compliance_abs = abs(picp_abs - 0.90)
    compliance_norm = abs(picp_norm - 0.90)

    print(f"   [{ds_name}] seed={seed} n_calib_units={len(calib_units)} n_calib_windows={n_calib}  "
          f"CP-norm: PICP={picp_norm:.3f} MPIW={mpiw_norm:.2f} ECE={ece_norm:.4f} "
          f"|dev|={compliance_norm:.3f}{'  <-- outside +/-0.03 tolerance' if compliance_norm > 0.03 else ''}")

    return {
        'seed': seed, 'n_calib_units': len(calib_units), 'n_calib_windows': int(n_calib),
        'rmse': rmse, 'score': score,
        'cp_abs': {'picp': picp_abs, 'mpiw': mpiw_abs, 'ece': ece_abs, 'q': q_abs,
                   'deviation_from_090': compliance_abs, 'q_by_level': q_abs_by_level},
        'cp_norm': {'picp': picp_norm, 'mpiw': mpiw_norm, 'ece': ece_norm, 'q': q_norm,
                    'deviation_from_090': compliance_norm, 'q_by_level': q_norm_by_level},
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Backbone=Transformer  Split-CP (reuse NLL checkpoint)  ALPHA={ALPHA_MAIN}")

    all_out = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        all_out[ds] = [run_one_seed(ds, seed, device) for seed in C.SEEDS]

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
    print("T2-A3 (Transformer Split-CP, reused NLL checkpoint) complete.")
