"""
R8-B1（推理级，不重训）：重新评估 Split-CP 的三个文件。checkpoint 权重不变；
calib 集上标定的 q/q_by_level 只依赖训练标签（不读官方RUL文件），B2 不影响，
重算一遍不会变，只是为了代码简单在这里也一并重算（不额外读旧文件）。
真正受 B2 影响的是 rmse/score/picp/mpiw/ece（依赖官方测试集真值）。

覆盖：
  split_cp_leakfree.json            (LSTM, FD001/FD002/FD004, cp_abs+cp_norm)
  stepFD003_splitcp_leakfree_results.json (LSTM, FD003, cp_norm only)
  t2_transformer_splitcp_leakfree_results.json (Transformer, 全部4个数据集, cp_abs+cp_norm)
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
ALPHA_MAIN = 0.10
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)


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


def eval_one(model, ds, seed, canon, device, with_abs):
    fit_units = canon[ds][str(seed)]['fit_units']
    calib_units = canon[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
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
    s_norm = np.abs(y_calib - mu_calib) / np.clip(sigma_calib, 1e-6, None)
    q_norm = conformal_quantile(s_norm, ALPHA_MAIN, n_calib)
    picp_norm, mpiw_norm = picp_mpiw(y_test, mu_test - q_norm * sigma_test, mu_test + q_norm * sigma_test)
    q_norm_by_level, emp = {}, []
    for p in CONF_LEVELS:
        q = conformal_quantile(s_norm, 1 - p, n_calib)
        q_norm_by_level[f"{p:.2f}"] = q
        lo, hi = mu_test - q * sigma_test, mu_test + q * sigma_test
        emp.append(np.mean((y_test >= lo) & (y_test <= hi)))
    ece_norm = float(np.mean(np.abs(np.array(emp) - CONF_LEVELS)))
    out = {'seed': seed, 'n_calib_units': len(calib_units), 'n_calib_windows': int(n_calib),
           'rmse': rmse, 'score': score,
           'cp_norm': {'picp': picp_norm, 'mpiw': mpiw_norm, 'ece': ece_norm, 'q': q_norm,
                       'q_by_level': q_norm_by_level, 'deviation_from_090': abs(picp_norm - 0.90)}}

    if with_abs:
        s_abs = np.abs(y_calib - mu_calib)
        q_abs = conformal_quantile(s_abs, ALPHA_MAIN, n_calib)
        picp_abs, mpiw_abs = picp_mpiw(y_test, mu_test - q_abs, mu_test + q_abs)
        q_abs_by_level, emp_abs = {}, []
        for p in CONF_LEVELS:
            q = conformal_quantile(s_abs, 1 - p, n_calib)
            q_abs_by_level[f"{p:.2f}"] = q
            lo, hi = mu_test - q, mu_test + q
            emp_abs.append(np.mean((y_test >= lo) & (y_test <= hi)))
        ece_abs = float(np.mean(np.abs(np.array(emp_abs) - CONF_LEVELS)))
        out['cp_abs'] = {'picp': picp_abs, 'mpiw': mpiw_abs, 'ece': ece_abs, 'q': q_abs,
                          'q_by_level': q_abs_by_level, 'deviation_from_090': abs(picp_abs - 0.90)}
    return out


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        CANON = json.load(f)

    ckpt_dir_leakfree = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

    # ---- 1. LSTM FD001/FD002/FD004 ----
    out1 = {}
    for ds in C.DATASETS:
        print(f"\n{'=' * 10} LSTM SplitCP {ds} {'=' * 10}")
        out1[ds] = []
        for seed in C.SEEDS:
            model = C.load_checkpoint_model(os.path.join(ckpt_dir_leakfree, f"{ds}_SplitCP_seed{seed}.pt"), device)
            r = eval_one(model, ds, seed, CANON, device, with_abs=True)
            out1[ds].append(r)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  seed={seed}: PICP(norm)={r['cp_norm']['picp']:.3f} MPIW={r['cp_norm']['mpiw']:.2f}")
    p1 = os.path.join(RESULTS_DIR, 'split_cp_leakfree.json')
    with open(p1, 'w') as fp:
        json.dump(out1, fp, indent=2, default=float)
    print(f"Saved -> {p1}")

    # ---- 2. LSTM FD003 (cp_norm only) ----
    print(f"\n{'=' * 10} LSTM SplitCP FD003 {'=' * 10}")
    out2 = []
    for seed in C.SEEDS:
        model = C.load_checkpoint_model(os.path.join(ckpt_dir_leakfree, f"FD003_SplitCP_seed{seed}.pt"), device)
        r = eval_one(model, 'FD003', seed, CANON, device, with_abs=False)
        out2.append(r)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  seed={seed}: PICP(norm)={r['cp_norm']['picp']:.3f} MPIW={r['cp_norm']['mpiw']:.2f}")
    p2 = os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json')
    with open(p2, 'w') as fp:
        json.dump(out2, fp, indent=2, default=float)
    print(f"Saved -> {p2}")

    # ---- 3. Transformer, all 4 datasets ----
    out3 = {}
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"\n{'=' * 10} Transformer SplitCP {ds} {'=' * 10}")
        out3[ds] = []
        for seed in C.SEEDS:
            model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)
            r = eval_one(model, ds, seed, CANON, device, with_abs=True)
            out3[ds].append(r)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  seed={seed}: PICP(norm)={r['cp_norm']['picp']:.3f} MPIW={r['cp_norm']['mpiw']:.2f}")
    p3 = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json')
    with open(p3, 'w') as fp:
        json.dump(out3, fp, indent=2, default=float)
    print(f"Saved -> {p3}")

    print("\nR8-B1 SplitCP re-eval complete (no retraining).")
