"""
T2-A2：Transformer 骨干，MSE 训练 + MC-Dropout(Kendall&Gal修正)，leakfree 协议，
4 数据集 × 5 seeds = 20 模型。

与 train_lstm_fd003_mse_mcdropout.py 逐字同一协议（纯MSE训练/canonical
fit-val选择/leakfree scaler/T=50主T=100附/aleatory_var=fit_units残差方差），
唯一区别是模型换成 T2.MC_Transformer，且循环全部4个数据集。
"""
import os
import json
import time
import random
import copy
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler
from scipy import stats

import common as C
import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
MC_T_MAIN = 50
MC_T_EXTRA = 100
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def picp_mpiw(yt, mu, sigma, z=C.Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((yt >= lo) & (yt <= hi))), float(np.mean(hi - lo))


def compute_ece(mu_all, sigma_all, ytrue_all, conf_levels):
    empirical = []
    for p in conf_levels:
        z = stats.norm.ppf((1 + p) / 2)
        lo = mu_all - z * sigma_all; hi = mu_all + z * sigma_all
        empirical.append(np.mean((ytrue_all >= lo) & (ytrue_all <= hi)))
    empirical = np.array(empirical)
    return float(np.mean(np.abs(empirical - conf_levels)))


def batched_forward(model, X_t, batch=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            with autocast(device_type=X_t.device.type, enabled=True):
                o = model(X_t[i:i + batch])
            outs.append(o.float().cpu().numpy())
    return np.concatenate(outs).flatten()


def run_one_seed(ds_name, seed, device, use_amp):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[ds_name][str(seed)]
    fit_units, val_units = split['fit_units'], split['val_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds_name, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')

    ckpt_path = T2.mc_ckpt_path('Transformer', ds_name, seed)
    if os.path.exists(ckpt_path):
        # 断点续跑：checkpoint 已存在（此前训练完成后系统重启中断了后续数据集/
        # 汇总json的写出），不重训，直接加载已训好的权重跳到评估阶段。
        print(f"   [{ds_name}] seed={seed}: checkpoint exists, resuming from disk (skip training)")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = T2.MC_Transformer(ck['input_dim'], ck['hidden_dim'], ck['dropout'], T2.SEQUENCE_LENGTH).to(device)
        model.load_state_dict(ck['state_dict'])
        best_val_rmse = ck['best_val_rmse_cycles']
        elapsed = ck.get('elapsed_train_s', 0.0)
    else:
        X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')

        loader = DataLoader(
            TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
            batch_size=T2.T2_BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
            num_workers=4 if device.type == 'cuda' else 0)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_cycles = y_val * 125.0

        model = T2.MC_Transformer(input_dim, T2.T2_HIDDEN_DIM, T2.T2_DROPOUT, T2.SEQUENCE_LENGTH).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=T2.T2_LR)
        sc = GradScaler(device='cuda', enabled=use_amp)
        crit = nn.MSELoss()
        best_val_rmse, best_state = float('inf'), None
        t0 = time.time()

        for _ in range(T2.T2_EPOCHS):
            model.train()
            for bx, by in loader:
                bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp):
                    loss = crit(model(bx), by)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
            model.eval()
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=use_amp):
                    mv = model(X_val_t)
                mv = np.clip(mv.float().cpu().numpy().flatten() * 125, 0, 125)
                r = np.sqrt(mean_squared_error(y_val_cycles, mv))
            if r < best_val_rmse:
                best_val_rmse = r; best_state = copy.deepcopy(model.state_dict())
        elapsed = time.time() - t0
        model.load_state_dict(best_state)

        torch.save({'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': T2.T2_HIDDEN_DIM,
                    'dropout': T2.T2_DROPOUT, 'seed': seed, 'dataset': ds_name,
                    'fit_units': fit_units, 'val_units': val_units,
                    'best_val_rmse_cycles': best_val_rmse, 'train_epochs': T2.T2_EPOCHS,
                    'elapsed_train_s': elapsed,
                    'selection_protocol': 'canonical_split fit/val, leakfree, Transformer backbone (T2, 2026-09-18)'},
                   ckpt_path)

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    yhat_fit_scaled = batched_forward(model, X_fit_t)
    yhat_fit = np.clip(yhat_fit_scaled * 125.0, 0, 125)
    y_fit_true = y_fit * 125.0
    resid = y_fit_true - yhat_fit
    aleatory_var = float(np.var(resid, ddof=1))
    del X_fit_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.train()
    T_max = MC_T_EXTRA
    samples = []
    lat_t0 = time.time()
    with torch.no_grad():
        for _ in range(T_max):
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(X_test_t)
            samples.append(out.float().cpu().numpy().flatten() * 125.0)
    lat_total_100 = time.time() - lat_t0
    samples = np.stack(samples)

    def eval_variant(T):
        s = samples[:T]
        mu = np.clip(s.mean(0), 0, 125)
        eps_var = s.var(0)
        sigma_sampling = np.sqrt(eps_var)
        rmse, score = C.rmse_score(y_test, mu)
        p_s, w_s = picp_mpiw(y_test, mu, sigma_sampling)
        ece_s = compute_ece(mu, sigma_sampling, y_test, CONF_LEVELS)
        sigma_full = np.sqrt(aleatory_var + eps_var)
        p_f, w_f = picp_mpiw(y_test, mu, sigma_full)
        ece_f = compute_ece(mu, sigma_full, y_test, CONF_LEVELS)
        return {'T': T, 'rmse': rmse, 'score': score,
                'sampling_only': {'picp': p_s, 'mpiw': w_s, 'ece': ece_s, 'sigma_mean': float(sigma_sampling.mean())},
                'kendall_gal_full': {'picp': p_f, 'mpiw': w_f, 'ece': ece_f, 'sigma_mean': float(sigma_full.mean()),
                                      'aleatory_var': aleatory_var, 'epistemic_var_mean': float(eps_var.mean())}}

    res_T50 = eval_variant(MC_T_MAIN)
    res_T100 = eval_variant(MC_T_EXTRA)
    lat_per_sample_T50_ms = (lat_total_100 * (MC_T_MAIN / MC_T_EXTRA)) / len(y_test) * 1000.0

    model.eval()
    mu_mse = batched_forward(model, X_test_t)
    mu_mse = np.clip(mu_mse * 125.0, 0, 125)
    sigma_fixed = float(np.sqrt(aleatory_var))
    sigma_arr = np.full_like(mu_mse, sigma_fixed)
    rmse_mse, score_mse = C.rmse_score(y_test, mu_mse)
    picp_mse, mpiw_mse = picp_mpiw(y_test, mu_mse, sigma_arr)
    ece_mse = compute_ece(mu_mse, sigma_arr, y_test, CONF_LEVELS)

    print(f"   [{ds_name}] seed={seed} n_fit={len(fit_units)} n_val={len(val_units)} train={elapsed:.1f}s "
          f"best_val_rmse={best_val_rmse:.3f}  [MSE] RMSE={rmse_mse:.3f} PICP={picp_mse:.3f} "
          f"[MC-Dropout T50] PICP={res_T50['kendall_gal_full']['picp']:.3f}")

    return {'seed': seed, 'elapsed_train_s': elapsed, 'best_val_rmse': best_val_rmse,
            'T50': res_T50, 'T100': res_T100, 'latency_ms_per_sample_T50': lat_per_sample_T50_ms,
            'mse_row': {'rmse': rmse_mse, 'score': score_mse, 'picp': picp_mse, 'mpiw': mpiw_mse,
                        'ece': ece_mse, 'sigma_fixed': sigma_fixed}}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  Backbone=Transformer(MSE+MCD)  DATASETS={DATASETS}  SEEDS={C.SEEDS}")

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json')
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as fp:
            all_out = json.load(fp)
        print(f"Resuming: found existing partial results for {list(all_out.keys())}")

    for ds in DATASETS:
        if ds in all_out and len(all_out[ds]) == len(C.SEEDS):
            print(f"\n{'=' * 20} {ds} (already complete, skipping) {'=' * 20}")
            continue
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        all_out[ds] = [run_one_seed(ds, seed, device, use_amp) for seed in C.SEEDS]
        # 每个数据集跑完立即落盘，避免系统重启（此前发生过一次）再丢失整段进度
        with open(out_path, 'w') as fp:
            json.dump(all_out, fp, indent=2, default=float)
        print(f"  [checkpoint] saved partial results -> {out_path}")

    print(f"\nSaved -> {out_path}")
    print("T2-A2 (Transformer MSE+MCDropout) complete.")
