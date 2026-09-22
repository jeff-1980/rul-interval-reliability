"""
STEP 1b：MC-Dropout/MSE checkpoint 的测试集泄漏修复（与 STEP0c 同一套
canonical_splits.json，fit_units/val_units 逐发动机与 STEP0c/STEP3b 一致）。

原 `mc_dropout_model.py` 同样每 epoch 在官方测试集上算 RMSE 选
best_state，且 scaler 在全部 train 发动机（含 val）上拟合——两个问题与
STEP0 完全同构，修复方式也完全同构：checkpoint 选择改用 val 集 RMSE，
scaler 改用 `C.load_and_process_leakfree(ds, fit_units)`。

架构/loss/aleatory方差定义与原 STEP1 逐字一致（MC_LSTM，纯MSE训练，
T=50/100 dropout 采样，aleatory_var=训练残差方差——但"训练残差"现在必须
用 fit_units 的残差算，不能再用全部train数据的残差，因为原定义就是
"var(y_train - yhat_train)"，train 在新协议下等价于 fit_units）。

checkpoint 存到 `checkpoints_leakfree/`（与 STEP0c/STEP3b 共用同一目录，
文件名前缀区分方法）。
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
import mc_dropout_model as S1  # reuse MC_LSTM class definition, unchanged

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = C.SEEDS
DATASETS = C.DATASETS
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
HIDDEN_DIM = C.HIDDEN_DIM
Z_SCORE = C.Z_SCORE
MC_T_MAIN = 50
MC_T_EXTRA = 100
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)

with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)


def picp_mpiw(yt, mu, sigma, z=Z_SCORE):
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
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')

    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = S1.MC_LSTM(input_dim, HIDDEN_DIM, 0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sc = GradScaler(device='cuda', enabled=use_amp)
    crit = nn.MSELoss()
    best_val_rmse, best_state = float('inf'), None
    t0 = time.time()

    for _ in range(EPOCHS):
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

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_MCDropoutMSE_seed{seed}.pt")
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
        'dropout': 0.2, 'seed': seed, 'dataset': ds_name,
        'fit_units': fit_units, 'val_units': val_units,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
        'selection_protocol': 'canonical_split fit/val, leakfree scaler (2026-09-15)',
    }, ckpt_path)

    # ---- aleatory: fit_units 残差方差（eval, no dropout） ----
    X_fit_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    yhat_fit_scaled = batched_forward(model, X_fit_t)
    yhat_fit = np.clip(yhat_fit_scaled * 125.0, 0, 125)
    y_fit_true = y_fit * 125.0
    resid = y_fit_true - yhat_fit
    aleatory_var = float(np.var(resid, ddof=1))
    del X_fit_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- 官方测试集：评价一次 ----
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.train()  # keep dropout active for MC sampling
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

    print(f"   seed={seed}  n_fit={len(fit_units)} n_val={len(val_units)}  train={elapsed:.1f}s  "
          f"best_val_rmse={best_val_rmse:.3f}  "
          f"[test] T50 kendall_gal_full PICP={res_T50['kendall_gal_full']['picp']:.3f}")

    return {'seed': seed, 'elapsed_train_s': elapsed, 'best_val_rmse': best_val_rmse,
            'T50': res_T50, 'T100': res_T100, 'latency_ms_per_sample_T50': lat_per_sample_T50_ms}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"STEP1b: leakfree retrain, seeds={SEEDS}  CKPT_DIR={CKPT_DIR}")

    all_out = {}
    for ds in DATASETS:
        print(f"\n{'='*20} {ds} {'='*20}")
        seed_results = []
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=' ', flush=True)
            r = run_one_seed(ds, seed, device, use_amp)
            seed_results.append(r)
        all_out[ds] = seed_results

    out_path = os.path.join(RESULTS_DIR, 'mcdropout_fixed_leakfree.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    print("\n=== 汇总（mean across 5 seeds, T=50） ===")
    for ds in DATASETS:
        rs = all_out[ds]
        picp_new = np.mean([r['T50']['kendall_gal_full']['picp'] for r in rs])
        mpiw_new = np.mean([r['T50']['kendall_gal_full']['mpiw'] for r in rs])
        print(f"{ds}: kendall_gal_full PICP={picp_new:.3f}  MPIW={mpiw_new:.2f}")
    print("\nSTEP1b complete.")
