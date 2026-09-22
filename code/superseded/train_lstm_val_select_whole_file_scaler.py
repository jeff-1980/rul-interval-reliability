"""
STEP 0b：修复测试集泄漏——checkpoint 选择改用 train 侧切出的 val 集。

背景：`train_lstm_leaked_test_select_whole_file_scaler.py` 每个 epoch
结束都在官方测试集（test_{ds}.txt + RUL_{ds}.txt）上算 RMSE，150 个 epoch
里最优的那个存为 checkpoint——这是模型选择阶段的测试集泄漏：梯度从未碰
测试集，但"选哪个epoch保留"这个决策本身用了测试集表现，等价于用测试集
做了一次隐式的超参数搜索（150个"候选模型"里选测试集表现最好的一个）。

修复：按发动机单元把 train_{ds}.txt 切成 fit(80%)/val(20%)（种子派生，
`common.split_units_two_way`，与 STEP3 现有的 fit/calib 切分用同一套
逻辑，避免全项目出现两套不一致的切分实现）。val 集用全滑窗（与训练同分布，
覆盖每台发动机从早期到临近失效的全部RUL区间，比"每台发动机一个截断点"
的测试协议给出的早停信号更稳定，不依赖某个具体截断点的运气）。checkpoint
选择只看 val RMSE。训练结束后，用被选中的 best_state 在官方测试集上**评价
一次**（不参与任何选择决策），这是本次修复后测试集唯一被触碰的地方。

其余全部保持与 STEP0 逐字一致：同一套超参（150 epoch, batch 256, lr 1e-3,
hidden_dim 64, log_sigma∈[-3,2], dropout 0.2）、同一个 HeteroscedasticLSTM
架构、同一个 gaussian_nll_loss、同一 5 seeds、同一特征选择。

checkpoint 落盘到**新目录** `checkpoints_valselect/`（不覆盖 `checkpoints/`
里已标记 COMPROMISED 的旧文件，保留旧结果供退化幅度对比）。
"""
import os
import json
import time
import random
import copy
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler
from scipy import stats

import common as C  # sibling module in code/, not code/superseded/ -- see superseded/README.md

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'superseded_generated')
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints_valselect')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEEDS = C.SEEDS
DATASETS = C.DATASETS
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
HIDDEN_DIM = C.HIDDEN_DIM
LOG_SIGMA_MIN = C.LOG_SIGMA_MIN
LOG_SIGMA_MAX = C.LOG_SIGMA_MAX
Z_SCORE = C.Z_SCORE
VAL_FRAC = 0.20
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)


def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()


def run_one_seed(ds_name, seed, device, use_amp, train_df, test_df, true_ruls, feat_cols):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    all_units = train_df['unit_nr'].unique().tolist()
    fit_units, val_units = C.split_units_two_way(all_units, VAL_FRAC, seed)

    X_fit, y_fit = C.create_sequences(
        train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(
        train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')
    input_dim = len(feat_cols)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0  # create_sequences(mode='train') scales y by /125

    model = C.HeteroscedasticLSTM(input_dim, HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=LOG_SIGMA_MIN, log_sigma_max=LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                mu, log_sigma = model(bx)
                loss = gaussian_nll_loss(mu, log_sigma, by)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mu_val, _ = model(X_val_t)
            mu_val = mu_val.float().cpu().numpy().flatten() * 125.0
            mu_val = np.clip(mu_val, 0, 125)
            curr_val_rmse = np.sqrt(mean_squared_error(y_val_cycles, mu_val))

        if curr_val_rmse < best_val_rmse:
            best_val_rmse = curr_val_rmse
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()

    # ---- 官方测试集：本次修复后唯一被触碰的地方，只评价，不参与任何选择 ----
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            mu_out, log_sigma_out = model(X_test_t)
    mu_np = mu_out.float().cpu().numpy().flatten() * 125.0
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0
    mu_np = np.clip(mu_np, 0, 125)

    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, CONF_LEVELS)

    print(f"   seed={seed}  n_fit_units={len(fit_units)} n_val_units={len(val_units)} "
          f"n_val_windows={len(y_val)}  train={elapsed:.1f}s  best_val_rmse={best_val_rmse:.3f}")
    print(f"      [official test, evaluated once] RMSE={rmse:.3f} Score={score:.1f} "
          f"PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_seed{seed}.pt")
    torch.save({
        'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': HIDDEN_DIM,
        'dropout': 0.2, 'log_sigma_min': LOG_SIGMA_MIN, 'log_sigma_max': LOG_SIGMA_MAX,
        'seed': seed, 'dataset': ds_name, 'fit_units': sorted(fit_units),
        'val_units': sorted(val_units), 'val_frac': VAL_FRAC,
        'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
        'selection_protocol': 'val_split_from_train_units (leakage-fixed, 2026-09-14)',
    }, ckpt_path)

    return {
        'seed': seed, 'n_fit_units': len(fit_units), 'n_val_units': len(val_units),
        'n_val_windows': int(len(y_val)), 'best_val_rmse': best_val_rmse,
        'elapsed_train_s': elapsed,
        'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
        'sigma_mean': float(sigma_np.mean()),
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  VAL_FRAC={VAL_FRAC}  seeds={SEEDS}  epochs={EPOCHS}")
    print(f"Checkpoint dir: {CKPT_DIR} (NEW, does not overwrite compromised checkpoints/)")

    all_out = {}
    t_start_all = time.time()
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)
        seed_results = []
        for seed in SEEDS:
            r = run_one_seed(ds, seed, device, use_amp, train_df, test_df, true_ruls, feat_cols)
            seed_results.append(r)
        all_out[ds] = seed_results

    total_elapsed = time.time() - t_start_all

    out_path = os.path.join(RESULTS_DIR, 'step0b_valselect_results.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    rows = []
    for ds in DATASETS:
        rs = all_out[ds]
        rmse_arr = np.array([r['rmse'] for r in rs])
        score_arr = np.array([r['score'] for r in rs])
        picp_arr = np.array([r['picp'] for r in rs])
        mpiw_arr = np.array([r['mpiw'] for r in rs])
        ece_arr = np.array([r['ece'] for r in rs])
        rows.append({
            'Dataset': ds, 'Model': 'LSTM',
            'RMSE (Mean)': rmse_arr.mean(), 'RMSE (Std)': rmse_arr.std(ddof=1),
            'Score (Mean)': score_arr.mean(), 'Score (Std)': score_arr.std(ddof=1),
            'PICP (Mean)': picp_arr.mean(), 'PICP (Std)': picp_arr.std(ddof=1),
            'MPIW (Mean)': mpiw_arr.mean(), 'MPIW (Std)': mpiw_arr.std(ddof=1),
            'ECE (Mean)': ece_arr.mean(), 'ECE (Std)': ece_arr.std(ddof=1),
        })
    final_df = pd.DataFrame(rows)
    final_df.to_csv(os.path.join(LOG_DIR, 'step0b_valselect_summary.csv'), index=False)
    print("\n" + final_df.to_string(index=False))

    n_ckpt = len([f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')])
    print(f"\nCheckpoints saved: {n_ckpt} / {len(DATASETS) * len(SEEDS)} expected")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print("STEP0b (leakage-fixed retrain) complete.")
