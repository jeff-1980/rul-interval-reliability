"""
Original (pre-audit) checkpoint training script, kept for provenance only
原 rul_nll_gpu.py / E3_run_save_ece.py 从未 torch.save 过模型权重，只留了预测 .npy。
本脚本原样复用 E3 的 LSTM 训练管线（同一套超参、同一 5 seeds、同一数据处理），
唯一区别：额外把 best_state（早停选出的最优权重）落盘到
results/checkpoints/ (this script's own output dir, not the release's checkpoints/lstm/)，供 STEP 1-5（MC Dropout 修正 / Deep Ensemble /
Split-CP / per-engine / 噪声敏感性）复用，避免每个 STEP 都重训。

只训 LSTM（不训 Transformer）：STEP 0 复现闸门表只列了单套 RMSE/PICP/MPIW/ECE，


不写入任何已有 results 子目录（铁律 3：旧结果只读）。
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

# ==========================================
# 路径配置
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)
RESULTS_DIR    = os.path.join(PROJ_DIR, 'results', 'superseded_generated')
CKPT_DIR   = os.path.join(RESULTS_DIR, 'checkpoints')
LOG_DIR    = os.path.join(RESULTS_DIR, 'logs')

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# GPU
# ==========================================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram_gb:.1f} GB")
        use_amp = True
    else:
        device = torch.device('cpu')
        use_amp = False
        print("CPU mode")
    return device, use_amp

# ==========================================
# 配置（与 CLAUDE.md / E3 锁定超参保持一致，逐字不改）
# ==========================================
class Config:
    def __init__(self, device, use_amp):
        self.data_dir        = '/home/jeffwork/rul_project/data'
        self.target_datasets = ['FD001', 'FD002', 'FD004']
        self.models_to_run   = ['LSTM']
        self.seeds            = [42, 2024, 7, 888, 123]

        self.sequence_length = 30
        self.max_rul         = 125
        self.batch_size      = 256 if device.type == 'cuda' else 32
        self.hidden_dim      = 64
        self.num_layers      = 2

        self.log_sigma_min = -3.0
        self.log_sigma_max =  2.0

        self.picp_confidence = 0.90
        self.z_score          = 1.645

        self.epochs      = 150
        self.lr          = 0.001
        self.device      = device
        self.use_amp     = use_amp
        self.pin_memory  = (device.type == 'cuda')
        self.num_workers = 4 if device.type == 'cuda' else 0

        self.index_names   = ['unit_nr', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names  = ['s_{}'.format(i) for i in range(1, 22)]
        self.fd001_feats   = [
            's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11',
            's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'
        ]
        self.conf_levels = np.arange(0.05, 1.00, 0.05)

# ==========================================
# 数据处理（与 E3 完全一致）
# ==========================================
class DataHandler:
    def __init__(self, config, dataset_name):
        self.cfg          = config
        self.dataset_name = dataset_name
        self.scaler       = MinMaxScaler(feature_range=(-1, 1))

    def get_feature_names(self):
        if self.dataset_name == 'FD001':
            return self.cfg.fd001_feats
        else:
            drop_cols = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
            sensors   = [s for s in self.cfg.sensor_names if s not in drop_cols]
            return self.cfg.setting_names + sensors

    def load_and_process(self):
        train_path = os.path.join(self.cfg.data_dir, f'train_{self.dataset_name}.txt')
        test_path  = os.path.join(self.cfg.data_dir, f'test_{self.dataset_name}.txt')
        rul_path   = os.path.join(self.cfg.data_dir, f'RUL_{self.dataset_name}.txt')

        col_names = self.cfg.index_names + self.cfg.setting_names + self.cfg.sensor_names
        train_df  = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
        test_df   = pd.read_csv(test_path,  sep=r'\s+', header=None, names=col_names)
        true_ruls = pd.read_csv(rul_path,   sep=r'\s+', header=None, names=['RUL'])

        max_cycles = train_df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_nr', 'max']
        train_df = train_df.merge(max_cycles, on='unit_nr', how='left')
        train_df['RUL'] = (train_df['max'] - train_df['time_cycles']).clip(upper=self.cfg.max_rul)

        feature_cols = self.get_feature_names()
        self.scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_df[feature_cols]  = self.scaler.transform(test_df[feature_cols])
        return train_df, test_df, true_ruls, feature_cols

    def create_sequences(self, df, feature_cols, mode='train', true_ruls=None):
        X_list, y_list = [], []
        for unit in df['unit_nr'].unique():
            unit_data = df[df['unit_nr'] == unit][feature_cols].values
            if mode == 'train':
                rul_arr = df[df['unit_nr'] == unit]['RUL'].values
                for i in range(len(unit_data) - self.cfg.sequence_length):
                    X_list.append(unit_data[i: i + self.cfg.sequence_length])
                    y_list.append(rul_arr[i + self.cfg.sequence_length] / 125.0)
            elif mode == 'test':
                if len(unit_data) >= self.cfg.sequence_length:
                    X_list.append(unit_data[-self.cfg.sequence_length:])
                    y_list.append(min(true_ruls.iloc[unit - 1].item(), 125))
        return np.array(X_list), np.array(y_list)

# ==========================================
# 模型（与 E3 完全一致）
# ==========================================
class HeteroscedasticLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout, log_sigma_min, log_sigma_max):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.lstm    = nn.LSTM(input_size, hidden_dim, num_layers=2,
                               batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mu_head        = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat, _ = self.lstm(x)
        feat    = self.dropout(feat[:, -1, :])
        mu        = self.mu_head(feat)
        log_sigma = torch.clamp(self.log_sigma_head(feat),
                                self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma

def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()

def calculate_metrics(y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    d     = y_pred - y_true
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return rmse, score

def calculate_picp_mpiw(y_true, y_mean, y_sigma, z=1.645):
    lower = y_mean - z * y_sigma
    upper = y_mean + z * y_sigma
    picp  = np.mean((y_true >= lower) & (y_true <= upper))
    mpiw  = np.mean(upper - lower)
    return picp, mpiw

def compute_ece(mu_all, sigma_all, ytrue_all, conf_levels):
    empirical = []
    for p in conf_levels:
        z    = stats.norm.ppf((1 + p) / 2)
        lo   = mu_all - z * sigma_all
        hi   = mu_all + z * sigma_all
        cov  = np.mean((ytrue_all >= lo) & (ytrue_all <= hi))
        empirical.append(cov)
    empirical = np.array(empirical)
    ece = np.mean(np.abs(empirical - conf_levels))
    return ece, empirical

# ==========================================
# 单 seed 训练 + checkpoint 落盘
# ==========================================
def run_one_seed(cfg, seed, data_bundle, input_dim, ds_name):
    gc.collect()
    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X_train, y_train, X_test, y_test = data_bundle

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers
    )
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(cfg.device)

    model = HeteroscedasticLSTM(
        input_dim, cfg.hidden_dim, dropout=0.2,
        log_sigma_min=cfg.log_sigma_min,
        log_sigma_max=cfg.log_sigma_max
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    amp_scaler = GradScaler(device='cuda', enabled=cfg.use_amp)

    best_rmse  = float('inf')
    best_state = None
    t0         = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(cfg.device, non_blocking=True)
            by = by.to(cfg.device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                mu, log_sigma = model(bx)
                loss          = gaussian_nll_loss(mu, log_sigma, by)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        with torch.no_grad():
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                mu_val, _ = model(X_test_t)
            mu_val    = mu_val.float().cpu().numpy().flatten() * 125.0
            mu_val    = np.clip(mu_val, 0, 125)
            curr_rmse = np.sqrt(mean_squared_error(y_test, mu_val))

        if curr_rmse < best_rmse:
            best_rmse  = curr_rmse
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
            mu_out, log_sigma_out = model(X_test_t)

    mu_np    = mu_out.float().cpu().numpy().flatten() * 125.0
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0
    mu_np    = np.clip(mu_np, 0, 125)

    rmse,  score = calculate_metrics(y_test, mu_np)
    picp,  mpiw  = calculate_picp_mpiw(y_test, mu_np, sigma_np, z=cfg.z_score)

    print(f"   -> RMSE={rmse:.3f}  Score={score:.1f}  "
          f"PICP={picp:.3f}  MPIW={mpiw:.2f}  sigma_mean={sigma_np.mean():.2f}  "
          f"Time={elapsed:.1f}s")

    # === 新增：落盘 checkpoint（原 E3 脚本从未做过这一步）===
    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_LSTM_seed{seed}.pt")
    torch.save({
        'state_dict':    best_state,
        'input_dim':     input_dim,
        'hidden_dim':    cfg.hidden_dim,
        'dropout':       0.2,
        'log_sigma_min': cfg.log_sigma_min,
        'log_sigma_max': cfg.log_sigma_max,
        'seed':          seed,
        'dataset':       ds_name,
        'best_val_rmse_scaled': best_rmse,
        'train_epochs':  cfg.epochs,
    }, ckpt_path)

    return rmse, score, picp, mpiw, sigma_np.mean(), elapsed, mu_np, sigma_np, y_test

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    device, use_amp = setup_device()
    cfg = Config(device, use_amp)

    print(f"\nlog_sigma in [{cfg.log_sigma_min}, {cfg.log_sigma_max}]  "
          f"epochs={cfg.epochs}  seeds={cfg.seeds}  models={cfg.models_to_run}")
    print(f"Saving checkpoints to: {CKPT_DIR}\n")

    all_results = []
    ece_rows    = []
    t_start_all = time.time()

    for ds_name in cfg.target_datasets:
        print(f"\n{'#'*20} Dataset: {ds_name} {'#'*20}")
        handler = DataHandler(cfg, ds_name)
        train_df, test_df, true_ruls, feat_cols = handler.load_and_process()

        X_train, y_train = handler.create_sequences(train_df, feat_cols, mode='train')
        X_test,  y_test  = handler.create_sequences(
            test_df, feat_cols, mode='test', true_ruls=true_ruls)
        data_bundle = (X_train, y_train, X_test, y_test)
        input_dim   = len(feat_cols)

        print(f"\n>>> LSTM on {ds_name}")
        metrics = {k: [] for k in ['rmse', 'score', 'picp', 'mpiw', 'sigma', 'time']}
        all_mu, all_sigma, all_ytrue = [], [], []

        for seed in cfg.seeds:
            print(f"\n  Seed {seed} ...", end=' ', flush=True)
            r, s, p, w, sg, t, mu_np, sigma_np, ytrue = run_one_seed(
                cfg, seed, data_bundle, input_dim, ds_name)

            metrics['rmse'].append(r)
            metrics['score'].append(s)
            metrics['picp'].append(p)
            metrics['mpiw'].append(w)
            metrics['sigma'].append(sg)
            metrics['time'].append(t)
            all_mu.append(mu_np); all_sigma.append(sigma_np); all_ytrue.append(ytrue)

            ece_seed, _ = compute_ece(mu_np, sigma_np, ytrue, cfg.conf_levels)
            ece_rows.append({'Dataset': ds_name, 'Model': 'LSTM', 'Seed': seed, 'ECE': round(ece_seed, 4)})
            print(f"  ECE(seed)={ece_seed:.4f}")

        mu_all    = np.concatenate(all_mu)
        sigma_all = np.concatenate(all_sigma)
        ytrue_all = np.concatenate(all_ytrue)
        ece_agg, _ = compute_ece(mu_all, sigma_all, ytrue_all, cfg.conf_levels)
        ece_rows.append({'Dataset': ds_name, 'Model': 'LSTM', 'Seed': 'agg', 'ECE': round(ece_agg, 4)})

        all_results.append({
            'Dataset':      ds_name,
            'Model':        'LSTM',
            'RMSE (Mean)':  np.mean(metrics['rmse']),
            'RMSE (Std)':   np.std(metrics['rmse'], ddof=1),
            'Score (Mean)': np.mean(metrics['score']),
            'Score (Std)':  np.std(metrics['score'], ddof=1),
            'PICP (Mean)':  np.mean(metrics['picp']),
            'PICP (Std)':   np.std(metrics['picp'], ddof=1),
            'MPIW (Mean)':  np.mean(metrics['mpiw']),
            'ECE (Agg)':    round(ece_agg, 4),
        })
        print(f"\n  [{ds_name}/LSTM] ECE(agg)={ece_agg:.4f}")

    total_elapsed = time.time() - t_start_all

    ece_df = pd.DataFrame(ece_rows)
    ece_df.to_csv(os.path.join(LOG_DIR, 'checkpoint_retrain_ECE.csv'), index=False)

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(LOG_DIR, 'checkpoint_retrain_summary.csv'), index=False)
    print("\n" + final_df.to_string(index=False))

    n_ckpt = len([f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')])
    print(f"\nCheckpoints saved: {n_ckpt} / {len(cfg.target_datasets) * len(cfg.seeds)} expected")
    print(f"Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("STEP0 retrain complete.")
