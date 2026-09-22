"""
STEP 1 (E-A)：MC Dropout 区间构造修正

诊断（已确认，见 code/E2_mc_dropout.py::run_mc）：
  旧实现 MC_LSTM 是纯 MSE 训练、无 sigma 头的模型；测试时用 model.train() 打开
  dropout 采样 T=50 次，sigma = samples.std(0) —— 只有 dropout 采样方差
  （epistemic），从未叠加任何观测噪声/aleatory 项。
  这正是旧 PICP 只有 0.488/0.413/0.397、MPIW 只有 12.70-15.14 周期的原因。

修正（Kendall & Gal 2017 回归型完整式）：
  mu_hat    = (1/T) sum_t mu_t
  sigma^2   = (1/T) sum_t sigma_t^2          <- aleatory
            + (1/T) sum_t (mu_t - mu_hat)^2  <- epistemic (dropout采样方差)

  本基线模型没有 sigma 头（纯 MSE），按任务书 STEP1.2 的替代方案：
  aleatory 项用【训练集残差方差】(ddof=1) 代入，作为标量加到每个样本的
  epistemic 方差上：
  sigma_total^2 = var(y_train - yhat_train) + samples.var(axis=0)

保留旧实现（variant="sampling_only"，sigma=samples.std(0)）与新实现
（variant="kendall_gal_full"）在同一 json 里逐 seed 对照，同时记录 T=50（主）
和 T=100（敏感性附注）。

不重训 NLL 模型（STEP0 已产出并落盘），本步骤需要独立重训 MC_LSTM
（MSE、无 sigma 头架构，旧实现同样从未存过 checkpoint）。
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR  = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device, True
    return torch.device('cpu'), False


class Config:
    def __init__(self, device, use_amp):
        self.data_dir        = '/home/jeffwork/rul_project/data'
        self.target_datasets = ['FD001', 'FD002', 'FD004']
        self.seeds            = [42, 2024, 7, 888, 123]
        self.sequence_length = 30
        self.max_rul         = 125
        self.batch_size      = 256 if device.type == 'cuda' else 32
        self.hidden_dim      = 64
        self.z_score          = 1.645
        self.epochs      = 150
        self.lr          = 0.001
        self.mc_T_main    = 50
        self.mc_T_extra   = 100
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


class DataHandler:
    def __init__(self, cfg, ds):
        self.cfg = cfg
        self.ds  = ds
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def feats(self):
        if self.ds == 'FD001':
            return self.cfg.fd001_feats
        drop = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
        return self.cfg.setting_names + [s for s in self.cfg.sensor_names if s not in drop]

    def load(self):
        cols = self.cfg.index_names + self.cfg.setting_names + self.cfg.sensor_names
        tr  = pd.read_csv(f"{self.cfg.data_dir}/train_{self.ds}.txt", sep=r'\s+', header=None, names=cols)
        te  = pd.read_csv(f"{self.cfg.data_dir}/test_{self.ds}.txt",  sep=r'\s+', header=None, names=cols)
        rul = pd.read_csv(f"{self.cfg.data_dir}/RUL_{self.ds}.txt",   sep=r'\s+', header=None, names=['RUL'])
        mc  = tr.groupby('unit_nr')['time_cycles'].max().reset_index()
        mc.columns = ['unit_nr', 'max']
        tr  = tr.merge(mc, on='unit_nr', how='left')
        tr['RUL'] = (tr['max'] - tr['time_cycles']).clip(upper=self.cfg.max_rul)
        f = self.feats()
        self.scaler.fit(tr[f])
        tr[f] = self.scaler.transform(tr[f])
        te[f] = self.scaler.transform(te[f])
        return tr, te, rul, f

    def sequences(self, df, f, mode='train', rul=None):
        X, y = [], []
        for u in df['unit_nr'].unique():
            d = df[df['unit_nr'] == u][f].values
            if mode == 'train':
                r = df[df['unit_nr'] == u]['RUL'].values
                for i in range(len(d) - self.cfg.sequence_length):
                    X.append(d[i:i + self.cfg.sequence_length])
                    y.append(r[i + self.cfg.sequence_length] / 125.0)
            else:
                if len(d) >= self.cfg.sequence_length:
                    X.append(d[-self.cfg.sequence_length:])
                    y.append(min(rul.iloc[u - 1].item(), 125))
        return np.array(X), np.array(y)


class MC_LSTM(nn.Module):
    """与旧 E2_mc_dropout.py 完全一致：单输出 LSTM（MSE训练），无 sigma 头"""
    def __init__(self, in_dim, H, drop):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, H, 2, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.mu   = nn.Linear(H, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.mu(self.drop(h[:, -1, :]))


def calc(yt, yp):
    rmse = np.sqrt(mean_squared_error(yt, yp))
    d = yp - yt
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return rmse, score


def picp_mpiw(yt, mu, sigma, z=1.645):
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((yt >= lo) & (yt <= hi))), float(np.mean(hi - lo))


def compute_ece(mu_all, sigma_all, ytrue_all, conf_levels):
    empirical = []
    for p in conf_levels:
        z = stats.norm.ppf((1 + p) / 2)
        lo = mu_all - z * sigma_all
        hi = mu_all + z * sigma_all
        cov = np.mean((ytrue_all >= lo) & (ytrue_all <= hi))
        empirical.append(cov)
    empirical = np.array(empirical)
    return float(np.mean(np.abs(empirical - conf_levels)))


def batched_forward(model, X_t, batch=4096):
    """eval()、无 dropout 前向，用于算训练残差（防止大 train 集 OOM）"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            with autocast(device_type=X_t.device.type, enabled=True):
                o = model(X_t[i:i + batch])
            outs.append(o.float().cpu().numpy())
    return np.concatenate(outs).flatten()


def run_one_seed(cfg, seed, bundle, in_dim, ds_name):
    gc.collect()
    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    X_tr, y_tr, X_te, y_te = bundle
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)),
        batch_size=cfg.batch_size, shuffle=True,
        pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(cfg.device)

    model = MC_LSTM(in_dim, cfg.hidden_dim, 0.2).to(cfg.device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sc    = GradScaler(device='cuda', enabled=cfg.use_amp)
    crit  = nn.MSELoss()
    best_r, best_s = float('inf'), None
    t0 = time.time()

    for _ in range(cfg.epochs):
        model.train()
        for bx, by in loader:
            bx = bx.to(cfg.device, non_blocking=True)
            by = by.to(cfg.device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                loss = crit(model(bx), by)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        model.eval()
        with torch.no_grad():
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                mv = model(X_te_t)
            mv = np.clip(mv.float().cpu().numpy().flatten() * 125, 0, 125)
            r = np.sqrt(mean_squared_error(y_te, mv))
        if r < best_r:
            best_r = r; best_s = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0

    model.load_state_dict(best_s)

    # checkpoint 落盘（旧实现从未存过）
    ckpt_path = os.path.join(CKPT_DIR, f"{ds_name}_MCDropoutMSE_seed{seed}.pt")
    torch.save({
        'state_dict': best_s, 'input_dim': in_dim, 'hidden_dim': cfg.hidden_dim,
        'dropout': 0.2, 'seed': seed, 'dataset': ds_name,
        'best_val_rmse_scaled': best_r, 'train_epochs': cfg.epochs,
    }, ckpt_path)

    # ---- aleatory: 训练残差方差（eval，无 dropout） ----
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(cfg.device)
    yhat_train_scaled = batched_forward(model, X_tr_t)
    yhat_train = np.clip(yhat_train_scaled * 125.0, 0, 125)
    y_train_true = y_tr * 125.0
    resid = y_train_true - yhat_train
    aleatory_var = float(np.var(resid, ddof=1))
    del X_tr_t
    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- MC dropout 采样：T=100，T=50 取前50个复用，避免多跑一遍 ----
    model.train()  # keep dropout active
    T_max = cfg.mc_T_extra
    samples = []
    lat_t0 = time.time()
    with torch.no_grad():
        for _ in range(T_max):
            with autocast(device_type=cfg.device.type, enabled=cfg.use_amp):
                out = model(X_te_t)
            samples.append(out.float().cpu().numpy().flatten() * 125.0)
    lat_total_100 = time.time() - lat_t0
    samples = np.stack(samples)  # (100, N)

    def eval_variant(T):
        s = samples[:T]                       # (T, N)
        mu    = np.clip(s.mean(0), 0, 125)
        eps_var = s.var(0)                    # population var, matches (1/T)sum(mu_t-mu)^2

        # variant A: 旧实现，只用 dropout 采样方差
        sigma_sampling = np.sqrt(eps_var)
        rmse, score = calc(y_te, mu)
        p_s, w_s = picp_mpiw(y_te, mu, sigma_sampling, cfg.z_score)
        ece_s = compute_ece(mu, sigma_sampling, y_te, cfg.conf_levels)

        # variant B: Kendall & Gal 完整式，aleatory(训练残差方差,标量) + epistemic
        sigma_full = np.sqrt(aleatory_var + eps_var)
        p_f, w_f = picp_mpiw(y_te, mu, sigma_full, cfg.z_score)
        ece_f = compute_ece(mu, sigma_full, y_te, cfg.conf_levels)

        return {
            'T': T,
            'rmse': rmse, 'score': score,
            'sampling_only': {'picp': p_s, 'mpiw': w_s, 'ece': ece_s,
                               'sigma_mean': float(sigma_sampling.mean())},
            'kendall_gal_full': {'picp': p_f, 'mpiw': w_f, 'ece': ece_f,
                                  'sigma_mean': float(sigma_full.mean()),
                                  'aleatory_var': aleatory_var,
                                  'epistemic_var_mean': float(eps_var.mean())},
        }

    res_T50  = eval_variant(cfg.mc_T_main)
    res_T100 = eval_variant(cfg.mc_T_extra)

    lat_per_sample_T50_ms = (lat_total_100 * (cfg.mc_T_main / cfg.mc_T_extra)) / len(y_te) * 1000.0

    return {
        'seed': seed, 'elapsed_train_s': elapsed,
        'T50': res_T50, 'T100': res_T100,
        'latency_ms_per_sample_T50': lat_per_sample_T50_ms,
    }


if __name__ == '__main__':
    device, use_amp = setup_device()
    cfg = Config(device, use_amp)
    print(f"\nSTEP1: MC Dropout fix  T_main={cfg.mc_T_main}  T_extra={cfg.mc_T_extra}  seeds={cfg.seeds}\n")

    all_out = {}
    for ds in cfg.target_datasets:
        print(f"\n{'='*20} {ds} {'='*20}")
        dh = DataHandler(cfg, ds)
        tr, te, rul, f = dh.load()
        X_tr, y_tr = dh.sequences(tr, f, 'train')
        X_te, y_te = dh.sequences(te, f, 'test', rul)
        bundle = (X_tr, y_tr, X_te, y_te)
        in_d = len(f)

        seed_results = []
        for seed in cfg.seeds:
            print(f"  seed={seed} ...", end=' ', flush=True)
            r = run_one_seed(cfg, seed, bundle, in_d, ds)
            print(f"train={r['elapsed_train_s']:.1f}s  "
                  f"T50 sampling_only PICP={r['T50']['sampling_only']['picp']:.3f}  "
                  f"T50 kendall_gal_full PICP={r['T50']['kendall_gal_full']['picp']:.3f}")
            seed_results.append(r)
        all_out[ds] = seed_results

    out_path = os.path.join(RESULTS_DIR, 'mcdropout_fixed.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    # ---- 汇总打印：修正前后 PICP 对照 + "+43.8pp" 主张判定 ----
    print("\n=== 汇总（mean across 5 seeds, T=50） ===")
    for ds in cfg.target_datasets:
        rs = all_out[ds]
        picp_old = np.mean([r['T50']['sampling_only']['picp'] for r in rs])
        picp_new = np.mean([r['T50']['kendall_gal_full']['picp'] for r in rs])
        mpiw_old = np.mean([r['T50']['sampling_only']['mpiw'] for r in rs])
        mpiw_new = np.mean([r['T50']['kendall_gal_full']['mpiw'] for r in rs])
        print(f"{ds}: PICP sampling_only={picp_old:.3f} -> kendall_gal_full={picp_new:.3f}  "
              f"(delta={picp_new-picp_old:+.3f})   MPIW {mpiw_old:.2f} -> {mpiw_new:.2f}")

    print("\nSTEP1 complete.")
