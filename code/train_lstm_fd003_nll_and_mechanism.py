"""
FD003 补充：§6.2 基线clamp_frac vs σ̂贡献占比对应关系的第四个点。

范围说明（与用户指令对齐，刻意收窄）：只训练 NLL 与 CP-norm（clamp_frac
表1 + 冻结σ̂分解只需要这两个有σ̂头的方法），不训练 MC-Dropout/不算
Deep Ensemble/MSE——这些不是"基线clamp_frac vs σ̂贡献占比"这个具体
交叉验证点需要的东西，为控制本次追加的计算规模，明确不做，不是遗漏。
若后续需要FD003完整代价表（含MC-Dropout/Ensemble/MSE/per-engine），
是独立的、更大的任务，需要用户另外确认再做。

数据来源：FD003官方C-MAPSS数据集，此前从未被拷入本项目 DATA_DIR
（/home/jeffwork/rul_project/data/），2026-09-18 从
a separate local copy of（同一份标准NASA C-MAPSS发行版，
train/test/RUL三个文件行数与格式核实一致：100 train engines,
100 test engines匹配RUL_FD003.txt的100行,26列标准格式）拷贝补齐，
不是新造数据。

工况核实：KMeans k=6 vs k=1 惯性比=0.074，setting_3恒为100，与FD001
同构（单一工况）——`common.get_feature_names`已相应更新，FD003复用
FD001的14特征集，不用FD002/FD004那套"丢6个随工况变化传感器"的选择
（那是为多工况场景设计的，FD003不适用）。

协议：与FD001/FD002/FD004完全一致的canonical三向切分(fit60%/val20%/
calib20%)+leakfree scaler，5个canonical seeds。因单一工况，臂A退化为
臂B（与FD001同理）。
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.amp import autocast, GradScaler

import common as C
import noise_injection as V4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
LEAKFREE_DIR = os.path.join(RESULTS_DIR, 'leakfree')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')
os.makedirs(LEAKFREE_DIR, exist_ok=True)

DS = 'FD003'
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 256
Z_SCORE = C.Z_SCORE
ALPHA_MAIN = 0.10
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)
SNR_LEVELS_ALL = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0, -5, -10]
PCT_LEVELS = V4.PCT_LEVELS
N_TRIALS = V4.N_TRIALS
CLAMP_EPS = V4.CLAMP_EPS

CANON_PATH = os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')
with open(CANON_PATH) as f:
    CANON = json.load(f)


def gaussian_nll_loss(mu, log_sigma, y_true):
    sigma = torch.exp(log_sigma)
    return (log_sigma + 0.5 * ((y_true - mu) / sigma) ** 2).mean()


# ---------------------------------------------------------------------
# STEP 1: extend canonical_splits.json with FD003
# ---------------------------------------------------------------------
def extend_canonical_splits():
    col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
    train_df_raw = pd.read_csv(os.path.join(C.DATA_DIR, f'train_{DS}.txt'), sep=r'\s+', header=None, names=col_names)
    all_units = sorted(train_df_raw['unit_nr'].unique().tolist())
    CANON.setdefault(DS, {})
    for seed in C.SEEDS:
        fit_units, val_units, calib_units = C.compute_canonical_split(all_units, seed)
        assert set(fit_units) | set(val_units) | set(calib_units) == set(all_units)
        assert not (set(fit_units) & set(val_units))
        assert not (set(fit_units) & set(calib_units))
        assert not (set(val_units) & set(calib_units))
        CANON[DS][str(seed)] = {'fit_units': fit_units, 'val_units': val_units, 'calib_units': calib_units}
    with open(CANON_PATH, 'w') as fp:
        json.dump(CANON, fp, indent=2)
    print(f"canonical_splits.json extended with {DS}: "
          f"fit={len(CANON[DS][str(C.SEEDS[0])]['fit_units'])} "
          f"val={len(CANON[DS][str(C.SEEDS[0])]['val_units'])} "
          f"calib={len(CANON[DS][str(C.SEEDS[0])]['calib_units'])} (total={len(all_units)})")


# ---------------------------------------------------------------------
# STEP 2: train NLL (5 seeds, leakfree)
# ---------------------------------------------------------------------
def train_nll_one_seed(seed, device, use_amp):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[DS][str(seed)]
    fit_units, val_units = split['fit_units'], split['val_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(DS, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val, y_val = C.create_sequences(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, mode='train')

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
        num_workers=4 if device.type == 'cuda' else 0)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_cycles = y_val * 125.0

    model = C.HeteroscedasticLSTM(input_dim, C.HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=C.LOG_SIGMA_MIN, log_sigma_max=C.LOG_SIGMA_MAX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler(device='cuda', enabled=use_amp)

    best_val_rmse, best_state = float('inf'), None
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                mu, log_sigma = model(bx)
                loss = gaussian_nll_loss(mu, log_sigma, by)
            amp_scaler.scale(loss).backward(); amp_scaler.step(optimizer); amp_scaler.update()
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mu_val, _ = model(X_val_t)
            mu_val = np.clip(mu_val.float().cpu().numpy().flatten() * 125.0, 0, 125)
            curr_val_rmse = np.sqrt(mean_squared_error(y_val_cycles, mu_val))
        if curr_val_rmse < best_val_rmse:
            best_val_rmse = curr_val_rmse; best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0
    model.load_state_dict(best_state); model.eval()

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            mu_out, log_sigma_out = model(X_test_t)
    mu_np = np.clip(mu_out.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_np = torch.exp(log_sigma_out).float().cpu().numpy().flatten() * 125.0
    rmse, score = C.rmse_score(y_test, mu_np)
    picp, mpiw = C.picp_mpiw(y_test, mu_np, sigma_np, z=Z_SCORE)
    ece = C.compute_ece(mu_np, sigma_np, y_test, CONF_LEVELS)
    print(f"   [NLL] seed={seed} n_fit={len(fit_units)} n_val={len(val_units)} train={elapsed:.1f}s "
          f"best_val_rmse={best_val_rmse:.3f} [test] RMSE={rmse:.3f} PICP={picp:.3f} MPIW={mpiw:.2f} ECE={ece:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt")
    torch.save({'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': C.HIDDEN_DIM, 'dropout': 0.2,
                'log_sigma_min': C.LOG_SIGMA_MIN, 'log_sigma_max': C.LOG_SIGMA_MAX, 'seed': seed, 'dataset': DS,
                'fit_units': fit_units, 'val_units': val_units, 'best_val_rmse_cycles': best_val_rmse,
                'train_epochs': EPOCHS, 'selection_protocol': 'canonical_split fit/val, leakfree (FD003, 2026-09-18)'},
               ckpt_path)
    return {'seed': seed, 'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece,
            'sigma_mean': float(sigma_np.mean()), 'best_val_rmse': best_val_rmse, 'elapsed_train_s': elapsed}


# ---------------------------------------------------------------------
# STEP 3: train CP-norm (5 seeds, leakfree, with calib)
# ---------------------------------------------------------------------
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
    k = int(np.ceil((n + 1) * (1 - alpha))); k = min(k, n)
    return float(np.quantile(scores, k / n, method='higher'))


def picp_mpiw(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper))), float(np.mean(upper - lower))


def train_cp_one_seed(seed, device, use_amp):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    split = CANON[DS][str(seed)]
    fit_units, val_units, calib_units = split['fit_units'], split['val_units'], split['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(DS, fit_units)
    input_dim = len(feat_cols)

    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_val_raw, y_val_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(val_units)], feat_cols, val_units)
    y_val_cycles = np.clip(y_val_raw, 0, C.MAX_RUL)
    X_val_t = torch.tensor(X_val_raw, dtype=torch.float32).to(device)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'),
        num_workers=4 if device.type == 'cuda' else 0)

    model = C.HeteroscedasticLSTM(input_dim, C.HIDDEN_DIM, dropout=0.2,
                                   log_sigma_min=C.LOG_SIGMA_MIN, log_sigma_max=C.LOG_SIGMA_MAX).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sc = GradScaler(device='cuda', enabled=use_amp)

    def nll_loss(mu, log_sigma, y):
        sigma = torch.exp(log_sigma)
        return (log_sigma + 0.5 * ((y - mu) / sigma) ** 2).mean()

    best_val_rmse, best_state = float('inf'), None
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                mu, ls = model(bx); loss = nll_loss(mu, ls, by)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        model.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                mv, _ = model(X_val_t)
            mv = np.clip(mv.float().cpu().numpy().flatten() * 125, 0, 125)
            r = np.sqrt(mean_squared_error(y_val_cycles, mv))
        if r < best_val_rmse:
            best_val_rmse = r; best_state = copy.deepcopy(model.state_dict())
    elapsed = time.time() - t0
    model.load_state_dict(best_state); model.eval()

    ckpt_path = os.path.join(CKPT_DIR, f"{DS}_SplitCP_seed{seed}.pt")
    torch.save({'state_dict': best_state, 'input_dim': input_dim, 'hidden_dim': C.HIDDEN_DIM, 'dropout': 0.2,
                'log_sigma_min': C.LOG_SIGMA_MIN, 'log_sigma_max': C.LOG_SIGMA_MAX, 'seed': seed, 'dataset': DS,
                'fit_units': fit_units, 'val_units': val_units, 'calib_units': calib_units,
                'best_val_rmse_cycles': best_val_rmse, 'train_epochs': EPOCHS,
                'selection_protocol': 'canonical_split fit/val/calib, leakfree (FD003, 2026-09-18)'}, ckpt_path)

    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu_c, ls_c = model(X_calib_t)
    mu_calib = np.clip(mu_c.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_calib = torch.exp(ls_c).float().cpu().numpy().flatten() * 125.0
    with torch.no_grad():
        mu_t, ls_t = model(X_test_t)
    mu_test = np.clip(mu_t.float().cpu().numpy().flatten() * 125.0, 0, 125)
    sigma_test = torch.exp(ls_t).float().cpu().numpy().flatten() * 125.0
    rmse, score = C.rmse_score(y_test, mu_test)

    n_calib = len(y_calib)
    s_norm = np.abs(y_calib - mu_calib) / np.clip(sigma_calib, 1e-6, None)
    q_norm = conformal_quantile(s_norm, ALPHA_MAIN, n_calib)
    picp_norm, mpiw_norm = picp_mpiw(y_test, mu_test - q_norm * sigma_test, mu_test + q_norm * sigma_test)

    q_norm_by_level = {}
    emp = []
    for p in CONF_LEVELS:
        q = conformal_quantile(s_norm, 1 - p, n_calib)
        q_norm_by_level[f"{p:.2f}"] = q
        lo, hi = mu_test - q * sigma_test, mu_test + q * sigma_test
        emp.append(np.mean((y_test >= lo) & (y_test <= hi)))
    ece_norm = float(np.mean(np.abs(np.array(emp) - CONF_LEVELS)))
    compliance = abs(picp_norm - 0.90)

    print(f"   [CP-norm] seed={seed} n_calib_units={len(calib_units)} n_calib_windows={n_calib} "
          f"train={elapsed:.1f}s best_val_rmse={best_val_rmse:.3f} "
          f"PICP={picp_norm:.3f} MPIW={mpiw_norm:.2f} ECE={ece_norm:.4f} |dev|={compliance:.3f}"
          f"{'  <-- 超出±0.03容差' if compliance > 0.03 else ''}")

    return {'seed': seed, 'rmse': rmse, 'score': score,
            'cp_norm': {'picp': picp_norm, 'mpiw': mpiw_norm, 'ece': ece_norm, 'q': q_norm,
                        'q_by_level': q_norm_by_level, 'deviation_from_090': compliance}}


# ---------------------------------------------------------------------
# STEP 4: three-arm sweep (NLL + CP-norm only) -> clamp_frac + frozen-sigma
# ---------------------------------------------------------------------
def infer_nll(model, X_t, batch=8192):
    mus, lss = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            m, s = model(X_t[i:i + batch])
            mus.append(m.cpu().numpy().flatten()); lss.append(s.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0, np.concatenate(lss)


def picp_mpiw_z(y_true, mu, sigma, z=Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi))), float(np.mean(hi - lo))


def run_sweep_arm(arm, test_df_raw, true_ruls, feat_cols, device, levels, is_pct,
                   global_std, scalers_by_seed, cp_json, full_scale=None):
    out = {'feat_oob': {}, 'NLL': {}, 'CP_norm': {}, 'NLL_clamp_frac': {}, 'CP_norm_clamp_frac': {},
           'NLL_frozen_sigma': {}, 'CP_norm_frozen_sigma': {}}

    nll_models, cp_models = {}, {}
    clean_sigma_nll, clean_sigma_cp = {}, {}
    for seed in C.SEEDS:
        nll_models[seed] = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt"), device)
        cp_models[seed] = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_SplitCP_seed{seed}.pt"), device)
        scaler = scalers_by_seed[seed]
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_clean_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        _, ls_n = infer_nll(nll_models[seed], X_clean_t)
        clean_sigma_nll[seed] = np.exp(ls_n) * 125.0
        _, ls_c = infer_nll(cp_models[seed], X_clean_t)
        clean_sigma_cp[seed] = np.exp(ls_c) * 125.0

    for level in levels:
        level_key = ('inf' if (not is_pct and np.isinf(level)) else str(level))
        nll_cells, cp_cells, nll_frozen_cells, cp_frozen_cells = ({} for _ in range(4))
        nll_ls_pool, cp_ls_pool, feat_oob_trials = [], [], []
        trial_y = None
        for i in range(5):
            nll_cells[i] = []; cp_cells[i] = []; nll_frozen_cells[i] = []; cp_frozen_cells[i] = []

        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(DS, arm, level_key, t)))
            if is_pct:
                raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, level, rng, full_scale)
            else:
                raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, 'global', global_std=global_std)
            for i, seed in enumerate(C.SEEDS):
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                if seed == C.SEEDS[0]:
                    feat_oob_trials.append(float(np.mean((scaled_feat < -1.0) | (scaled_feat > 1.0))))

                mu_n, ls_n = infer_nll(nll_models[seed], X_t)
                sigma_n = np.exp(ls_n) * 125.0
                picp_n, mpiw_n = picp_mpiw_z(y_test, mu_n, sigma_n)
                rmse_n, _ = C.rmse_score(y_test, mu_n)
                nll_cells[i].append({'rmse': rmse_n, 'picp': picp_n, 'mpiw': mpiw_n, 'sigma_mean': float(sigma_n.mean())})
                nll_ls_pool.append(ls_n)
                sigma_n_frozen = clean_sigma_nll[seed]
                picp_nf, mpiw_nf = picp_mpiw_z(y_test, mu_n, sigma_n_frozen)
                nll_frozen_cells[i].append({'picp': picp_nf, 'mpiw': mpiw_nf})

                q_norm = cp_json[str(seed)]['cp_norm']['q']
                mu_c, ls_c = infer_nll(cp_models[seed], X_t)
                sigma_c = np.exp(ls_c) * 125.0
                picp_c, mpiw_c = picp_mpiw_z(y_test, mu_c, sigma_c, z=q_norm)
                rmse_c, _ = C.rmse_score(y_test, mu_c)
                cp_cells[i].append({'rmse': rmse_c, 'picp': picp_c, 'mpiw': mpiw_c, 'sigma_mean': float(sigma_c.mean())})
                cp_ls_pool.append(ls_c)
                sigma_c_frozen = clean_sigma_cp[seed]
                picp_cf, mpiw_cf = picp_mpiw_z(y_test, mu_c, sigma_c_frozen, z=q_norm)
                cp_frozen_cells[i].append({'picp': picp_cf, 'mpiw': mpiw_cf})

        def agg(cells_dict, key):
            arr = np.array([[cells_dict[i][t][key] for t in range(N_TRIALS)] for i in range(5)])
            return {'grand_mean': float(arr.mean()), 'grand_std': float(arr.std(ddof=1))}

        out['feat_oob'][level_key] = float(np.mean(feat_oob_trials))
        out['NLL'][level_key] = {'picp': agg(nll_cells, 'picp'), 'mpiw': agg(nll_cells, 'mpiw'),
                                  'rmse': agg(nll_cells, 'rmse'), 'sigma_mean': agg(nll_cells, 'sigma_mean')}
        out['CP_norm'][level_key] = {'picp': agg(cp_cells, 'picp'), 'mpiw': agg(cp_cells, 'mpiw'),
                                      'rmse': agg(cp_cells, 'rmse'), 'sigma_mean': agg(cp_cells, 'sigma_mean')}
        out['NLL_frozen_sigma'][level_key] = {'picp': agg(nll_frozen_cells, 'picp'), 'mpiw': agg(nll_frozen_cells, 'mpiw')}
        out['CP_norm_frozen_sigma'][level_key] = {'picp': agg(cp_frozen_cells, 'picp'), 'mpiw': agg(cp_frozen_cells, 'mpiw')}
        out['NLL_clamp_frac'][level_key] = float(np.mean(np.concatenate(nll_ls_pool) <= (C.LOG_SIGMA_MIN + CLAMP_EPS)))
        out['CP_norm_clamp_frac'][level_key] = float(np.mean(np.concatenate(cp_ls_pool) <= (C.LOG_SIGMA_MIN + CLAMP_EPS)))

        print(f"    {level_key:>6}: feat_oob={out['feat_oob'][level_key]:.4f}  "
              f"NLL_PICP={out['NLL'][level_key]['picp']['grand_mean']:.3f}  "
              f"NLL_frozen_PICP={out['NLL_frozen_sigma'][level_key]['picp']['grand_mean']:.3f}  "
              f"NLL_clamp={out['NLL_clamp_frac'][level_key]:.3f}  "
              f"CP_PICP={out['CP_norm'][level_key]['picp']['grand_mean']:.3f}  "
              f"CP_clamp={out['CP_norm_clamp_frac'][level_key]:.3f}")

    for seed in C.SEEDS:
        del nll_models[seed], cp_models[seed]
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def crossover_feat_oob(points, threshold=0.80):
    pts = sorted(points, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i]; fo1, p1 = pts[i + 1]
        if p0 >= threshold and p1 < threshold:
            if p1 == p0:
                return fo0
            frac = (threshold - p0) / (p1 - p0)
            return fo0 + frac * (fo1 - fo0)
    if pts and pts[0][1] < threshold:
        return pts[0][0]
    return None


def interp(sorted_pts, x):
    xs = [p[0] for p in sorted_pts]; ys = [p[1] for p in sorted_pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = 0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    use_amp = device.type == 'cuda'

    print("\n=== STEP 1: extend canonical_splits.json ===")
    extend_canonical_splits()

    print("\n=== STEP 2: train NLL (5 seeds) ===")
    nll_results = [train_nll_one_seed(seed, device, use_amp) for seed in C.SEEDS]
    with open(os.path.join(RESULTS_DIR, 'stepFD003_nll_leakfree_results.json'), 'w') as fp:
        json.dump(nll_results, fp, indent=2, default=float)

    print("\n=== STEP 3: train CP-norm (5 seeds) ===")
    cp_results = [train_cp_one_seed(seed, device, use_amp) for seed in C.SEEDS]
    with open(os.path.join(RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json'), 'w') as fp:
        json.dump(cp_results, fp, indent=2, default=float)
    cp_json = {str(r['seed']): r for r in cp_results}

    print("\n=== STEP 4: three-arm sweep (NLL+CP-norm) + clamp_frac + frozen-sigma ===")
    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(DS)
    scalers_by_seed = {}
    for seed in C.SEEDS:
        fit_units = CANON[DS][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(DS, fit_units)
        scalers_by_seed[seed] = scaler
    global_std = np.std(train_df_raw[feat_cols].values, axis=0)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)

    print("  --- main+extended SNR grid (single condition) ---")
    main_result = run_sweep_arm('B_pooled', test_df_raw, true_ruls, feat_cols, device, SNR_LEVELS_ALL,
                                 is_pct=False, global_std=global_std, scalers_by_seed=scalers_by_seed, cp_json=cp_json)
    print("  --- Arm C ---")
    armC_result = run_sweep_arm('C_fixedpct', test_df_raw, true_ruls, feat_cols, device, PCT_LEVELS,
                                 is_pct=True, global_std=None, scalers_by_seed=scalers_by_seed, cp_json=cp_json,
                                 full_scale=full_scale)

    with open(os.path.join(LEAKFREE_DIR, 'FD003_sweep_leakfree.json'), 'w') as fp:
        json.dump({'main': main_result, 'armC': armC_result}, fp, indent=2, default=float)

    print("\n=== Decomposition ===")
    snr_keys_all = ['inf', '40', '30', '25', '20', '15', '10', '5', '0', '-5', '-10']
    pct_keys = [str(p) for p in PCT_LEVELS]
    decomposition = {}
    clamp_at_clean = {}
    for method, real_key, frozen_key, clamp_key in [('NLL', 'NLL', 'NLL_frozen_sigma', 'NLL_clamp_frac'),
                                                      ('CP_norm', 'CP_norm', 'CP_norm_frozen_sigma', 'CP_norm_clamp_frac')]:
        pts_real, pts_frozen = [], []
        for k in snr_keys_all:
            fo = main_result['feat_oob'][k]
            pts_real.append((fo, main_result[real_key][k]['picp']['grand_mean']))
            pts_frozen.append((fo, main_result[frozen_key][k]['picp']['grand_mean']))
        for k in pct_keys:
            fo = armC_result['feat_oob'][k]
            pts_real.append((fo, armC_result[real_key][k]['picp']['grand_mean']))
            pts_frozen.append((fo, armC_result[frozen_key][k]['picp']['grand_mean']))

        pts_real_sorted = sorted(pts_real, key=lambda p: p[0])
        pts_frozen_sorted = sorted(pts_frozen, key=lambda p: p[0])
        co = crossover_feat_oob(pts_real, threshold=0.80)
        picp_clean_real = min(pts_real_sorted, key=lambda p: p[0])[1]
        picp_clean_frozen = min(pts_frozen_sorted, key=lambda p: p[0])[1]
        clamp_at_clean[method] = main_result[clamp_key]['inf']

        if co is None:
            decomposition[method] = {'note': 'never crosses PICP=0.80 in measured range'}
            continue
        picp_frozen_at_co = interp(pts_frozen_sorted, co)
        delta_total = picp_clean_real - 0.80
        delta_mu_only = picp_clean_frozen - picp_frozen_at_co
        delta_sigma = delta_total - delta_mu_only
        sigma_fraction = delta_sigma / delta_total if delta_total != 0 else None
        decomposition[method] = {
            'crossover_feat_oob': co, 'picp_clean_real': picp_clean_real,
            'picp_clean_frozen_sigma_counterfactual': picp_clean_frozen,
            'picp_frozen_sigma_counterfactual_at_crossover': picp_frozen_at_co,
            'delta_picp_total': delta_total, 'delta_picp_mu_only_frozen_sigma_counterfactual': delta_mu_only,
            'delta_picp_sigma_contribution': delta_sigma, 'sigma_contribution_fraction': sigma_fraction,
            'clamp_frac_at_clean': clamp_at_clean[method],
        }
        print(f"{DS} {method}: clean_clamp_frac={clamp_at_clean[method]:.3f}  crossover_feat_oob={co:.4f}  "
              f"ΔPICP_total={delta_total:.4f}  ΔPICP_mu_only={delta_mu_only:.4f}  "
              f"ΔPICP_sigma={delta_sigma:.4f}  sigma_contribution_fraction={sigma_fraction:.3f}")

    with open(os.path.join(LEAKFREE_DIR, 'FD003_frozen_sigma_decomposition_leakfree.json'), 'w') as fp:
        json.dump(decomposition, fp, indent=2, default=float)
    print(f"\nSaved -> leakfree/FD003_sweep_leakfree.json, leakfree/FD003_frozen_sigma_decomposition_leakfree.json")
    print("FD003 mechanism supplement complete.")
