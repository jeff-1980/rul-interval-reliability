"""
T2 通用扰动扫描引擎：与 run_sweep_noise_lstm_fd003.py 的 run_arm() 同一
协议（5 seed 模型 × N_TRIALS 共享噪声流、NLL/MC-Dropout/Ensemble/CP-norm/
MSE-fixed 五机制 + NLL/CP-norm 冻结σ̂反事实、feat_oob 记录），泛化出两个维度：
  1. backbone ∈ {LSTM, Transformer} —— 模型加载走 transformer_common 的 checkpoint
     调度（Transformer 的 CP-norm 复用 NLL checkpoint，见该文件顶部说明）。
  2. 扰动类型 —— 分两类调用方式：
     (a) run_df_perturb_sweep：扰动在整张 raw df 上一次性注入（噪声/bias/
         gain 都属于这类，注入函数签名统一为
         inject_fn(test_df_raw, feat_cols, level, rng, full_scale) -> raw_noisy）。
     (b) run_drift_sweep：drift 需要窗口内部按位置施加斜坡，必须先把 raw df
         窗口化（未标准化）再注入，见 noise_injection.extract_raw_windows /
         inject_drift_fixed_pct_windows。

LSTM 侧复用已有 checkpoint（不重训）；Transformer 侧复用 T2 训练产出的
checkpoint。CP-norm 的 q / q_by_level、MC-Dropout 的 aleatory_var 从对应
backbone 已有的结果文件读取（适配层 load_mc_cp_for_seed）。
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2

CONF_LEVELS = np.arange(0.05, 1.00, 0.05)
Z_SCORE = 1.645
N_TRIALS = V4.N_TRIALS

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)

_MC_CP_CACHE = {}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_mc_cp_for_seed(ds, backbone):
    """返回 (mc_by_seed, cp_by_seed) 两个 dict[seed_str] -> record，统一
    LSTM(FD001/2/4 用 mcdropout_fixed_leakfree.json + split_cp_leakfree.json
    的 dataset-keyed 结构；FD003 用 stepFD003_*_leakfree_results.json 的
    flat-list 结构) 与 Transformer(leakfree_t2/t2_transformer_*_results.json，
    dataset-keyed) 三种来源格式的差异。"""
    key = (ds, backbone)
    if key in _MC_CP_CACHE:
        return _MC_CP_CACHE[key]

    if backbone == 'Transformer':
        mc_all = _load_json(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json'))
        cp_all = _load_json(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json'))
        mc_list, cp_list = mc_all[ds], cp_all[ds]
    else:  # LSTM
        if ds == 'FD003':
            mc_list = _load_json(os.path.join(T2.RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json'))
            cp_list = _load_json(os.path.join(T2.RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json'))
        else:
            mc_all = _load_json(os.path.join(T2.RESULTS_DIR, 'mcdropout_fixed_leakfree.json'))
            cp_all = _load_json(os.path.join(T2.RESULTS_DIR, 'split_cp_leakfree.json'))
            mc_list, cp_list = mc_all[ds], cp_all[ds]

    mc_by_seed = {str(r['seed']): r for r in mc_list}
    cp_by_seed = {str(r['seed']): r for r in cp_list}
    _MC_CP_CACHE[key] = (mc_by_seed, cp_by_seed)
    return mc_by_seed, cp_by_seed


_CALIB_ALEATORY_CACHE = {}


def _sequences_for_units(df, feature_cols, unit_set):
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def calib_aleatory_var(ds, backbone, seed, device, mc_model=None):
    """2026-09-21 公平校准修复：此前全部退化/噪声扫描里 MSE-fixed/MC-Dropout
    的 aleatory_var 来自 load_mc_cp_for_seed() 读的旧结果文件——那是训练集
    残差方差，跟 Table II 的"公平校准"版本（校准集残差方差，
    maintenance_decision_two_sided.calib_sigma_fixed / fair_calibration_main_table
    同一定义）不是一回事。这里复刻同一个计算，改成在 calib_units 上现算，
    贯穿所有退化扫描，不再读那个训练残差版文件。按 (ds,backbone,seed) 缓存，
    避免同一扫描里对不同 level 重复计算。"""
    key = (ds, backbone, seed)
    if key in _CALIB_ALEATORY_CACHE:
        return _CALIB_ALEATORY_CACHE[key]
    fit_units = CANON[ds][str(seed)]['fit_units']
    calib_units = CANON[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_calib, y_calib_raw = _sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)],
                                                 feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    owns_model = mc_model is None
    if owns_model:
        mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    mc_model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X_calib_t.shape[0], 8192):
            mus.append(mc_model(X_calib_t[i:i + 8192]).cpu().numpy().flatten())
    yhat_calib = np.clip(np.concatenate(mus) * 125.0, 0, 125)
    aleatory_var = float(np.var(y_calib - yhat_calib, ddof=1))
    if owns_model:
        del mc_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    _CALIB_ALEATORY_CACHE[key] = aleatory_var
    return aleatory_var


def load_models_for_seed(ds, backbone, seed, device):
    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
    mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    if backbone == 'LSTM':
        cp_ckpt = os.path.join(T2.PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_SplitCP_seed{seed}.pt")
        cp_model = C.load_checkpoint_model(cp_ckpt, device)
    else:
        cp_model = nll_model  # Transformer: CP 复用 NLL checkpoint，见文件顶部说明
    return nll_model, mc_model, cp_model


def picp_mpiw(y_true, mu, sigma, z=Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi))), float(np.mean(hi - lo))


def metrics_basic(y_true, mu, sigma):
    rmse, score = C.rmse_score(y_true, mu)
    picp, mpiw = picp_mpiw(y_true, mu, sigma)
    ece = C.compute_ece(mu, sigma, y_true, CONF_LEVELS)
    return {'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece, 'sigma_mean': float(sigma.mean())}


def grand_and_marginal_stats(cells, n_trials=N_TRIALS):
    keys = ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']
    out = {}
    arr = {k: np.array([[cells[i][t][k] for t in range(n_trials)] for i in range(5)]) for k in keys}
    for k in keys:
        a = arr[k]
        model_marg = a.mean(axis=1); trial_marg = a.mean(axis=0)
        out[k] = {'grand_mean': float(a.mean()), 'grand_std': float(a.std(ddof=1)),
                   'model_marginal_mean': float(model_marg.mean()), 'model_marginal_std': float(model_marg.std(ddof=1)),
                   'trial_marginal_mean': float(trial_marg.mean()), 'trial_marginal_std': float(trial_marg.std(ddof=1))}
    return out


def infer_nll(model, X_t, batch=8192):
    mus, ls = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            m, s = model(X_t[i:i + batch])
            mus.append(m.cpu().numpy().flatten()); ls.append(s.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0, np.concatenate(ls)


def infer_mc_dropout(mc_model, X_t, T, aleatory_var, batch=4096, seed=None):
    """seed: 2026-09-21 复现性修复——T=50 的 dropout 采样本身是活的随机数
    （mc_model.train()），此前从未播种，导致跨进程不可复现（run1 vs run2
    的 MD5 比对发现只有这个机制的数字不一致）。传入确定性 seed（用
    C.stable_seed 派生）后在采样前 torch.manual_seed，使其也可复现——
    代价是这是一次"新的"确定性采样，不等同于旧结果里那次未记录种子的
    采样，旧结果里的 MC_Dropout_fixed 具体数字因此无法逐比特复现，只有
    此后的重跑之间能互相复现。"""
    if seed is not None:
        torch.manual_seed(seed)
    mc_model.train()
    all_samples = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            xb = X_t[i:i + batch]
            samples = [mc_model(xb).cpu().numpy().flatten() * 125.0 for _ in range(T)]
            all_samples.append(np.stack(samples))
    samples = np.concatenate(all_samples, axis=1)
    mu = np.clip(samples.mean(0), 0, 125)
    sigma = np.sqrt(aleatory_var + samples.var(0))
    return mu, sigma


def infer_mse_fixed(mc_model, X_t, sigma_fixed, batch=8192):
    mc_model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mus.append(mc_model(X_t[i:i + batch]).cpu().numpy().flatten())
    mu = np.clip(np.concatenate(mus) * 125.0, 0, 125)
    sigma = np.full_like(mu, sigma_fixed)
    return mu, sigma


def _empty_out():
    return {'NLL': {}, 'MC_Dropout_fixed': {}, 'Deep_Ensemble': {}, 'CP_norm': {}, 'MSE_fixed': {}, 'feat_oob': {},
            'NLL_frozen_sigma': {}, 'CP_norm_frozen_sigma': {}}


def _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                     device, aleatory_var_calib_by_seed):
    nll_cells = [[None] * N_TRIALS for _ in range(5)]
    mc_cells = [[None] * N_TRIALS for _ in range(5)]
    cp_cells = [[None] * N_TRIALS for _ in range(5)]
    mse_cells = [[None] * N_TRIALS for _ in range(5)]
    nll_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
    cp_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
    nll_mu_grid = [[None] * N_TRIALS for _ in range(5)]
    nll_sigma_grid = [[None] * N_TRIALS for _ in range(5)]

    _, cp_by_seed = load_mc_cp_for_seed(ds, backbone)

    for i, seed in enumerate(C.SEEDS):
        nll_model, mc_model, cp_model = models_by_seed[seed]
        # 2026-09-21 公平校准修复：aleatory_var 改用校准集残差方差
        # （与 Table II 同一口径），不再读训练残差版结果文件。
        aleatory_var = aleatory_var_calib_by_seed[seed]
        q_norm = cp_by_seed[str(seed)]['cp_norm']['q']
        q_norm_by_level = cp_by_seed[str(seed)]['cp_norm']['q_by_level']

        for t in range(N_TRIALS):
            X_t = trial_X[t][seed]

            mu_n, ls_n = infer_nll(nll_model, X_t)
            sigma_n = np.exp(ls_n) * 125.0
            nll_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n)
            nll_mu_grid[i][t] = mu_n; nll_sigma_grid[i][t] = sigma_n
            sigma_n_frozen = clean_sigma_nll[seed]
            nll_frozen_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n_frozen)

            mc_seed = C.stable_seed(ds, backbone, level_key, t, seed, 'mc_dropout')
            mu_m, sigma_m = infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
            mc_cells[i][t] = metrics_basic(trial_y, mu_m, sigma_m)

            mu_mse, sigma_mse = infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
            mse_cells[i][t] = metrics_basic(trial_y, mu_mse, sigma_mse)

            mu_c, ls_c = infer_nll(cp_model, X_t)
            sigma_c = np.exp(ls_c) * 125.0
            picp_c, mpiw_c = picp_mpiw(trial_y, mu_c, sigma_c, z=q_norm)
            rmse_c, score_c = C.rmse_score(trial_y, mu_c)
            emp = []
            for p in CONF_LEVELS:
                q = q_norm_by_level[f"{p:.2f}"]
                lo, hi = mu_c - q * sigma_c, mu_c + q * sigma_c
                emp.append(np.mean((trial_y >= lo) & (trial_y <= hi)))
            ece_c = float(np.mean(np.abs(np.array(emp) - CONF_LEVELS)))
            cp_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_c, 'mpiw': mpiw_c,
                               'ece': ece_c, 'sigma_mean': float(sigma_c.mean())}
            sigma_c_frozen = clean_sigma_cp[seed]
            picp_cf, mpiw_cf = picp_mpiw(trial_y, mu_c, sigma_c_frozen, z=q_norm)
            cp_frozen_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_cf, 'mpiw': mpiw_cf,
                                      'ece': float('nan'), 'sigma_mean': float(sigma_c_frozen.mean())}

    out['NLL'][level_key] = grand_and_marginal_stats(nll_cells)
    out['MC_Dropout_fixed'][level_key] = grand_and_marginal_stats(mc_cells)
    out['CP_norm'][level_key] = grand_and_marginal_stats(cp_cells)
    out['MSE_fixed'][level_key] = grand_and_marginal_stats(mse_cells)
    out['NLL_frozen_sigma'][level_key] = grand_and_marginal_stats(nll_frozen_cells)
    out['CP_norm_frozen_sigma'][level_key] = grand_and_marginal_stats(cp_frozen_cells)

    ens_trial_metrics = []
    for t in range(N_TRIALS):
        mu_mem = np.stack([nll_mu_grid[i][t] for i in range(5)])
        sigma_mem = np.stack([nll_sigma_grid[i][t] for i in range(5)])
        mu_ens = mu_mem.mean(0)
        sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
        sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
        mu_ens = np.clip(mu_ens, 0, 125)
        ens_trial_metrics.append(metrics_basic(trial_y, mu_ens, sigma_ens))
    ens_arr = {k: np.array([m[k] for m in ens_trial_metrics]) for k in ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']}
    out['Deep_Ensemble'][level_key] = {k: {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'n_trials': N_TRIALS}
                                        for k, v in ens_arr.items()}

    print(f"    {level_key:>6}: feat_oob={out['feat_oob'][level_key]:.4f}  "
          f"NLL_PICP={out['NLL'][level_key]['picp']['grand_mean']:.3f}  "
          f"MC_PICP={out['MC_Dropout_fixed'][level_key]['picp']['grand_mean']:.3f}  "
          f"Ens_PICP={out['Deep_Ensemble'][level_key]['picp']['mean']:.3f}  "
          f"CP_PICP={out['CP_norm'][level_key]['picp']['grand_mean']:.3f}  "
          f"MSE_PICP={out['MSE_fixed'][level_key]['picp']['grand_mean']:.3f}")


def _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed, models_by_seed):
    clean_sigma_nll, clean_sigma_cp = {}, {}
    for seed in C.SEEDS:
        nll_model, mc_model, cp_model = models_by_seed[seed]
        scaler = scalers_by_seed[seed]
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_clean_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        _, ls_nll = infer_nll(nll_model, X_clean_t)
        clean_sigma_nll[seed] = np.exp(ls_nll) * 125.0
        _, ls_cp = infer_nll(cp_model, X_clean_t)
        clean_sigma_cp[seed] = np.exp(ls_cp) * 125.0
    return clean_sigma_nll, clean_sigma_cp


def run_df_perturb_sweep(ds, backbone, perturb_name, inject_fn, levels, is_pct, device,
                          scalers_by_seed, full_scale=None, global_std=None):
    """levels: 数值列表；inject_fn(test_df_raw, feat_cols, level, rng, full_scale) -> raw_noisy
    （bias/gain/armC-gaussian 三者统一走这条路径，签名一致，只是 inject_fn 不同）。
    level_key 规则：is_pct=True 时用 str(level)（如 armC/bias/gain 的 0.1/0.5/1/2/5），
    is_pct=False 时 inf 记为 'inf'（用于噪声 SNR 档，本函数目前只服务 pct 类扰动，
    SNR 类沿用旧 run_arm，不在本引擎重复）。"""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    for level in levels:
        level_key = str(level)
        trial_X, trial_y, trial_feat_oob = {}, None, []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(ds, backbone, perturb_name, level_key, t)))
            raw_noisy = inject_fn(test_df_raw, feat_cols, level, rng, full_scale)
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                # 2026-09-21 f_oob 口径统一：改在实际送入模型的末端窗口 X_test 上算，
                # 不再用 scale_and_package 返回的整段轨迹 scaled_feat（drift 分支已经
                # 是窗口化的 X_scaled，这里补齐到同一口径，与 PICP 的 5x5 汇总对齐）。
                trial_feat_oob.append(float(np.mean((X_test < -1.0) | (X_test > 1.0))))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def run_snr_sweep(ds, backbone, scheme, levels, device, scalers_by_seed, global_std=None, km=None, cond_std=None):
    """主 SNR 臂（高斯噪声，per_condition 或 global 方案），与 LSTM 侧
    run_sweep_noise_lstm.py / run_sweep_noise_lstm_fd003.py
    的主臂逐字同一协议——2026-09-19 补做：Part A 最初只跑了臂C（5档，
    feat_oob范围5.6%-16%），但 LSTM 侧的半衰点主要落在主SNR臂覆盖的
    <2% feat_oob 区间内，臂C单独测不到那个区间，导致"相对半衰机制排序"
    对比不是同一 feat_oob 范围内的比较——如实发现后补上主臂，使
    Transformer 与 LSTM 的半衰点计算基于同一套（主臂+臂C）pooled 数据，
    口径完全对齐。scheme='per_condition' 用于 FD002/FD004（多工况），
    scheme='global' 用于 FD001/FD003（单一工况，等价于旧协议里
    A_percondition退化为B_pooled的情形）。"""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    for level in levels:
        level_key = 'inf' if np.isinf(level) else str(level)
        trial_X, trial_y, trial_feat_oob = {}, None, []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'mainarm', scheme, level_key, t)))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, scheme,
                                             global_std=global_std, km=km, cond_std=cond_std)
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                # 2026-09-21 f_oob 口径统一：见 run_df_perturb_sweep 同日同条注释。
                trial_feat_oob.append(float(np.mean((X_test < -1.0) | (X_test > 1.0))))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def run_drift_sweep(ds, backbone, levels, device, scalers_by_seed, full_scale):
    """drift：窗口级注入（每窗口独立 0->k*FS 斜坡），走
    extract_raw_windows(mode='test') + inject_drift_fixed_pct_windows +
    scale_raw_windows，其余（5模型×5trial共享流、五机制、冻结σ̂、feat_oob）
    与 run_df_perturb_sweep 完全一致。"""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    X_raw_clean, y_test_fixed, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')

    for level in levels:
        level_key = str(level)
        X_raw_drifted = V4.inject_drift_fixed_pct_windows(X_raw_clean, level, full_scale)
        trial_X, trial_y, trial_feat_oob = {}, y_test_fixed, []
        # drift 是确定性斜坡（无随机成分），N_TRIALS 个 trial 在扰动本身上退化为
        # 完全相同的注入；仍保留 N_TRIALS 维度只是为了与其它扰动类型共用同一套
        # grand_and_marginal_stats 聚合代码路径（trial 间方差在此应为 0，属预期）。
        for t in range(N_TRIALS):
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                X_scaled = V4.scale_raw_windows(X_raw_drifted, scaler)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                trial_feat_oob.append(float(np.mean((X_scaled < -1.0) | (X_scaled > 1.0))))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out
