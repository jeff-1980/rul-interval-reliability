"""
v4 噪声敏感性共用模块：训练集统计量（杜绝泄漏）+ 两臂注入方案。

臂 A（主协议，per_condition）：按 C-MAPSS 工况（KMeans k=6，训练集设定值上拟合）
分层，每层用该层在训练集上的逐通道 std 决定噪声尺度，测试集每行按自身工况
就近分配后注入。
臂 B（对照，pooled）：训练集全体（跨工况混合）算一个逐通道 std，用同一个
std 给所有行注入——用于分离"std 来源从测试集换成训练集"与"是否按工况分层"
这两个此前混在一起的变量。

FD001 只有 1 个真实工况（KMeans k=6 惯性比 0.074，远高于 FD002/4 的 0.00004，
且 setting_3 全程=100），臂 A 对它没有意义，只算一次（等价于臂 B），两个
臂标签指向同一份结果。

不使用测试集统计量的版本（v2/v3/5b/5c 之前用的定义）已被判定为方法学错误
（噪声尺度依赖被评估数据本身，构成信息泄漏，且部署时不可得），完全弃用，
不作为本文件任何函数的选项。
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import common as C

N_CLUSTERS = 6
SNR_LEVELS = [np.inf, 40, 30, 25, 20, 15, 10, 5, 0]
SNR_LEVELS_EXTENDED = [-5, -10]  # appended below 0dB to pin down the PICP=0.80 crossover
N_TRIALS = 5
CLAMP_EPS = 1e-3
PCT_LEVELS = [0.1, 0.5, 1, 2, 5]  # Arm C: fixed % of per-channel training full-scale range


def load_raw_train_test_and_scaler(dataset_name):
    col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
    train_path = os.path.join(C.DATA_DIR, f'train_{dataset_name}.txt')
    test_path = os.path.join(C.DATA_DIR, f'test_{dataset_name}.txt')
    rul_path = os.path.join(C.DATA_DIR, f'RUL_{dataset_name}.txt')
    train_df_raw = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    test_df_raw = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    true_ruls = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    feature_cols = C.get_feature_names(dataset_name)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df_raw[feature_cols])
    return train_df_raw, test_df_raw, true_ruls, feature_cols, scaler


def load_raw_train_test_and_scaler_leakfree(dataset_name, fit_units):
    """与 load_raw_train_test_and_scaler 相同，scaler 只在 fit_units 上拟合
    （与 checkpoint 训练时用的 scaler 定义一致，2026-09-15 leakfree协议）。
    工况聚类/逐工况std仍用全部train发动机算（这是噪声注入实验自身的统计量，
    不是模型训练用的标准化参数，不受本轮"训练侧scaler泄漏"修复范围约束，
    继续用全部train fleet代表真实传感器噪声特性是合理选择，非leakage）。
    """
    col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
    train_path = os.path.join(C.DATA_DIR, f'train_{dataset_name}.txt')
    test_path = os.path.join(C.DATA_DIR, f'test_{dataset_name}.txt')
    rul_path = os.path.join(C.DATA_DIR, f'RUL_{dataset_name}.txt')
    train_df_raw = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    test_df_raw = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    true_ruls = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    feature_cols = C.get_feature_names(dataset_name)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    fit_mask = train_df_raw['unit_nr'].isin(fit_units)
    scaler.fit(train_df_raw.loc[fit_mask, feature_cols])
    return train_df_raw, test_df_raw, true_ruls, feature_cols, scaler


def fit_condition_model(train_df_raw, feature_cols, n_clusters=N_CLUSTERS):
    settings = train_df_raw[C.SETTING_NAMES].values
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(settings)
    labels = km.labels_
    cond_std = {}
    for c in range(n_clusters):
        mask = labels == c
        cond_std[c] = np.std(train_df_raw.loc[mask, feature_cols].values, axis=0)
    global_std = np.std(train_df_raw[feature_cols].values, axis=0)
    return km, cond_std, global_std


def fit_fullscale_range(train_df_raw, feature_cols):
    """臂C：训练集逐通道 全幅量程 = max - min，工况无关的常数噪声尺度基准"""
    vals = train_df_raw[feature_cols].values.astype(np.float64)
    return vals.max(axis=0) - vals.min(axis=0)


def sensor_only_scale(feature_cols, scale):
    """R8-B1：把一个逐通道噪声/漂移/偏置/增益尺度数组（full_scale_range /
    global_std / cond_std[c]）在"工况设定"列上置零，使下游 inject_* 函数
    不再对这些列注入任何扰动——工况设定是指令量（commanded operating
    regime），不是传感器读数，物理上不该被"传感器退化"污染。R8-A1 诊断
    已确认：把FD002/FD004的扰动限制到只剩15个传感器列，L=20提前触发率
    与当前(18列联合)口径同量级（比值0.79-1.00，远不到"减半"），不是本
    结论的成因，但仍是应该修的口径问题。FD001/FD003 的 feature_cols 本来
    就不含工况设定列，这里是no-op，不受影响。不改动任何 inject_* 函数
    本身，也不改动 fit_fullscale_range/fit_condition_model 本身（那两个
    函数的原始、未过滤输出仍用于 R8-B4 的"18维联合扰动"对照）。"""
    is_setting = np.array([c in C.SETTING_NAMES for c in feature_cols])
    scale = np.array(scale, dtype=np.float64, copy=True)
    scale[..., is_setting] = 0.0
    return scale


def inject_noise(test_df_raw, feature_cols, scaler, snr_db, rng, scheme,
                  global_std=None, km=None, cond_std=None):
    """scheme: 'global' (train-set pooled std, 臂B) or 'per_condition' (train-set per-cluster std, 臂A)"""
    df = test_df_raw.copy()
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    if np.isinf(snr_db):
        scaled = scaler.transform(raw_vals)
        df[feature_cols] = scaled
        return df, scaled

    if scheme == 'global':
        noise_std = global_std * (10 ** (-snr_db / 20.0))
        noise = rng.normal(loc=0.0, scale=noise_std, size=raw_vals.shape)
    elif scheme == 'per_condition':
        labels = km.predict(test_df_raw[C.SETTING_NAMES].values)
        noise = np.zeros_like(raw_vals)
        for c in range(km.n_clusters):
            mask = labels == c
            if mask.sum() == 0:
                continue
            noise_std_c = cond_std[c] * (10 ** (-snr_db / 20.0))
            noise[mask] = rng.normal(loc=0.0, scale=noise_std_c, size=(mask.sum(), raw_vals.shape[1]))
    else:
        raise ValueError(scheme)

    scaled = scaler.transform(raw_vals + noise)
    df[feature_cols] = scaled
    return df, scaled


def inject_noise_raw(test_df_raw, feature_cols, snr_db, rng, scheme, global_std=None, km=None, cond_std=None):
    """与 inject_noise 相同的噪声生成逻辑，但不做 scaler.transform，返回
    raw_vals+noise（未标准化）。用于 leakfree 场景：同一份原始噪声要被
    5个seed各自的scaler分别标准化（"同一次噪声实例"这个 shared-trial 设计
    不能因为scaler逐seed不同就被破坏——噪声本身是物理量，加在原始单位上，
    标准化只是后续每个模型自己的输入映射）。"""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    if np.isinf(snr_db):
        return raw_vals
    if scheme == 'global':
        noise_std = global_std * (10 ** (-snr_db / 20.0))
        noise = rng.normal(loc=0.0, scale=noise_std, size=raw_vals.shape)
    elif scheme == 'per_condition':
        labels = km.predict(test_df_raw[C.SETTING_NAMES].values)
        noise = np.zeros_like(raw_vals)
        for c in range(km.n_clusters):
            mask = labels == c
            if mask.sum() == 0:
                continue
            noise_std_c = cond_std[c] * (10 ** (-snr_db / 20.0))
            noise[mask] = rng.normal(loc=0.0, scale=noise_std_c, size=(mask.sum(), raw_vals.shape[1]))
    else:
        raise ValueError(scheme)
    return raw_vals + noise


def inject_noise_fixed_pct_raw(test_df_raw, feature_cols, pct, rng, full_scale_range):
    """inject_noise_fixed_pct 的未标准化版本，同上原因。"""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    noise_std = full_scale_range * (pct / 100.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=raw_vals.shape)
    return raw_vals + noise


def inject_bias_fixed_pct_raw(test_df_raw, feature_cols, pct, rng, full_scale_range):
    """T2 Part B：确定性偏置。每通道常数偏移 k·FS（k=pct/100），符号按
    trial 逐通道随机（rng 由调用方按 trial 派生），避免"全通道同向偏移"这个
    方向性伪影。与 inject_noise_fixed_pct_raw 同一臂C绝对尺度基准（训练集
    逐通道全幅量程，工况无关）。"""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    n_feat = raw_vals.shape[1]
    sign = rng.choice([-1.0, 1.0], size=n_feat)
    bias = full_scale_range * (pct / 100.0) * sign
    return raw_vals + bias[None, :]


def inject_gain_fixed_pct_raw(test_df_raw, feature_cols, pct, rng, full_scale_range=None):
    """T2 Part B：确定性增益误差。每通道乘 (1±k)，k=pct/100，符号按 trial
    逐通道随机。直接作用于原始物理单位（未去均值），因此大直流偏置通道
    （如 s9≈9050rpm）在同样的 k% 下会有远大于其他通道的绝对位移——这是
    增益误差的真实物理行为，如实计入 feat_oob，不做去偏置处理。
    full_scale_range 参数保留仅为与其它注入函数同一调用签名，增益误差本身
    不依赖全幅量程（相对误差直接乘在读数上）。"""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    n_feat = raw_vals.shape[1]
    sign = rng.choice([-1.0, 1.0], size=n_feat)
    factor = 1.0 + (pct / 100.0) * sign
    return raw_vals * factor[None, :]


def extract_raw_windows(test_df_raw, feature_cols, true_ruls, mode='test'):
    """create_sequences/create_full_trajectory_test_windows 的未标准化版本，
    供 drift 注入使用（drift 需要在窗口内部按位置施加斜坡，必须先拿到窗口化
    但未标准化的原始读数，注入后再逐 seed 标准化，不能像 bias/gain/noise 那样
    在整张 raw df 上一次性注入再统一标准化）。
    mode='test'：每台发动机只取最后一个窗口（对应主 dose-response/半衰/冻结
    σ̂分析用的评估口径）；mode='full_trajectory'：每台发动机全部滑窗（对应
    per-engine 覆盖率评估口径），RUL 标签构造与 C.create_full_trajectory_test_windows
    完全同源（分段线性退化模型反推）。"""
    import common as C
    X_list, y_list, u_list = [], [], []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        if mode == 'test':
            X_list.append(unit_data[-C.SEQUENCE_LENGTH:])
            y_list.append(min(true_ruls.iloc[unit - 1].item() - 1, C.MAX_RUL))  # R8-B2, see common.create_sequences
            u_list.append(unit)
        elif mode == 'full_trajectory':
            rul_at_last_row = min(true_ruls.iloc[unit - 1].item() - 1, C.MAX_RUL)  # R8-B2
            last_row_idx = n - 1
            for i in range(n - C.SEQUENCE_LENGTH + 1):
                end_row_idx = i + C.SEQUENCE_LENGTH - 1
                rul_here = min(rul_at_last_row + (last_row_idx - end_row_idx), C.MAX_RUL)
                X_list.append(unit_data[i:i + C.SEQUENCE_LENGTH])
                y_list.append(rul_here)
                u_list.append(unit)
        else:
            raise ValueError(mode)
    return np.array(X_list), np.array(y_list), np.array(u_list)


def inject_drift_fixed_pct_windows(X_raw_windows, pct, full_scale_range):
    """T2 Part B：确定性漂移。每个窗口内部按时间位置线性斜坡，从 0 到
    +k·FS（k=pct/100），每个窗口独立重新起算（模拟"这次预测前的观测窗内
    传感器正在漂移"，而不是跨越整个测试文件的单一漂移趋势）。不做符号
    随机化（与 bias/gain 不同——drift 本身的"渐进偏离"方向由 pct 的正负
    刻画，用户任务描述里只对 bias/gain 提了逐通道随机符号，未提 drift，
    这里按字面理解为单向斜坡；如需改为符号随机，容易补）。
    X_raw_windows: (n_windows, seq_len, n_feat) 原始（未标准化）读数。"""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(0.0, 1.0, seq_len)[None, :, None]  # (1, T, 1)
    k = full_scale_range * (pct / 100.0)  # (n_feat,)
    delta = ramp * k[None, None, :]  # (1, T, n_feat) broadcasts over n
    return X_raw_windows + delta


def inject_drift_reverse_fixed_pct_windows(X_raw_windows, pct, full_scale_range):
    """R2-5 对照(1)反向斜坡：从 +k·FS 斜坡降到 0（原版是 0 升到 +k·FS），
    检验"最大扰动是否必须紧邻预测点"这一点是否重要。"""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(1.0, 0.0, seq_len)[None, :, None]
    k = full_scale_range * (pct / 100.0)
    delta = ramp * k[None, None, :]
    return X_raw_windows + delta


def inject_drift_shuffled_fixed_pct_windows(X_raw_windows, pct, full_scale_range, rng):
    """R2-5 对照(2)同幅值分布时间乱序：斜坡用到的同一组幅值
    {0, k/(T-1), ..., k} 保留，但按时间步随机打乱顺序（每个窗口独立打乱），
    检验是不是"平滑趋势"本身而不是"这些数值出现在窗口里"驱动了效应。"""
    n, seq_len, n_feat = X_raw_windows.shape
    base_values = np.linspace(0.0, 1.0, seq_len)  # (T,)
    k = full_scale_range * (pct / 100.0)  # (F,)
    delta = np.zeros_like(X_raw_windows)
    for i in range(n):
        perm = rng.permutation(seq_len)
        shuffled = base_values[perm]  # (T,)
        delta[i] = shuffled[:, None] * k[None, :]
    return X_raw_windows + delta


def inject_drift_fixed_endpoint_shuffle_windows(X_raw_windows, pct, full_scale_range, k_end, rng):
    """R3-D.1 判别对照：固定末端 k_end 步的真实斜坡值不变，只打乱前
    (T-k_end) 步之间的顺序（用真实斜坡在那些位置本该有的幅值集合，
    只重排它们的时间顺序，不改变末端）。检验"时序结构 vs 末端权重"：
    如果只要末端k步正确、前面乱序也不影响，说明是recency（只看最后
    几步）驱动；如果乱序仍然明显改变PICP，说明整个窗口的时序结构
    （不只是末端）也重要。"""
    n, seq_len, n_feat = X_raw_windows.shape
    base_values = np.linspace(0.0, 1.0, seq_len)  # (T,)
    k = full_scale_range * (pct / 100.0)  # (F,)
    delta = np.zeros_like(X_raw_windows)
    n_free = seq_len - k_end
    for i in range(n):
        perm = rng.permutation(n_free)
        shuffled_first = base_values[:n_free][perm]
        assigned = np.concatenate([shuffled_first, base_values[n_free:]])  # 末k_end步保持真实斜坡值
        delta[i] = assigned[:, None] * k[None, :]
    return X_raw_windows + delta


def inject_drift_singlechannel_fixed_pct_windows(X_raw_windows, pct, full_scale_range, channel_idx):
    """R2-5 对照(3)单通道 vs 全通道：标准 0->k·FS 斜坡只施加在 channel_idx
    这一个通道上，其余通道保持 clean，检验效应是否需要跨通道协同扰动。"""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(0.0, 1.0, seq_len)  # (T,)
    k_single = full_scale_range[channel_idx] * (pct / 100.0)
    delta = np.zeros_like(X_raw_windows)
    delta[:, :, channel_idx] = ramp[None, :] * k_single
    return X_raw_windows + delta


def inject_drift_continuous_trajectory_raw(test_df_raw, feature_cols, pct, full_scale_range):
    """R2-5 对照(4)连续轨迹上生成漂移后再切窗：与原版（每个窗口独立重新起算
    0->k·FS，同一物理时刻在不同窗口里的扰动值不同）相反，这里先在每台
    发动机的整条测试轨迹（按绝对行位置，即物理时间顺序）上生成一条连续的
    0->k·FS 斜坡，再按标准方式切出末端窗口——同一物理时刻在所有可能包含它
    的窗口里扰动值一致。用于检验"漂移被读成退化"假说：如果效应主要来自
    "每次预测前有一段局部斜坡"这个人为的按窗口重新起算的假象，连续版本
    应该表现不同。
    返回按 test 模式（每台发动机最后一个窗口）切好的 (n_engines, T, F) 原始
    （未标准化）窗口，可直接喂给 scale_raw_windows。"""
    import common as C
    X_list = []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        ramp_full = np.linspace(0.0, 1.0, n)  # 整条轨迹，物理时间顺序
        k = full_scale_range * (pct / 100.0)
        delta_full = ramp_full[:, None] * k[None, :]
        drifted_full = unit_data + delta_full
        X_list.append(drifted_full[-C.SEQUENCE_LENGTH:])
    return np.array(X_list)


def scale_raw_windows(X_raw_windows, scaler):
    """把 (n,T,F) 原始窗口用给定 scaler 标准化，逐窗口/逐时间步共用同一份
    2D transform（reshape 到 (n*T,F) 再 reshape 回去，与 scaler.transform
    的行级契约一致）。"""
    n, T, F = X_raw_windows.shape
    flat = X_raw_windows.reshape(n * T, F)
    scaled = scaler.transform(flat)
    return scaled.reshape(n, T, F)


def scale_and_package(test_df_raw, feature_cols, raw_noisy_vals, scaler):
    """把 inject_noise_raw/inject_noise_fixed_pct_raw 的输出用给定 scaler
    标准化并打包回 DataFrame，供 create_sequences 使用。"""
    df = test_df_raw.copy()
    scaled = scaler.transform(raw_noisy_vals)
    df[feature_cols] = scaled
    return df, scaled


def inject_noise_fixed_pct(test_df_raw, feature_cols, scaler, pct, rng, full_scale_range):
    """臂C：固定为训练集逐通道全幅量程的某百分比，工况无关（同一份噪声尺度
    用于所有行，物理上对应"传感器噪声不随工况浮动"），对应 GUM Type B 的
    绝对不确定度表达（例如"传感器精度 ±1% FS"）。"""
    df = test_df_raw.copy()
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    noise_std = full_scale_range * (pct / 100.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=raw_vals.shape)
    scaled = scaler.transform(raw_vals + noise)
    df[feature_cols] = scaled
    return df, scaled


def infer_nll(model, X_t, batch=8192):
    mus, log_sigmas = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mu, ls = model(X_t[i:i + batch])
            mus.append(mu.cpu().numpy().flatten())
            log_sigmas.append(ls.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0, np.concatenate(log_sigmas)


def infer_mc_dropout(mc_model, X_t, T, aleatory_var, batch=4096):
    all_samples = []
    mc_model.train()
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            xb = X_t[i:i + batch]
            samples = [mc_model(xb).cpu().numpy().flatten() * 125.0 for _ in range(T)]
            all_samples.append(np.stack(samples))
    samples = np.concatenate(all_samples, axis=1)
    mu = np.clip(samples.mean(0), 0, 125)
    sigma = np.sqrt(aleatory_var + samples.var(0))
    return mu, sigma


def coverage_half_life(picp_by_snr, snr_levels=SNR_LEVELS, threshold=0.80):
    for snr in snr_levels:
        key = 'inf' if np.isinf(snr) else str(snr)
        if picp_by_snr[key] < threshold:
            return key
    return 'not_reached_at_0dB'
