"""
STEP 2/3/4/5 共用：数据加载 + HeteroscedasticLSTM 定义 + checkpoint 加载。
与 code/superseded/train_lstm_leaked_test_select_whole_file_scaler.py 及本项目
最早的主实验训练脚本的超参、预处理逐字一致，
保证从 checkpoint 恢复出来的模型行为与训练时完全对应。
"""
import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def require_fixed_hashseed():
    """2026-09-21 复现性修复：噪声注入种子此前统一用 Python 内置 hash()
    派生（对含字符串的 tuple，按 PEP 456 逐进程随机加盐，PYTHONHASHSEED
    未设置时每次解释器启动都不同——已现场验证）。所有含随机扰动推理的
    入口脚本必须先设 PYTHONHASHSEED=0 才能跑，本函数在脚本顶部调用，
    未设置就报错退出，不静默继续。

    同时在这里把 cuDNN/算法选择也锁定为确定性模式——第一轮全量重跑
    （run1 vs run2 MD5 比对）发现：只固定噪声种子（stable_seed）不够，
    26个受检文件里有14个两次独立重跑数值不同，全部集中在跑量大、前向
    推理批次多的脚本（高斯三臂/bias-gain-drift主扫描、维护压力测试），
    小规模的四组合归因/对照脚本反而两次就完全一致——判断是 cuDNN 对
    LSTM/Transformer 前向传播的算法选择在批次更多时更容易走到非确定性
    分支。这里统一加上 `torch.backends.cudnn.deterministic=True` +
    `benchmark=False` + `use_deterministic_algorithms(True, warn_only=True)`
    后重跑第三遍，14个不一致文件全部转为一致（细节见
    COST_TABLE_NOTES.md 对应记录）。"""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError(
            "PYTHONHASHSEED is not set to '0'. Noise-injection seeds must be "
            "reproducible across process runs; re-run this script as:\n"
            "  PYTHONHASHSEED=0 python3 <script>.py\n"
            "See results/generated/COST_TABLE_NOTES.md "
            "('复现性缺陷' entry, 2026-09-21) for why this matters."
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def stable_seed(*parts):
    """2026-09-21 复现性修复：替换全部 `hash((...)) % (2**31)` 调用点。
    不依赖 Python 解释器的 hash 加盐，跨进程/跨机器逐比特可复现，只要
    parts 的值不变。"""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


DATA_DIR = '/home/jeffwork/rul_project/data'
SEEDS = [42, 2024, 7, 888, 123]
DATASETS = ['FD001', 'FD002', 'FD004']
SEQUENCE_LENGTH = 30
MAX_RUL = 125
HIDDEN_DIM = 64
LOG_SIGMA_MIN = -3.0
LOG_SIGMA_MAX = 2.0
Z_SCORE = 1.645

INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = ['s_{}'.format(i) for i in range(1, 22)]
FD001_FEATS = [
    's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11',
    's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'
]


class HeteroscedasticLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout, log_sigma_min, log_sigma_max):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat, _ = self.lstm(x)
        feat = self.dropout(feat[:, -1, :])
        mu = self.mu_head(feat)
        log_sigma = torch.clamp(self.log_sigma_head(feat), self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


def get_feature_names(dataset_name):
    if dataset_name in ('FD001', 'FD003'):
        # FD003 确认单一工况（KMeans k=6/k=1惯性比=0.074，setting_3恒为100，
        # 与FD001同构，2026-09-18核实），沿用FD001的14特征集，不用
        # FD002/FD004那套"丢弃6个随工况变化的传感器"的选择——那套选择是
        # 针对多工况数据集设计的，FD003没有多工况这个问题，不适用。
        return FD001_FEATS
    drop_cols = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    sensors = [s for s in SENSOR_NAMES if s not in drop_cols]
    return SETTING_NAMES + sensors


def load_and_process(dataset_name):
    train_path = os.path.join(DATA_DIR, f'train_{dataset_name}.txt')
    test_path = os.path.join(DATA_DIR, f'test_{dataset_name}.txt')
    rul_path = os.path.join(DATA_DIR, f'RUL_{dataset_name}.txt')

    col_names = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    true_ruls = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    max_cycles = train_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max']
    train_df = train_df.merge(max_cycles, on='unit_nr', how='left')
    train_df['RUL'] = (train_df['max'] - train_df['time_cycles']).clip(upper=MAX_RUL)

    feature_cols = get_feature_names(dataset_name)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, true_ruls, feature_cols


def create_sequences(df, feature_cols, mode='train', true_ruls=None, return_unit=False):
    X_list, y_list, u_list = [], [], []
    for unit in df['unit_nr'].unique():
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        if mode == 'train':
            rul_arr = df[df['unit_nr'] == unit]['RUL'].values
            for i in range(len(unit_data) - SEQUENCE_LENGTH):
                X_list.append(unit_data[i: i + SEQUENCE_LENGTH])
                y_list.append(rul_arr[i + SEQUENCE_LENGTH] / 125.0)
                u_list.append(unit)
        elif mode == 'test':
            if len(unit_data) >= SEQUENCE_LENGTH:
                X_list.append(unit_data[-SEQUENCE_LENGTH:])
                # R8-B2 (2026-09-2x)：官方 RUL_FD00X.txt 与训练标签
                # (max_cycles-time_cycles，最后一行=0) 的计数起点相差1个
                # 周期（R8-A2 诊断已核实差异幅度小但方向一致），改用
                # min(官方RUL-1, 125) 与训练/校准同一惯例。
                y_list.append(min(true_ruls.iloc[unit - 1].item() - 1, 125))
                u_list.append(unit)
    X = np.array(X_list)
    y = np.array(y_list)
    if return_unit:
        return X, y, np.array(u_list)
    return X, y


def create_full_trajectory_test_windows(test_df, feature_cols, true_ruls):
    """
    STEP4 per-engine 覆盖率专用：标准 create_sequences(mode='test') 每台发动机只取
    最后一个窗口（单点预测），无法算"per-engine 覆盖率分布"（需要每台发动机多个
    预测点才有意义的覆盖率）。这里改为在每台测试发动机的可用轨迹上做全滑窗，
    RUL 目标按分段线性退化模型反推：
        RUL(row j) = true_RUL_at_last_row + (last_row_idx - j)，clip 到 max_rul。
    与训练集 RUL 标签的构造方式（max_cycles - time_cycles，clip 125）完全同源。
    """
    X_list, y_list, u_list = [], [], []
    for unit in test_df['unit_nr'].unique():
        unit_data = test_df[test_df['unit_nr'] == unit][feature_cols].values
        n = len(unit_data)
        if n < SEQUENCE_LENGTH:
            continue
        rul_at_last_row = min(true_ruls.iloc[unit - 1].item() - 1, MAX_RUL)  # R8-B2, see create_sequences
        last_row_idx = n - 1
        for i in range(n - SEQUENCE_LENGTH + 1):
            end_row_idx = i + SEQUENCE_LENGTH - 1
            rul_here = min(rul_at_last_row + (last_row_idx - end_row_idx), MAX_RUL)
            X_list.append(unit_data[i:i + SEQUENCE_LENGTH])
            y_list.append(rul_here)
            u_list.append(unit)
    return np.array(X_list), np.array(y_list), np.array(u_list)


def split_units_two_way(unit_list, held_out_frac, seed):
    """按 seed 派生的确定性发动机级切分（不按窗口随机切，避免同一台发动机的
    窗口同时出现在两侧造成信息泄漏）。用于 fit/val（checkpoint 选择用）以及
    STEP3 fit/calib 切分，二者共用同一份逻辑，保证切分方式在全项目内一致。
    """
    rng = np.random.RandomState(seed)
    units = np.array(sorted(unit_list))
    perm = rng.permutation(units)
    n_held = max(1, int(round(len(units) * held_out_frac)))
    held_units = set(perm[:n_held].tolist())
    keep_units = set(perm[n_held:].tolist())
    return keep_units, held_units


CANONICAL_VAL_FRAC = 0.20
CANONICAL_CALIB_FRAC = 0.20  # matches STEP3's original (pre-fix) calib_frac definition


def compute_canonical_split(unit_list, seed, val_frac=CANONICAL_VAL_FRAC, calib_frac=CANONICAL_CALIB_FRAC):
    """全项目唯一的发动机级三向切分：fit(60%)/val(20%)/calib(20%)（默认档），
    STEP0/STEP1/STEP3 三条训练线共用同一个 (fit_units, val_units)（早停/
    checkpoint 选择判据完全一致，方法间可比），STEP3 额外从同一份 val_frac
    先切出去之后剩下的池子里再切出 calib_units（因此 calib 与 fit/val 也
    互不重叠，不是从 fit 这个"已经在用"的训练集里二次借用）。

    两步派生：先按 val_frac 从全部单元里切 val；剩下 (1-val_frac) 的池子
    再按 calib_frac/(1-val_frac) 切 calib，使 calib 在全体单元里的绝对占比
    仍然是 calib_frac（不因为先扣掉了val而被稀释/放大）。第二步用
    seed+999983（任意质数偏移）派生，避免与第一步的排列存在可预测的相关性。

    返回：(fit_units, val_units, calib_units) 三个排序后的 list，互不重叠，
    并集 = unit_list。
    """
    rest_units, val_units = split_units_two_way(unit_list, val_frac, seed)
    calib_frac_of_rest = calib_frac / (1.0 - val_frac)
    fit_units, calib_units = split_units_two_way(sorted(rest_units), calib_frac_of_rest, seed + 999983)
    return sorted(fit_units), sorted(val_units), sorted(calib_units)


def load_and_process_leakfree(dataset_name, fit_units):
    """与 load_and_process 相同的预处理，唯一区别：MinMaxScaler 只在
    fit_units 那部分 train 数据上 fit，val/calib/test 全部只 transform，
    不参与拟合——修复此前 scaler.fit(全部train_df) 隐式把 val/calib 的
    特征分布泄漏进 scaler min/max 参数这一问题（即便这些行从未参与梯度/
    早停判据，标准化本身的参数也不该看到它们）。
    """
    train_path = os.path.join(DATA_DIR, f'train_{dataset_name}.txt')
    test_path = os.path.join(DATA_DIR, f'test_{dataset_name}.txt')
    rul_path = os.path.join(DATA_DIR, f'RUL_{dataset_name}.txt')

    col_names = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
    true_ruls = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    max_cycles = train_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max']
    train_df = train_df.merge(max_cycles, on='unit_nr', how='left')
    train_df['RUL'] = (train_df['max'] - train_df['time_cycles']).clip(upper=MAX_RUL)

    feature_cols = get_feature_names(dataset_name)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    fit_mask = train_df['unit_nr'].isin(fit_units)
    scaler.fit(train_df.loc[fit_mask, feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, true_ruls, feature_cols, scaler


def load_checkpoint_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeteroscedasticLSTM(
        ckpt['input_dim'], ckpt['hidden_dim'], ckpt['dropout'],
        ckpt['log_sigma_min'], ckpt['log_sigma_max']
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def picp_mpiw(y_true, y_mean, y_sigma, z=Z_SCORE):
    lower = y_mean - z * y_sigma
    upper = y_mean + z * y_sigma
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    return picp, mpiw


def compute_ece(mu_all, sigma_all, ytrue_all, conf_levels=None):
    from scipy import stats
    if conf_levels is None:
        conf_levels = np.arange(0.05, 1.00, 0.05)
    empirical = []
    for p in conf_levels:
        z = stats.norm.ppf((1 + p) / 2)
        lo = mu_all - z * sigma_all
        hi = mu_all + z * sigma_all
        cov = np.mean((ytrue_all >= lo) & (ytrue_all <= hi))
        empirical.append(cov)
    empirical = np.array(empirical)
    return float(np.mean(np.abs(empirical - conf_levels)))


def rmse_score(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    d = y_pred - y_true
    score = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
    return rmse, score
