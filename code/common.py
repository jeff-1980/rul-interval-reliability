"""
Shared data loading, HeteroscedasticLSTM definition, and checkpoint loading.
Hyperparameters and preprocessing match the original training scripts
exactly, so a loaded checkpoint's behaviour matches training time.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def require_fixed_hashseed():
    """Noise-injection seeds are derived from Python's built-in hash(), which
    is randomly salted per process (PEP 456) unless PYTHONHASHSEED is fixed.
    Every script performing randomly-perturbed inference must set
    PYTHONHASHSEED=0 before running; this function asserts that and raises
    rather than continuing silently.

    Also locks cuDNN/algorithm selection to deterministic mode: fixing the
    noise seed alone was not sufficient for bit-for-bit reproducibility
    across independent reruns -- cuDNN's algorithm selection for LSTM/
    Transformer forward passes was itself non-deterministic on scripts with
    many forward-pass batches. Setting
    `torch.backends.cudnn.deterministic=True` + `benchmark=False` +
    `use_deterministic_algorithms(True, warn_only=True)` resolved it."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError(
            "PYTHONHASHSEED is not set to '0'. Noise-injection seeds must be "
            "reproducible across process runs; re-run this script as:\n"
            "  PYTHONHASHSEED=0 python3 <script>.py\n"
            "See results/generated/COST_TABLE_NOTES.md "
            "(reproducibility notes) for why this matters."
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def stable_seed(*parts):
    """Deterministic seed derivation independent of Python's salted hash();
    bit-for-bit reproducible across processes/machines given the same
    parts."""
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
        # FD003 is single-condition like FD001 (verified via KMeans
        # inertia ratio), so it reuses FD001's 14-feature set rather than
        # FD002/FD004's condition-varying-sensor-drop selection, which only
        # applies to multi-condition datasets.
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
                # Official RUL_FD00X.txt and the training label
                # (max_cycles-time_cycles, last row=0) count from a
                # one-cycle-different origin; use min(official RUL-1, 125)
                # to match the training/calibration convention.
                y_list.append(min(true_ruls.iloc[unit - 1].item() - 1, 125))
                u_list.append(unit)
    X = np.array(X_list)
    y = np.array(y_list)
    if return_unit:
        return X, y, np.array(u_list)
    return X, y


def create_full_trajectory_test_windows(test_df, feature_cols, true_ruls):
    """
    For per-engine coverage: standard create_sequences(mode='test') gives
    only one window (the last) per engine, which can't define a per-engine
    coverage distribution. This instead slides a window across the full
    available trajectory of each test engine, back-computing the RUL
    target under the piecewise-linear degradation model:
        RUL(row j) = true_RUL_at_last_row + (last_row_idx - j), clipped to
        max_rul -- the same construction as the training RUL label.
    """
    X_list, y_list, u_list = [], [], []
    for unit in test_df['unit_nr'].unique():
        unit_data = test_df[test_df['unit_nr'] == unit][feature_cols].values
        n = len(unit_data)
        if n < SEQUENCE_LENGTH:
            continue
        rul_at_last_row = min(true_ruls.iloc[unit - 1].item() - 1, MAX_RUL)  # see create_sequences
        last_row_idx = n - 1
        for i in range(n - SEQUENCE_LENGTH + 1):
            end_row_idx = i + SEQUENCE_LENGTH - 1
            rul_here = min(rul_at_last_row + (last_row_idx - end_row_idx), MAX_RUL)
            X_list.append(unit_data[i:i + SEQUENCE_LENGTH])
            y_list.append(rul_here)
            u_list.append(unit)
    return np.array(X_list), np.array(y_list), np.array(u_list)


def split_units_two_way(unit_list, held_out_frac, seed):
    """Deterministic engine-level split derived from seed (not a
    window-level random split, which would leak windows of the same engine
    across both sides). Used for both fit/val (checkpoint selection) and
    fit/calib splits, so the splitting logic is consistent project-wide.
    """
    rng = np.random.RandomState(seed)
    units = np.array(sorted(unit_list))
    perm = rng.permutation(units)
    n_held = max(1, int(round(len(units) * held_out_frac)))
    held_units = set(perm[:n_held].tolist())
    keep_units = set(perm[n_held:].tolist())
    return keep_units, held_units


CANONICAL_VAL_FRAC = 0.20
CANONICAL_CALIB_FRAC = 0.20  # matches the original calib_frac definition


def compute_canonical_split(unit_list, seed, val_frac=CANONICAL_VAL_FRAC, calib_frac=CANONICAL_CALIB_FRAC):
    """The single project-wide engine-level three-way split:
    fit(60%)/val(20%)/calib(20%) by default. Every training line shares the
    same (fit_units, val_units) so early-stopping/checkpoint-selection
    criteria are comparable across methods; calib_units is carved from the
    remaining pool after val is removed, so calib never overlaps fit/val.

    Two-step derivation: split off val_frac from all units first; from the
    remaining (1-val_frac) pool, split off calib_frac/(1-val_frac) so
    calib's absolute share of all units stays calib_frac. The second step
    uses a different seed offset to avoid a predictable correlation with
    the first split's permutation.

    Returns (fit_units, val_units, calib_units), pairwise disjoint,
    union = unit_list.
    """
    rest_units, val_units = split_units_two_way(unit_list, val_frac, seed)
    calib_frac_of_rest = calib_frac / (1.0 - val_frac)
    fit_units, calib_units = split_units_two_way(sorted(rest_units), calib_frac_of_rest, seed + 999983)
    return sorted(fit_units), sorted(val_units), sorted(calib_units)


def load_and_process_leakfree(dataset_name, fit_units):
    """Same preprocessing as load_and_process, except the MinMaxScaler is
    fit only on fit_units' training rows; val/calib/test are transform-only.
    This avoids leaking val/calib's feature distribution into the scaler's
    min/max parameters (those rows never enter the scaler's fit even though
    they were never used for gradients or early stopping either).
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
