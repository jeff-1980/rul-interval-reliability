"""
Shared noise-sensitivity module: leakage-free training-set statistics plus
two injection arms.

Arm A (main protocol, per_condition): stratify by C-MAPSS operating
condition (KMeans k=6, fit on training-set setting values); each stratum's
noise scale is that stratum's own per-channel training-set std; each test
row is assigned to its nearest condition before injection.
Arm B (control, pooled): a single per-channel std over the whole training
set (conditions pooled) used for every row -- separates "std source is
train vs test" from "stratified by condition or not".

FD001 has only one real operating condition (KMeans k=6 vs k=1 inertia
ratio 0.074, far above FD002/4's 0.00004, and setting_3 is always 100), so
arm A is meaningless for it -- computed once, equal to arm B.

The version using test-set statistics (an earlier, methodologically wrong
definition: the noise scale would depend on the very data being evaluated,
which is both leakage and unavailable at deployment) is fully deprecated
and is not an option for any function in this file.
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
    """Same as load_raw_train_test_and_scaler, but the scaler is fit only on
    fit_units (matching the scaler definition used at checkpoint-training
    time). Condition clustering/per-condition std still use the whole
    training fleet -- this is the noise-injection experiment's own
    statistic, not a model-training normalisation parameter, so using the
    full fleet to represent real sensor-noise characteristics is a
    reasonable choice, not a leak.
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
    """Arm C: per-channel training-set full-scale range (max - min), a
    condition-independent constant noise-scale reference."""
    vals = train_df_raw[feature_cols].values.astype(np.float64)
    return vals.max(axis=0) - vals.min(axis=0)


def sensor_only_scale(feature_cols, scale):
    """Zero a per-channel noise/drift/bias/gain scale array (full_scale_range
    / global_std / cond_std[c]) on the operating-condition setting columns,
    so downstream inject_* functions no longer perturb them -- settings are
    a commanded regime, not a sensor reading, and physically should not be
    corrupted by "sensor degradation". FD001/FD003's feature_cols never
    include setting columns, so this is a no-op for them. Does not modify
    any inject_* function itself, nor fit_fullscale_range/fit_condition_model
    (their raw, unfiltered output is still used for the joint 18-column
    perturbation control)."""
    is_setting = np.array([c in C.SETTING_NAMES for c in feature_cols])
    scale = np.array(scale, dtype=np.float64, copy=True)
    scale[..., is_setting] = 0.0
    return scale


def sensor_mask_for(feature_cols):
    """Single source of truth for "is this column a sensor (not a setting)"
    boolean mask, shared by inject_gain_fixed_pct_raw's explicit mask
    parameter and feat_oob()'s denominator, so the two never independently
    re-derive is_setting logic and drift out of sync."""
    return np.array([c not in C.SETTING_NAMES for c in feature_cols])


def feat_oob(X_scaled, sensor_mask):
    """The single project-wide f_oob implementation. X_scaled: (..., n_feat)
    already scaled to [-1,1] (windowed or not, last axis is feature
    columns); sensor_mask: boolean array of length n_feat, True=sensor
    column. The denominator counts only sensor_mask-selected columns --
    setting columns are never perturbed under the sensor-only protocol, so
    including them in the denominator would dilute/distort the ratio."""
    return float(np.mean((X_scaled[..., sensor_mask] < -1.0) | (X_scaled[..., sensor_mask] > 1.0)))


def inject_noise(test_df_raw, feature_cols, scaler, snr_db, rng, scheme,
                  global_std=None, km=None, cond_std=None):
    """scheme: 'global' (train-set pooled std, arm B) or 'per_condition' (train-set per-cluster std, arm A)"""
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
    """Same noise-generation logic as inject_noise, but skips
    scaler.transform and returns raw_vals+noise (unscaled). Used in the
    leakfree setting: the same raw noise instance is scaled independently
    by each of 5 seeds' own scalers -- the "shared noise trial" design must
    not break just because the scaler differs per seed, since the noise
    itself is a physical quantity added in raw units; scaling is only each
    model's own downstream input mapping."""
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
    """Unscaled version of inject_noise_fixed_pct, same reason as above."""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    noise_std = full_scale_range * (pct / 100.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=raw_vals.shape)
    return raw_vals + noise


def inject_bias_fixed_pct_raw(test_df_raw, feature_cols, pct, rng, full_scale_range):
    """Deterministic bias: a constant per-channel offset k*FS (k=pct/100),
    sign randomised per channel per trial (rng derived by the caller per
    trial), avoiding the directional artefact of "all channels biased the
    same way". Uses the same arm-C absolute scale reference as
    inject_noise_fixed_pct_raw (per-channel training-set full-scale range,
    condition-independent)."""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    n_feat = raw_vals.shape[1]
    sign = rng.choice([-1.0, 1.0], size=n_feat)
    bias = full_scale_range * (pct / 100.0) * sign
    return raw_vals + bias[None, :]


def inject_gain_fixed_pct_raw(test_df_raw, feature_cols, pct, rng, full_scale_range=None, sensor_mask=None):
    """Deterministic gain error: each channel multiplied by (1+/-k),
    k=pct/100, sign randomised per channel per trial. Applied directly in
    raw physical units (not mean-centred), so large-DC-offset channels
    (e.g. s9 approx 9050rpm) see a much larger absolute shift than other
    channels at the same k% -- this is the true physical behaviour of a
    gain error, faithfully reflected in feat_oob, not de-biased.
    full_scale_range is kept only to match the other injection functions'
    call signature; gain error itself does not depend on full-scale range
    (the relative error multiplies the reading directly).

    sensor_mask (boolean array of length n_feat, True=sensor column, see
    sensor_mask_for) explicitly controls which columns are actually
    multiplied by (1+/-k) -- masked-out columns keep factor=1 (unchanged).
    sensor_only_scale only zeroes full_scale_range, but gain's factor does
    not depend on full_scale_range at all (see above), so that alone never
    actually made gain injection sensor-only; setting columns kept being
    multiplied by (1+/-k) even after that fix -- a real bug, fixed here, not
    documentation polish. sensor_mask=None keeps the old behaviour (every
    column multiplied), reserved for any future joint 18-column-injection
    control; every call site in this project now passes sensor_mask
    explicitly."""
    raw_vals = test_df_raw[feature_cols].values.astype(np.float64)
    n_feat = raw_vals.shape[1]
    sign = rng.choice([-1.0, 1.0], size=n_feat)
    factor = 1.0 + (pct / 100.0) * sign
    if sensor_mask is not None:
        factor = np.where(np.asarray(sensor_mask), factor, 1.0)
    return raw_vals * factor[None, :]


def extract_raw_windows(test_df_raw, feature_cols, true_ruls, mode='test'):
    """Unscaled version of create_sequences/create_full_trajectory_test_windows,
    for drift injection (drift needs the windowed but unscaled raw readings
    to apply a position-dependent ramp inside the window, then scale per
    seed afterwards -- unlike bias/gain/noise, which inject once over the
    whole raw df and then scale).
    mode='test': last window only per engine (the evaluation convention
    used by the main dose-response/half-life/frozen-sigma analyses);
    mode='full_trajectory': every window per engine (the per-engine
    coverage evaluation convention), RUL label constructed the same way as
    C.create_full_trajectory_test_windows (back-computed under the
    piecewise-linear degradation model)."""
    import common as C
    X_list, y_list, u_list = [], [], []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        if mode == 'test':
            X_list.append(unit_data[-C.SEQUENCE_LENGTH:])
            y_list.append(min(true_ruls.iloc[unit - 1].item() - 1, C.MAX_RUL))  # see common.create_sequences
            u_list.append(unit)
        elif mode == 'full_trajectory':
            rul_at_last_row = min(true_ruls.iloc[unit - 1].item() - 1, C.MAX_RUL)
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
    """Deterministic drift: a linear ramp by position inside each window,
    from 0 to +k*FS (k=pct/100), independently re-zeroed per window
    (simulating "the sensor is drifting within this prediction's
    observation window", not a single drift trend spanning the whole test
    file). No sign randomisation (unlike bias/gain -- drift's own
    "progressive departure" direction is captured by the sign of pct; taken
    literally here as a one-directional ramp).
    X_raw_windows: (n_windows, seq_len, n_feat) raw (unscaled) readings."""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(0.0, 1.0, seq_len)[None, :, None]  # (1, T, 1)
    k = full_scale_range * (pct / 100.0)  # (n_feat,)
    delta = ramp * k[None, None, :]  # (1, T, n_feat) broadcasts over n
    return X_raw_windows + delta


def inject_drift_reverse_fixed_pct_windows(X_raw_windows, pct, full_scale_range):
    """Control (1), reversed ramp: from +k*FS down to 0 (the original goes
    0 up to +k*FS) -- tests whether the largest perturbation needs to sit
    right next to the prediction point."""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(1.0, 0.0, seq_len)[None, :, None]
    k = full_scale_range * (pct / 100.0)
    delta = ramp * k[None, None, :]
    return X_raw_windows + delta


def inject_drift_shuffled_fixed_pct_windows(X_raw_windows, pct, full_scale_range, rng):
    """Control (2), same-magnitude-distribution time-shuffle: the same set
    of ramp magnitudes {0, k/(T-1), ..., k} is kept but randomly reordered
    across time steps (independently per window) -- tests whether it is the
    smooth trend itself, rather than just these values appearing somewhere
    in the window, that drives the effect."""
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
    """Discriminating control: keep the true ramp values at the last k_end
    steps fixed, and only shuffle the order of the values among the first
    (T-k_end) steps (using the magnitude set the real ramp would have had
    there, just reordered in time; the tail is untouched). Tests
    "temporal structure vs. tail weighting": if only the last k steps being
    correct matters and shuffling the rest doesn't, that means recency
    (only the last few steps) drives it; if shuffling still changes PICP
    noticeably, the whole window's temporal structure (not just the tail)
    matters too."""
    n, seq_len, n_feat = X_raw_windows.shape
    base_values = np.linspace(0.0, 1.0, seq_len)  # (T,)
    k = full_scale_range * (pct / 100.0)  # (F,)
    delta = np.zeros_like(X_raw_windows)
    n_free = seq_len - k_end
    for i in range(n):
        perm = rng.permutation(n_free)
        shuffled_first = base_values[:n_free][perm]
        assigned = np.concatenate([shuffled_first, base_values[n_free:]])  # tail k_end steps keep the true ramp value
        delta[i] = assigned[:, None] * k[None, :]
    return X_raw_windows + delta


def inject_drift_singlechannel_fixed_pct_windows(X_raw_windows, pct, full_scale_range, channel_idx):
    """Control (3), single-channel vs. all-channel: the standard 0->k*FS
    ramp is applied only to channel_idx, leaving every other channel clean
    -- tests whether the effect requires perturbation coordinated across
    channels."""
    n, seq_len, n_feat = X_raw_windows.shape
    ramp = np.linspace(0.0, 1.0, seq_len)  # (T,)
    k_single = full_scale_range[channel_idx] * (pct / 100.0)
    delta = np.zeros_like(X_raw_windows)
    delta[:, :, channel_idx] = ramp[None, :] * k_single
    return X_raw_windows + delta


def inject_drift_continuous_trajectory_raw(test_df_raw, feature_cols, pct, full_scale_range):
    """Control (4), drift generated on the continuous trajectory then
    windowed: opposite of the original (where each window independently
    re-zeroes its own 0->k*FS ramp, so the same physical instant gets a
    different perturbation value in different windows). Here a single
    continuous 0->k*FS ramp is generated over each engine's whole test
    trajectory (by absolute row position, i.e. physical time order), then
    the trailing window is cut out the standard way -- the same physical
    instant gets the same perturbation value in every window that could
    contain it. Tests the "drift is read as degradation" hypothesis: if the
    effect mainly comes from the artefact of "a local ramp right before
    each prediction, re-zeroed per window", the continuous version should
    behave differently.
    Returns the last-window-per-engine (n_engines, T, F) raw (unscaled)
    windows, ready to feed to scale_raw_windows."""
    import common as C
    X_list = []
    for unit in test_df_raw['unit_nr'].unique():
        unit_data = test_df_raw[test_df_raw['unit_nr'] == unit][feature_cols].values.astype(np.float64)
        n = len(unit_data)
        if n < C.SEQUENCE_LENGTH:
            continue
        ramp_full = np.linspace(0.0, 1.0, n)  # whole trajectory, physical time order
        k = full_scale_range * (pct / 100.0)
        delta_full = ramp_full[:, None] * k[None, :]
        drifted_full = unit_data + delta_full
        X_list.append(drifted_full[-C.SEQUENCE_LENGTH:])
    return np.array(X_list)


def scale_raw_windows(X_raw_windows, scaler):
    """Scale (n,T,F) raw windows with the given scaler, sharing one 2D
    transform across all windows/time steps (reshape to (n*T,F) and back,
    matching scaler.transform's row-level contract)."""
    n, T, F = X_raw_windows.shape
    flat = X_raw_windows.reshape(n * T, F)
    scaled = scaler.transform(flat)
    return scaled.reshape(n, T, F)


def scale_and_package(test_df_raw, feature_cols, raw_noisy_vals, scaler):
    """Scale inject_noise_raw/inject_noise_fixed_pct_raw's output with the
    given scaler and repackage into a DataFrame for create_sequences."""
    df = test_df_raw.copy()
    scaled = scaler.transform(raw_noisy_vals)
    df[feature_cols] = scaled
    return df, scaled


def inject_noise_fixed_pct(test_df_raw, feature_cols, scaler, pct, rng, full_scale_range):
    """Arm C: fixed at some percentage of the per-channel training-set
    full-scale range, condition-independent (the same noise scale used for
    every row, physically corresponding to "sensor noise does not vary
    with operating condition"), matching a GUM Type-B absolute uncertainty
    statement (e.g. "sensor accuracy +/-1% FS")."""
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
