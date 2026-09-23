"""
R9 单元测试：Part2（增益显式掩码）+ Part3（feat_oob 统一实现）。
不是 pytest 套件，直接跑（python3 test_masks.py），assert 失败即报错退出。
"""
import os
import json

import numpy as np

import common as C
import noise_injection as V4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')


def test_gain_mask():
    ds = 'FD002'
    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    sensor_mask = V4.sensor_mask_for(feat_cols)
    assert sensor_mask[:3].sum() == 0, "FD002 feat_cols[:3] must be the 3 setting columns"
    assert sensor_mask[3:].all(), "FD002 feat_cols[3:] must all be sensor columns"

    rng = np.random.RandomState(0)
    raw_vals = test_df_raw[feat_cols].values.astype(np.float64)
    noisy = V4.inject_gain_fixed_pct_raw(test_df_raw, feat_cols, 5.0, rng, sensor_mask=sensor_mask)

    # setting columns (first 3) must be byte-for-byte unchanged
    assert np.array_equal(noisy[:, :3], raw_vals[:, :3]), \
        "setting columns changed under gain injection despite sensor_mask"

    # sensor columns must differ by exactly a (1 +/- 0.05) factor
    ratio = noisy[:, 3:] / raw_vals[:, 3:]
    close_to_plus = np.isclose(ratio, 1.05, atol=1e-9)
    close_to_minus = np.isclose(ratio, 0.95, atol=1e-9)
    assert np.all(close_to_plus | close_to_minus), \
        "sensor columns not scaled by exactly (1+/-5%)"
    assert close_to_plus.any() and close_to_minus.any(), \
        "expected a mix of +5%/-5% signs across the 15 sensor channels"

    # sanity: with sensor_mask=None (legacy behaviour), setting columns DO change
    rng2 = np.random.RandomState(0)
    noisy_unmasked = V4.inject_gain_fixed_pct_raw(test_df_raw, feat_cols, 5.0, rng2, sensor_mask=None)
    assert not np.array_equal(noisy_unmasked[:, :3], raw_vals[:, :3]), \
        "sanity check failed: unmasked call should still perturb setting columns"

    print("test_gain_mask: PASS")


def test_feat_oob_fd002_drift5pct():
    """Cross-check against the R9-fixed drift-5% sensors-only f_oob on FD002
    (mirrors the computation in diagnostic_perturbation_target_a1.py's
    variant_scale_and_mask('sensors',...) + feat_oob call)."""
    ds = 'FD002'
    train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
    sensor_mask = V4.sensor_mask_for(feat_cols)
    full_scale_sensors_only = V4.sensor_only_scale(feat_cols, full_scale)

    X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
    X_raw_drift = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, full_scale_sensors_only)

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    foob_all_seeds = []
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        X_scaled = V4.scale_raw_windows(X_raw_drift, scaler)
        foob_all_seeds.append(V4.feat_oob(X_scaled, sensor_mask))
    foob_sensor_denom = float(np.mean(foob_all_seeds))

    foob_all_seeds_18 = []
    for seed in C.SEEDS:
        fit_units = canon[ds][str(seed)]['fit_units']
        _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
        X_scaled = V4.scale_raw_windows(X_raw_drift, scaler)
        foob_all_seeds_18.append(float(np.mean((X_scaled < -1.0) | (X_scaled > 1.0))))
    foob_18col_denom = float(np.mean(foob_all_seeds_18))

    print(f"  sensor-denominator f_oob = {foob_sensor_denom*100:.2f}%  "
          f"(expected ~11.43%, matches A1)")
    print(f"  18-column-denominator f_oob = {foob_18col_denom*100:.2f}%  "
          f"(expected ~9.53%, the old/wrong value)")

    assert abs(foob_sensor_denom - 0.1143) < 0.002, \
        f"sensor-denominator f_oob = {foob_sensor_denom:.4f}, expected ~0.1143"
    assert abs(foob_18col_denom - 0.0953) < 0.002, \
        f"18-column f_oob = {foob_18col_denom:.4f}, expected ~0.0953"
    assert foob_sensor_denom > foob_18col_denom, \
        "sensor-only denominator must give a LARGER ratio than the 18-column one"

    print("test_feat_oob_fd002_drift5pct: PASS")


if __name__ == '__main__':
    test_gain_mask()
    test_feat_oob_fd002_drift5pct()
    print("\nALL R9 MASK TESTS PASS")
