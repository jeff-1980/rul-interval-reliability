import hashlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results')

FILES = """
degradation/lstm_noise_main_arm.json
degradation/lstm_noise_armc.json
degradation/lstm_fd003_noise_main_arm.json
degradation/lstm_fd003_noise_armc.json
degradation/dose_response_feat_oob.json
degradation/lstm_frozen_sigma_decomposition.json
degradation/lstm_frozen_sigma_decomposition_relative.json
degradation/lstm_clamp_frac_main_arm.json
degradation/lstm_clamp_frac_armc.json
degradation/lstm_relative_half_life.json
degradation/transformer_noise_main_arm.json
degradation/transformer_noise_armc.json
degradation/transformer_half_life.json
degradation/transformer_clamp_frac.json
degradation/transformer_frozen_sigma_decomposition.json
degradation/degradation_sweep_bias_gain_drift.json
degradation/degradation_half_life_bias_gain_drift.json
attribution/frozen_output_decomposition_2x2_nearest_grid.json
attribution/attribution_raw_per_seed_nearest_grid.json
attribution/attribution_transformer_exact_interp.json
attribution/attribution_lstm_exact_interp.json
degradation/threshold_crossover_refinement.json
degradation/threshold_crossover_leftside_extension.json
attribution/attribution_bootstrap_by_order_exact_interp.json
degradation/drift_controls.json
decision/maintenance_decision_two_sided.json
decision/maintenance_decision_one_sided.json
degradation/drift_fixed_endpoint_shuffle_control.json
decision/maintenance_decision_rul_corrected_v2.json
decision/maintenance_decision_rul_correction_exact_match_check.json
clean/table2_clean_full.json
clean/lstm_clean_ece_pooled.json
""".split()


def md5_of(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


if __name__ == '__main__':
    match, mismatch, missing = [], [], []
    for rel in FILES:
        snap_path = os.path.join(RESULTS_DIR, '_snapshot', rel)
        cur_path = os.path.join(RESULTS_DIR, rel)
        if not os.path.exists(snap_path) or not os.path.exists(cur_path):
            missing.append(rel)
            continue
        m1, m2 = md5_of(snap_path), md5_of(cur_path)
        (match if m1 == m2 else mismatch).append(rel if m1 == m2 else (rel, m1, m2))

    print(f"MATCH ({len(match)}):")
    for f in match:
        print(f"  OK   {f}")
    print(f"\nMISMATCH ({len(mismatch)}):")
    for f, m1, m2 in mismatch:
        print(f"  DIFF {f}  snapshot={m1[:12]}  current={m2[:12]}")
    if missing:
        print(f"\nMISSING ({len(missing)}): {missing}")
    print(f"\nSummary: {len(match)}/{len(FILES)} files match.")
