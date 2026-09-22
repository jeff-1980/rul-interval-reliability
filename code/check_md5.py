import hashlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results')

FILES = """
attribution/attribution_bootstrap_by_order_exact_interp.json
attribution/attribution_lstm_exact_interp.json
attribution/attribution_raw_per_seed_nearest_grid.json
attribution/attribution_transformer_exact_interp.json
attribution/frozen_output_decomposition_2x2_nearest_grid.json
clean/fair_calibration_main_table.json
clean/fair_calibration_sigma_fixed.json
clean/interval_score_wis_clean.json
clean/lstm_clean_ece_pooled.json
clean/lstm_clean_per_seed.json
clean/lstm_cost_table_fd001.csv
clean/lstm_cost_table_fd002.csv
clean/lstm_cost_table_fd003.csv
clean/lstm_cost_table_fd004.csv
clean/lstm_deep_ensemble.json
clean/lstm_fd003_clean_per_seed.json
clean/lstm_fd003_deep_ensemble.json
clean/lstm_fd003_mcdropout_mse_per_seed.json
clean/lstm_fd003_per_engine_coverage.json
clean/lstm_fd003_split_cp_per_seed.json
clean/lstm_mcdropout_per_seed.json
clean/lstm_mse_fixed_training_residual_per_seed.json
clean/lstm_per_engine_coverage.json
clean/lstm_split_cp_per_seed.json
clean/table1_2x2_per_seed.json
clean/table1_2x2_summary.json
clean/table2_clean_full.json
clean/transformer_clean_per_seed.json
clean/transformer_cost_table_fd001.csv
clean/transformer_cost_table_fd002.csv
clean/transformer_cost_table_fd003.csv
clean/transformer_cost_table_fd004.csv
clean/transformer_deep_ensemble.json
clean/transformer_mcdropout_mse_per_seed.json
clean/transformer_per_engine_coverage.json
clean/transformer_split_cp_per_seed.json
decision/maintenance_decision_one_sided.json
decision/maintenance_decision_rul_corrected_v2.json
decision/maintenance_decision_rul_correction_exact_match_check.json
decision/maintenance_decision_two_sided.json
degradation/degradation_dose_response_overlap_summary.json
degradation/degradation_half_life_bias_gain_drift.json
degradation/degradation_sweep_bias_gain_drift.json
degradation/dose_response_feat_oob.json
degradation/drift_controls.json
degradation/drift_fixed_endpoint_shuffle_control.json
degradation/lstm_armc_mu_std.json
degradation/lstm_clamp_frac_armc.json
degradation/lstm_clamp_frac_main_arm.json
degradation/lstm_ensemble_scale_sweep.json
degradation/lstm_fd003_armc_mu_std.json
degradation/lstm_fd003_ensemble_scale_sweep.json
degradation/lstm_fd003_frozen_sigma_decomposition.json
degradation/lstm_fd003_noise_armc.json
degradation/lstm_fd003_noise_main_arm.json
degradation/lstm_fd003_sweep.json
degradation/lstm_frozen_sigma_decomposition.json
degradation/lstm_frozen_sigma_decomposition_relative.json
degradation/lstm_noise_armc.json
degradation/lstm_noise_main_arm.json
degradation/lstm_relative_half_life.json
degradation/noise_backbone_comparison_summary.json
degradation/threshold_crossover_leftside_extension.json
degradation/threshold_crossover_refinement.json
degradation/transformer_clamp_frac.json
degradation/transformer_frozen_sigma_decomposition.json
degradation/transformer_half_life.json
degradation/transformer_noise_armc.json
degradation/transformer_noise_main_arm.json
seeds/ensemble_independent_replication.json
seeds/ensemble_size_sweep_fd004_interval_score.json
seeds/samesplit_ensemble_control.json
""".split()

# These 3 files are only guaranteed reproducible except for their
# latency_ms_per_sample field (wall-clock timing) -- see REPRODUCE.md.
LATENCY_EXEMPT = {
    'clean/lstm_deep_ensemble.json',
    'clean/lstm_fd003_deep_ensemble.json',
    'clean/transformer_deep_ensemble.json',
}


def md5_of(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


if __name__ == '__main__':
    match, mismatch, latency_only, missing = [], [], [], []
    for rel in FILES:
        snap_path = os.path.join(RESULTS_DIR, '_snapshot', rel)
        cur_path = os.path.join(RESULTS_DIR, rel)
        if not os.path.exists(snap_path) or not os.path.exists(cur_path):
            missing.append(rel)
            continue
        m1, m2 = md5_of(snap_path), md5_of(cur_path)
        if m1 == m2:
            match.append(rel)
        elif rel in LATENCY_EXEMPT:
            latency_only.append((rel, m1, m2))
        else:
            mismatch.append((rel, m1, m2))

    print(f"MATCH ({len(match)}):")
    for f in match:
        print(f"  OK   {f}")
    if latency_only:
        print(f"\nDIFFER ONLY IN latency_ms_per_sample ({len(latency_only)}, expected -- see REPRODUCE.md):")
        for f, m1, m2 in latency_only:
            print(f"  LAT  {f}  snapshot={m1[:12]}  current={m2[:12]}")
    print(f"\nMISMATCH -- UNEXPECTED ({len(mismatch)}):")
    for f, m1, m2 in mismatch:
        print(f"  DIFF {f}  snapshot={m1[:12]}  current={m2[:12]}")
    if missing:
        print(f"\nMISSING ({len(missing)}): {missing}")
    print(f"\nSummary: {len(match)}/{len(FILES)} byte-identical, "
          f"{len(latency_only)} latency-only diff, {len(mismatch)} unexpected mismatch.")
