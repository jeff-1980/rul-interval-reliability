"""
Copies each script's output from results/generated/ (the internal working
directory used while reproducing this repository's results) to its
released name and location under results/{category}/ -- the layout
documented in README.md and results/table_provenance.csv. Run this after
run_pipeline.sh's scripts have populated results/generated/.
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
SRC = os.path.join(PROJ_DIR, 'results', 'generated')
DST = os.path.join(PROJ_DIR, 'results')

# category -> list of (src_relpath, dst_filename)
mapping = {
 'clean': [
   ('canonical_splits.json', None),  # actually top-level, handled separately
   ('leakfree_r8/table1_2x2_per_seed.json', 'table1_2x2_per_seed.json'),
   ('leakfree_r8/table1_2x2_summary.json', 'table1_2x2_summary.json'),
   ('step0c_leakfree_results.json', 'lstm_clean_per_seed.json'),
   ('step0c_leakfree_ece_agg.json', 'lstm_clean_ece_pooled.json'),
   ('mcdropout_fixed_leakfree.json', 'lstm_mcdropout_per_seed.json'),
   ('deep_ensemble_leakfree.json', 'lstm_deep_ensemble.json'),
   ('split_cp_leakfree.json', 'lstm_split_cp_per_seed.json'),
   ('mse_row_recomputed_leakfree.json', 'lstm_mse_fixed_training_residual_per_seed.json'),
   ('per_engine_coverage_leakfree.json', 'lstm_per_engine_coverage.json'),
   ('stepFD003_nll_leakfree_results.json', 'lstm_fd003_clean_per_seed.json'),
   ('stepFD003_mcdropout_mse_leakfree_results.json', 'lstm_fd003_mcdropout_mse_per_seed.json'),
   ('stepFD003_deep_ensemble_leakfree.json', 'lstm_fd003_deep_ensemble.json'),
   ('stepFD003_splitcp_leakfree_results.json', 'lstm_fd003_split_cp_per_seed.json'),
   ('stepFD003_per_engine_coverage_leakfree.json', 'lstm_fd003_per_engine_coverage.json'),
   ('COST_TABLE_FD001_leakfree.csv', 'lstm_cost_table_fd001.csv'),
   ('COST_TABLE_FD002_leakfree.csv', 'lstm_cost_table_fd002.csv'),
   ('COST_TABLE_FD003_leakfree.csv', 'lstm_cost_table_fd003.csv'),
   ('COST_TABLE_FD004_leakfree.csv', 'lstm_cost_table_fd004.csv'),
   ('leakfree_t2/t2_transformer_nll_leakfree_results.json', 'transformer_clean_per_seed.json'),
   ('leakfree_t2/t2_transformer_msemcd_leakfree_results.json', 'transformer_mcdropout_mse_per_seed.json'),
   ('leakfree_t2/t2_transformer_ensemble_clean_leakfree.json', 'transformer_deep_ensemble.json'),
   ('leakfree_t2/t2_transformer_splitcp_leakfree_results.json', 'transformer_split_cp_per_seed.json'),
   ('leakfree_t2/t2_transformer_per_engine_coverage_leakfree.json', 'transformer_per_engine_coverage.json'),
   ('leakfree_t2/COST_TABLE_FD001_Transformer_leakfree.csv', 'transformer_cost_table_fd001.csv'),
   ('leakfree_t2/COST_TABLE_FD002_Transformer_leakfree.csv', 'transformer_cost_table_fd002.csv'),
   ('leakfree_t2/COST_TABLE_FD003_Transformer_leakfree.csv', 'transformer_cost_table_fd003.csv'),
   ('leakfree_t2/COST_TABLE_FD004_Transformer_leakfree.csv', 'transformer_cost_table_fd004.csv'),
   ('leakfree_r2/table2_clean_full.json', 'table2_clean_full.json'),
   ('leakfree_r2/fair_calibration_main_table.json', 'fair_calibration_main_table.json'),
   ('leakfree_r2/fair_calibration_sigma_fixed.json', 'fair_calibration_sigma_fixed.json'),
   ('leakfree_r2/interval_score_wis_clean.json', 'interval_score_wis_clean.json'),
   ('leakfree_r2/protocol_2x2_quadrant4_testselect_fitonlyscaler.json', 'protocol_2x2_quadrant4_testselect_fitonlyscaler.json'),
   ('leakfree/step0d_qc_report.json', 'lstm_extra_seeds_qc_report.json'),
   ('leakfree/stepFD003_step0d_qc_report.json', 'lstm_fd003_extra_seeds_qc_report.json'),
   ('step0d_extra_seeds_leakfree_results.json', 'lstm_extra_seeds_results.json'),
   ('stepFD003_extra_seeds_leakfree_results.json', 'lstm_fd003_extra_seeds_results.json'),
   ('leakfree/shap_spearman_vs_earlier.json', 'shap_stability_spearman.json'),
   ('leakfree/shap_spearman_vs_earlier_ns200.json', 'shap_stability_spearman_ns200.json'),
   ('leakfree/shap_spearman_vs_earlier_ns800.json', 'shap_stability_spearman_ns800.json'),
   ('leakfree/shap_ranking_diff.csv', 'shap_ranking_diff.csv'),
   ('leakfree/shap_ranking_diff_ns200.csv', 'shap_ranking_diff_ns200.csv'),
   ('leakfree/shap_ranking_diff_ns800.csv', 'shap_ranking_diff_ns800.csv'),
   ('leakfree/shap_ranking_leakfree.csv', 'shap_ranking.csv'),
   ('leakfree/shap_ranking_leakfree_ns200.csv', 'shap_ranking_ns200.csv'),
   ('leakfree/shap_ranking_leakfree_ns800.csv', 'shap_ranking_ns800.csv'),
   ('leakfree/shap_spearman_leakfree.json', 'shap_spearman.json'),
   ('leakfree/shap_spearman_leakfree_ns200.json', 'shap_spearman_ns200.json'),
   ('leakfree/shap_spearman_leakfree_ns800.json', 'shap_spearman_ns800.json'),
 ],
 'degradation': [
   ('noise_sensitivity_leakfree.json', 'lstm_noise_main_arm.json'),
   ('noise_sensitivity_leakfree_armC.json', 'lstm_noise_armc.json'),
   ('noise_sensitivity_leakfree_FD003.json', 'lstm_fd003_noise_main_arm.json'),
   ('noise_sensitivity_leakfree_armC_FD003.json', 'lstm_fd003_noise_armc.json'),
   ('dose_response_feat_oob_leakfree.json', 'dose_response_feat_oob.json'),
   ('frozen_sigma_decomposition_leakfree.json', 'lstm_frozen_sigma_decomposition.json'),
   ('frozen_sigma_decomposition_RELATIVE_leakfree.json', 'lstm_frozen_sigma_decomposition_relative.json'),
   ('leakfree/clamp_frac_leakfree.json', 'lstm_clamp_frac_main_arm.json'),
   ('leakfree/clamp_frac_leakfree_armC.json', 'lstm_clamp_frac_armc.json'),
   ('leakfree/relative_half_life_feat_oob.json', 'lstm_relative_half_life.json'),
   ('leakfree/armC_mu_std_leakfree.json', 'lstm_armc_mu_std.json'),
   ('leakfree/FD003_armC_mu_std_leakfree.json', 'lstm_fd003_armc_mu_std.json'),
   ('leakfree/ensemble_scale_sweep_leakfree.json', 'lstm_ensemble_scale_sweep.json'),
   ('leakfree/FD003_ensemble_scale_sweep_leakfree.json', 'lstm_fd003_ensemble_scale_sweep.json'),
   ('leakfree/n5_vs_n15_variance_comparison_leakfree.json', 'lstm_ensemble_n5_vs_n15_variance.json'),
   ('leakfree/FD003_n5_vs_n15_variance_comparison_leakfree.json', 'lstm_fd003_ensemble_n5_vs_n15_variance.json'),
   ('leakfree/FD003_sweep_leakfree.json', 'lstm_fd003_sweep.json'),
   ('leakfree/FD003_frozen_sigma_decomposition_leakfree.json', 'lstm_fd003_frozen_sigma_decomposition.json'),
   ('leakfree_t2/t2_transformer_mainarm_sweep_leakfree.json', 'transformer_noise_main_arm.json'),
   ('leakfree_t2/t2_transformer_armC_sweep_leakfree.json', 'transformer_noise_armc.json'),
   ('leakfree_t2/t2_transformer_half_life_feat_oob.json', 'transformer_half_life.json'),
   ('leakfree_t2/t2_transformer_clamp_frac.json', 'transformer_clamp_frac.json'),
   ('leakfree_t2/t2_transformer_frozen_sigma_decomposition.json', 'transformer_frozen_sigma_decomposition.json'),
   ('leakfree_t2/t2_degradation_sweep_leakfree.json', 'degradation_sweep_bias_gain_drift.json'),
   ('leakfree_t2/t2_degradation_half_life_feat_oob.json', 'degradation_half_life_bias_gain_drift.json'),
   ('leakfree_t2/t2_degradation_mpiw.json', 'degradation_mpiw_bias_gain_drift.json'),
   ('leakfree_t2/t2_partA_comparison_summary.json', 'noise_backbone_comparison_summary.json'),
   ('leakfree_t2/t2_partB_dose_response_overlap_summary.json', 'degradation_dose_response_overlap_summary.json'),
   ('leakfree_r2/drift_controls.json', 'drift_controls.json'),
   ('leakfree_r3/D1_fixed_endpoint_shuffle.json', 'drift_fixed_endpoint_shuffle_control.json'),
   ('leakfree_r3/D3_refined_grid_transformer_fd001_ensemble.json', 'threshold_refined_grid_transformer_fd001_ensemble.json'),
   ('leakfree_r4/threshold_960_refinement.json', 'threshold_crossover_refinement.json'),
   ('leakfree_r6/threshold_960_leftside_extension.json', 'threshold_crossover_leftside_extension.json'),
 ],
 'attribution': [
   ('leakfree_r2/frozen_decomposition_2x2.json', 'frozen_output_decomposition_2x2_nearest_grid.json'),
   ('leakfree_r3/B_attribution_raw_per_seed.json', 'attribution_raw_per_seed_nearest_grid.json'),
   ('leakfree_r3/G_transformer_exact_interp_attribution.json', 'attribution_transformer_exact_interp.json'),
   ('leakfree_r3/H_lstm_exact_interp_attribution.json', 'attribution_lstm_exact_interp.json'),
   ('leakfree_r3/bootstrap_absmu_minus_abssigma.json', 'attribution_bootstrap_order_averaged_nearest_grid.json'),
   ('leakfree_r3/bootstrap_by_order_exact_interp.json', 'attribution_bootstrap_by_order_exact_interp.json'),
 ],
 'decision': [
   ('leakfree_r2/maintenance_decision.json', 'maintenance_decision_two_sided.json'),
   ('leakfree_r3/C_maintenance_full_onesided.json', 'maintenance_decision_one_sided.json'),
   ('leakfree_r4/C_maintenance_rul_corrected.json', 'maintenance_decision_rul_corrected_v1.json'),
   ('leakfree_r4/C_maintenance_rul_corrected_v2.json', 'maintenance_decision_rul_corrected_v2.json'),
   ('leakfree_r4/exact_match_log.json', 'maintenance_decision_rul_correction_exact_match_check.json'),
 ],
 'latency': [
   ('leakfree_r2/latency_redesign_run1.json', 'lstm_latency.json'),
   ('leakfree_t2/latency_T2_transformer_batch1024_run1.json', 'transformer_latency_batch1024.json'),
   ('leakfree_t2/latency_T2_unified_platform_batch512_run1.json', 'transformer_latency_unified_platform_batch512.json'),
 ],
 'seeds': [
   ('leakfree_r4/samesplit_ensemble.json', 'samesplit_ensemble_control.json'),
   ('leakfree_r3/A2_fd004_msweep_IS.json', 'ensemble_size_sweep_fd004_interval_score.json'),
   ('leakfree_r3/E_ensemble_independent_replication.json', 'ensemble_independent_replication.json'),
 ],
 'diagnostics': [
   ('leakfree_r8/A1_perturbation_target_diagnostic.json', 'A1_perturbation_target_diagnostic.json'),
   ('leakfree_r8/A2_target_alignment_diagnostic.json', 'A2_target_alignment_diagnostic.json'),
 ],
}

count = 0
missing = []
for cat, items in mapping.items():
    catdir = os.path.join(DST, cat)
    os.makedirs(catdir, exist_ok=True)
    for src_rel, dst_name in items:
        if dst_name is None:
            continue
        src_path = os.path.join(SRC, src_rel)
        if not os.path.exists(src_path):
            missing.append(src_rel)
            continue
        shutil.copy2(src_path, os.path.join(catdir, dst_name))
        count += 1

print(f"Copied {count} files.")
if missing:
    print("MISSING SOURCE FILES:")
    for m in missing:
        print(" ", m)
