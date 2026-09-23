#!/bin/bash
export PYTHONHASHSEED=0
cd "$(dirname "$0")"

LOG_DIR=/tmp/repro_logs_$1
mkdir -p "$LOG_DIR"

run() {
  name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name ===" | tee -a "$LOG_DIR/_master.log"
  if python3 "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $name ===" | tee -a "$LOG_DIR/_master.log"
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (see $LOG_DIR/$name.log) ===" | tee -a "$LOG_DIR/_master.log"
    exit 1
  fi
}

# Resume-skip guard workaround: these scripts skip if their output already
# exists -- delete before every run.
rm -f ../results/generated/leakfree_t2/t2_transformer_mainarm_sweep_leakfree.json
rm -f ../results/generated/leakfree_t2/t2_transformer_armC_sweep_leakfree.json
rm -f ../results/generated/leakfree_t2/t2_degradation_sweep_leakfree.json
rm -f ../results/generated/leakfree_t2/t2_transformer_per_engine_coverage_leakfree.json

# --- Phase 0: re-evaluate (inference only, no retraining) the clean-
# condition files that depend on the current test-truth convention, by
# reloading existing checkpoints under results/checkpoints/ ---
run 00a_eval_nll_clean eval_nll_clean_b1.py
run 00b_eval_splitcp eval_splitcp_b1.py
run 00c_eval_mcdropout eval_mcdropout_b1.py
run 00d_fd003_mechanism_eval eval_fd003_mechanism_b1.py
run 00e_samesplit_ensemble eval_samesplit_ensemble_b1.py

# --- Phase 1: base degradation sweeps (sensor-only perturbation on
# FD002/FD004; FD001/FD003 unchanged since they have no setting columns) ---
run 01_lstm_main_sweep run_sweep_noise_lstm.py
run 02_lstm_fd003_sweep run_sweep_noise_lstm_fd003.py
run 03_t2_mainarm run_sweep_noise_transformer_mainarm.py
run 04_t2_armc run_sweep_noise_transformer_armc.py
run 05_t2_partB_degradation run_sweep_degradation_transformer.py
run 06_lstm_clamp_frac clamp_frac.py
run 07_t2_mechanism mechanism_transformer.py
run 08_t2_degradation_halflife half_life_degradation.py
run 09_lstm_halflife_regen half_life_lstm_regen.py
run 21_step5j_dose_response dose_response_frozen_sigma.py
run 22_step5m_relative frozen_sigma_relative.py

# --- Phase 2: attribution ---
run 10_frozen2x2 attribution_frozen_2x2_nearest_grid.py
run 11_stepB_perseed attribution_raw_per_seed.py
run 12_stepG_transformer attribution_interp_transformer.py
run 13_stepH_lstm attribution_interp_lstm.py
# threshold_crossover_refinement.py reads a frozen comparison file that has
# no surviving generator script -- seed it into results/generated/ first.
run 13b_seed_frozen_D3 seed_generated_dir.py
run 14_threshold_960 threshold_crossover_refinement.py
run 24_threshold_leftside threshold_crossover_leftside_extension.py
run 25_bootstrap_by_order attribution_bootstrap_by_order.py

# --- Phase 3: drift controls ---
run 15_drift_controls drift_controls.py
run 16_bias_equal drift_bias_equal_magnitude_control.py
run 17_d1_shuffle drift_fixed_endpoint_shuffle_control.py

# --- Phase 4: maintenance / decision stress test ---
run 18_maintenance_two_sided maintenance_decision_two_sided.py
run 19_maintenance_one_sided maintenance_decision_one_sided.py
run 20_maintenance_rul_corrected maintenance_stress_test_rul_correction_v2.py

# --- Phase 5: Table I (2x2 protocol, 4 quadrants, no retraining -- reads
# the T/W', V/W', V/F, T/F checkpoints already shipped under
# results/checkpoints/. T/W' and V/W' (results/checkpoints/lstm_2x2_controlled/)
# were trained once by protocol_2x2_controlled_retrain.py -- that script is
# not part of this pipeline and is not asserted bit-reproducible (training,
# not inference); see code/superseded/README.md for why the original T/W/V/W
# cells were replaced. ---
run 30_table1 build_table1.py

# --- Phase 6: Table II clean benchmark + cost tables + per-engine +
# ensembles. Per-engine / ensemble / fair-calibration / interval-score
# must run BEFORE table2_merge and the cost tables, which read their
# output files. ---
run 33_fair_main_table fair_calibration_main_table.py
run 34_fair_sigma_fixed fair_calibration_sigma_fixed.py
run 35_interval_score interval_score.py
run 39_per_engine_lstm eval_lstm_per_engine_coverage.py
run 40_per_engine_transformer eval_transformer_per_engine_coverage.py
run 41_per_engine_fd003 eval_lstm_fd003_per_engine_coverage.py
run 42_deep_ensemble_lstm eval_lstm_deep_ensemble.py
run 43_deep_ensemble_fd003 eval_lstm_fd003_deep_ensemble.py
run 44_deep_ensemble_transformer eval_transformer_deep_ensemble.py
run 45_mse_row_recomputed eval_lstm_mse_fixed_training_residual.py
run 31_table2_merge table2_clean_merge.py
run 32_ece_pooled clean_ece_pooled_reconstruction.py
run 36_cost_table_lstm cost_table_lstm.py
run 37_cost_table_lstm_fd003 cost_table_lstm_fd003.py
run 38_cost_table_transformer cost_table_transformer.py
run 46_armc_mu_std_lstm eval_lstm_armc_mu_std.py
run 47_armc_mu_std_fd003 eval_lstm_fd003_armc_mu_std.py
run 48_ensemble_scale_sweep eval_lstm_ensemble_scale_sweep.py
run 49_ensemble_scale_sweep_fd003 eval_lstm_fd003_ensemble_scale_sweep.py
run 50_ensemble_size_replication ensemble_size_and_independent_replication.py
run 51_compare_backbones compare_backbones_noise.py
run 52_compare_degradations compare_degradations_dose_response.py
run 54_ensemble_n5_vs_n15 eval_lstm_ensemble_n5_vs_n15.py

# --- Phase 7: pre-flight diagnostic now cited in the main text (not a
# run_pipeline.sh output historically, but held to the same reproducibility
# bar -- see REPRODUCE.md) ---
run 55_a1_perturbation_target diagnostic_perturbation_target_a1.py

# --- Phase 8: publish results/generated/ outputs to their released
# names/locations under results/{category}/ ---
run 53_publish_results publish_results.py

echo "=== ALL DONE (run $1) ===" | tee -a "$LOG_DIR/_master.log"
