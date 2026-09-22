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

# Resume-skip guard workaround: these 3 scripts skip if their output already
# exists -- delete before every run.
rm -f ../results/generated/leakfree_t2/t2_transformer_mainarm_sweep_leakfree.json
rm -f ../results/generated/leakfree_t2/t2_transformer_armC_sweep_leakfree.json
rm -f ../results/generated/leakfree_t2/t2_degradation_sweep_leakfree.json

# --- Phase 0: seed results/generated/ with the frozen, checkpoint-derived
# clean-condition per-seed files that table2_clean_merge.py aggregates
# (these are not retrained here; see seed_generated_dir.py docstring) ---
run 00_seed_generated_dir seed_generated_dir.py

# --- Phase 1: base sweeps ---
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
run 14_threshold_960 threshold_crossover_refinement.py
run 24_threshold_leftside threshold_crossover_leftside_extension.py
run 25_bootstrap_by_order attribution_bootstrap_by_order.py

# --- Phase 3: drift controls ---
run 15_drift_controls drift_controls.py
run 16_bias_equal drift_bias_equal_magnitude_control.py
run 17_d1_shuffle drift_fixed_endpoint_shuffle_control.py

# --- Phase 4: maintenance ---
run 18_maintenance_two_sided maintenance_decision_two_sided.py
run 19_maintenance_one_sided maintenance_decision_one_sided.py
run 20_maintenance_rul_corrected maintenance_stress_test_rul_correction_v2.py

# --- Phase 5: reconstructed aggregations (checked to exact agreement, not bit-for-bit) ---
run 26_table2_clean_merge table2_clean_merge.py
run 27_clean_ece_pooled clean_ece_pooled_reconstruction.py

# --- Phase 6: publish results/generated/ outputs to their released names/locations ---
run 28_publish_results publish_results.py

echo "=== ALL DONE (run $1) ===" | tee -a "$LOG_DIR/_master.log"
