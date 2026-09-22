"""
Copies the already-verified, non-stochastic clean-condition per-seed
evaluation files (checkpoint-derived, not retraining outputs) from their
released location under results/clean/ into results/generated/ under the
internal names table2_clean_merge.py expects. These files are direct,
deterministic functions of the checkpoints under results/checkpoints/
(already shipped) -- regenerating them would mean retraining every
checkpoint from scratch, which this repository does not assert bit-
reproducible (see README.md). table2_clean_merge.py itself performs no
new computation; it only re-aggregates these files, and is checked for
exact agreement against the released results/clean/table2_clean_full.json.
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
CLEAN_DIR = os.path.join(PROJ_DIR, 'results', 'clean')
DEGRADATION_DIR = os.path.join(PROJ_DIR, 'results', 'degradation')
GEN_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
GEN_T2_DIR = os.path.join(GEN_DIR, 'leakfree_t2')
GEN_R2_DIR = os.path.join(GEN_DIR, 'leakfree_r2')
GEN_R3_DIR = os.path.join(GEN_DIR, 'leakfree_r3')

os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(GEN_T2_DIR, exist_ok=True)
os.makedirs(GEN_R2_DIR, exist_ok=True)
os.makedirs(GEN_R3_DIR, exist_ok=True)

MAPPING = {
    'lstm_fd003_clean_per_seed.json': 'stepFD003_nll_leakfree_results.json',
    'lstm_fd003_mcdropout_mse_per_seed.json': 'stepFD003_mcdropout_mse_leakfree_results.json',
    'lstm_fd003_split_cp_per_seed.json': 'stepFD003_splitcp_leakfree_results.json',
    'lstm_fd003_deep_ensemble.json': 'stepFD003_deep_ensemble_leakfree.json',
    'lstm_clean_per_seed.json': 'step0c_leakfree_results.json',
    'lstm_mcdropout_per_seed.json': 'mcdropout_fixed_leakfree.json',
    'lstm_mse_fixed_training_residual_per_seed.json': 'mse_row_recomputed_leakfree.json',
    'lstm_split_cp_per_seed.json': 'split_cp_leakfree.json',
    'lstm_deep_ensemble.json': 'deep_ensemble_leakfree.json',
    'lstm_per_engine_coverage.json': 'per_engine_coverage_leakfree.json',
    'lstm_fd003_per_engine_coverage.json': 'stepFD003_per_engine_coverage_leakfree.json',
}
MAPPING_T2 = {
    'transformer_clean_per_seed.json': 't2_transformer_nll_leakfree_results.json',
    'transformer_mcdropout_mse_per_seed.json': 't2_transformer_msemcd_leakfree_results.json',
    'transformer_split_cp_per_seed.json': 't2_transformer_splitcp_leakfree_results.json',
    'transformer_deep_ensemble.json': 't2_transformer_ensemble_clean_leakfree.json',
    'transformer_per_engine_coverage.json': 't2_transformer_per_engine_coverage_leakfree.json',
}
MAPPING_R2 = {
    'interval_score_wis_clean.json': 'interval_score_wis_clean.json',
    'fair_calibration_main_table.json': 'fair_calibration_main_table.json',
}
# D3_refined_grid_transformer_fd001_ensemble.json has no surviving generator
# script anywhere in this codebase (discovered during the export -- it is
# read only, as a "prior_D3_refined_crossover" comparison field, by
# threshold_crossover_refinement.py, which does not itself produce it).
# It ships as a frozen data file under results/degradation/; seed it back
# for that one read.
MAPPING_R3 = {
    'threshold_refined_grid_transformer_fd001_ensemble.json': 'D3_refined_grid_transformer_fd001_ensemble.json',
}

if __name__ == '__main__':
    for src_name, dst_name in MAPPING.items():
        shutil.copy2(os.path.join(CLEAN_DIR, src_name), os.path.join(GEN_DIR, dst_name))
    for src_name, dst_name in MAPPING_T2.items():
        shutil.copy2(os.path.join(CLEAN_DIR, src_name), os.path.join(GEN_T2_DIR, dst_name))
    for src_name, dst_name in MAPPING_R2.items():
        shutil.copy2(os.path.join(CLEAN_DIR, src_name), os.path.join(GEN_R2_DIR, dst_name))
    for src_name, dst_name in MAPPING_R3.items():
        shutil.copy2(os.path.join(DEGRADATION_DIR, src_name), os.path.join(GEN_R3_DIR, dst_name))
    print(f"Seeded {len(MAPPING) + len(MAPPING_T2) + len(MAPPING_R2) + len(MAPPING_R3)} "
          f"frozen files into results/generated/.")
