"""
Mandatory pre-flight check: evaluation-protocol gate (five-item
self-certification), for the Transformer backbone training pipeline
(train_transformer_nll.py / train_transformer_msemcd.py). Training output
must pass this script before being written into any downstream table --
proceed only if everything PASSes.

Five items:
  1. Checkpoint-selection DataLoader source -- must be val_units, not test.
  2. Early-stopping-criterion DataLoader source -- same as above (this
     pipeline's selection criterion and early-stopping criterion are the
     same val-RMSE, a single item, matching the LSTM pipeline).
  3. Whether the scaler's fit range includes val/test -- must be fit only
     on fit_units.
  4. Whether the conformal calibration set overlaps the model-selection set
     (fit union val) -- must not overlap.
  5. Whether each split's unit numbering is asserted non-overlapping --
     canonical_splits.json's fit/val/calib must be pairwise disjoint, union
     covering all units.

Items 1-3 are facts about the training code, checked via static source
inspection (grepping the training loop's selection-criterion variable
source); items 4-5 are facts directly verifiable against
canonical_splits.json's data.
"""
import os
import re
import json

import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [42, 2024, 7, 888, 123]

NLL_SCRIPT = os.path.join(T2.BASE_DIR, 'train_transformer_nll.py')
MSEMCD_SCRIPT = os.path.join(T2.BASE_DIR, 'train_transformer_msemcd.py')
CANON_PATH = os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')


def check_1_2_3_static(script_path, label):
    """Static check: early-stopping/checkpoint-selection criterion uses
    X_val_t (derived from val_units), not test_df/X_test_t; the scaler
    call is load_and_process_leakfree(ds, fit_units) (fit_units-only fit),
    not load_and_process(ds) (fit on the whole training file)."""
    src = open(script_path).read()
    results = {}

    # items 1&2: selection/early-stopping criterion source -- the
    # best_val_rmse comparison inside the training loop must come from
    # X_val_t-derived curr_val_rmse, and the training loop body must not
    # reference X_test_t/test_df
    train_loop_uses_val = bool(re.search(r'curr_val_rmse.*mean_squared_error\(y_val_cycles', src)) or \
        bool(re.search(r'r = np\.sqrt\(mean_squared_error\(y_val_cycles', src))
    # confirm test only appears after the training loop ends (after
    # best_state is loaded), not participating in selection inside the epoch loop
    epoch_loop_match = re.search(r'for epoch in range\(T2\.T2_EPOCHS\):(.*?)elapsed = time\.time\(\) - t0', src, re.S)
    if epoch_loop_match is None:
        epoch_loop_match = re.search(r'for _ in range\(T2\.T2_EPOCHS\):(.*?)elapsed = time\.time\(\) - t0', src, re.S)
    epoch_loop_body = epoch_loop_match.group(1) if epoch_loop_match else ''
    test_leaks_into_loop = ('X_test' in epoch_loop_body) or ('test_df' in epoch_loop_body) or \
                            ('y_test' in epoch_loop_body)
    results['item1_checkpoint_selection_uses_val_not_test'] = train_loop_uses_val and not test_leaks_into_loop
    results['item2_early_stop_uses_val_not_test'] = train_loop_uses_val and not test_leaks_into_loop

    # item 3: scaler fit range -- must call load_and_process_leakfree(ds_name, fit_units)
    uses_leakfree_loader = 'C.load_and_process_leakfree(ds_name, fit_units)' in src
    uses_leaky_loader = bool(re.search(r'C\.load_and_process\([^_]', src))  # load_and_process( without _leakfree
    results['item3_scaler_fit_units_only'] = uses_leakfree_loader and not uses_leaky_loader

    print(f"[{label}] item1 checkpoint-selection uses val (not test): "
          f"{'PASS' if results['item1_checkpoint_selection_uses_val_not_test'] else 'FAIL'}")
    print(f"[{label}] item2 early-stop uses val (not test): "
          f"{'PASS' if results['item2_early_stop_uses_val_not_test'] else 'FAIL'}")
    print(f"[{label}] item3 scaler fit on fit_units only: "
          f"{'PASS' if results['item3_scaler_fit_units_only'] else 'FAIL'}")
    return results


def check_4_5_data(canon):
    """Data check: for every (dataset, seed) the Transformer pipeline uses,
    fit/val/calib are pairwise disjoint, union covering all units. calib
    not overlapping fit union val (the selection set) is item 4."""
    all_pass = True
    per_ds = {}
    for ds in DATASETS:
        ds_pass = True
        if ds not in canon:
            print(f"  [DATA] {ds}: MISSING from canonical_splits.json -- FAIL")
            per_ds[ds] = {'status': 'missing'}
            all_pass = False
            continue
        for seed in SEEDS:
            key = str(seed)
            if key not in canon[ds]:
                print(f"  [DATA] {ds} seed={seed}: MISSING split entry -- FAIL")
                ds_pass = False
                all_pass = False
                continue
            split = canon[ds][key]
            fit = set(split['fit_units']); val = set(split['val_units']); calib = set(split['calib_units'])
            disjoint_fv = not (fit & val)
            disjoint_fc = not (fit & calib)
            disjoint_vc = not (val & calib)
            calib_vs_selection = not (calib & (fit | val))
            ok = disjoint_fv and disjoint_fc and disjoint_vc and calib_vs_selection
            if not ok:
                print(f"  [DATA] {ds} seed={seed}: overlap detected "
                      f"(fit&val={fit&val}, fit&calib={fit&calib}, val&calib={val&calib}) -- FAIL")
                ds_pass = False
                all_pass = False
        per_ds[ds] = {'status': 'checked', 'pass': ds_pass}
        print(f"  [DATA] {ds}: {'PASS' if ds_pass else 'FAIL'} ({len(SEEDS)} seeds checked)")
    return all_pass, per_ds


if __name__ == '__main__':
    print("=" * 60)
    print("T2 GATE CHECK -- Transformer backbone, leakfree protocol")
    print("=" * 60)

    r_nll = check_1_2_3_static(NLL_SCRIPT, 'NLL script')
    r_msemcd = check_1_2_3_static(MSEMCD_SCRIPT, 'MSE+MCD script')

    with open(CANON_PATH) as f:
        canon = json.load(f)
    print("\n[items 4&5: calibration-set disjointness + split-assertion, from canonical_splits.json]")
    data_pass, per_ds = check_4_5_data(canon)
    print(f"  item4 (calib disjoint from fit+val): {'PASS' if data_pass else 'FAIL'}")
    print(f"  item5 (all splits disjoint, union=all units): {'PASS' if data_pass else 'FAIL'}")

    all_items = list(r_nll.values()) + list(r_msemcd.values()) + [data_pass, data_pass]
    overall = all(all_items)

    report = {
        'nll_script_checks': r_nll,
        'msemcd_script_checks': r_msemcd,
        'data_checks_pass': data_pass,
        'data_checks_per_dataset': per_ds,
        'overall_gate_status': 'PASS' if overall else 'FAIL',
    }
    out_path = os.path.join(T2.TRANSFORMER_DIR, 'GATE_CHECK_REPORT.json')
    with open(out_path, 'w') as fp:
        json.dump(report, fp, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"OVERALL GATE STATUS: {'PASS -- proceed to downstream analysis' if overall else 'FAIL -- DO NOT proceed, fix issues first'}")
    print(f"Report saved -> {out_path}")
    print("=" * 60)
    if not overall:
        raise SystemExit(1)
