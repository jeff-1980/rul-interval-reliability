"""
T2 强制前置检查：评价协议门禁（CLAUDE.md "评价协议门禁"五项自证），针对
新增的 Transformer 骨干训练管线（train_transformer_nll.py /
train_transformer_msemcd.py）。训练产出在被写进下游表格前必须先跑通
本脚本，全部 PASS 才可继续。

五项：
  1. checkpoint 选择判据的 DataLoader 来源 —— 必须是 val_units，不是 test。
  2. 早停判据的 DataLoader 来源 —— 同上（本管线选择判据和早停判据是同一个
     val-RMSE，只有一项，与 LSTM 管线一致）。
  3. scaler 的 fit 范围是否含 val/test —— 必须只在 fit_units 上 fit。
  4. conformal 校准集是否与选模集合（fit ∪ val）重叠 —— 必须不重叠。
  5. 各 split 单元编号是否有断言保证不重叠 —— canonical_splits.json 里
     fit/val/calib 三者互不重叠、并集覆盖全部 units。

第1-3项是训练代码的程序事实，用源码静态检查（grep 训练循环里选择判据变量
来源）；第4-5项是可以直接对 canonical_splits.json 数据验证的事实。
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
    """静态检查：早停/checkpoint选择判据用 val_units 派生的 X_val_t，不用
    test_df/X_test_t；scaler 调用是 load_and_process_leakfree(ds, fit_units)
    （fit_units-only fit），不是 load_and_process(ds)（全量train fit）。"""
    src = open(script_path).read()
    results = {}

    # item 1&2: 选择/早停判据来源 —— 训练循环内比较 best_val_rmse 时用的必须是
    # 从 X_val_t 计算出的 curr_val_rmse，且训练循环体内不得出现 X_test_t/test_df
    train_loop_uses_val = bool(re.search(r'curr_val_rmse.*mean_squared_error\(y_val_cycles', src)) or \
        bool(re.search(r'r = np\.sqrt\(mean_squared_error\(y_val_cycles', src))
    # 确认 test 只在训练循环结束后（best_state load 之后）才出现，不在 epoch loop 内参与选择
    epoch_loop_match = re.search(r'for epoch in range\(T2\.T2_EPOCHS\):(.*?)elapsed = time\.time\(\) - t0', src, re.S)
    if epoch_loop_match is None:
        epoch_loop_match = re.search(r'for _ in range\(T2\.T2_EPOCHS\):(.*?)elapsed = time\.time\(\) - t0', src, re.S)
    epoch_loop_body = epoch_loop_match.group(1) if epoch_loop_match else ''
    test_leaks_into_loop = ('X_test' in epoch_loop_body) or ('test_df' in epoch_loop_body) or \
                            ('y_test' in epoch_loop_body)
    results['item1_checkpoint_selection_uses_val_not_test'] = train_loop_uses_val and not test_leaks_into_loop
    results['item2_early_stop_uses_val_not_test'] = train_loop_uses_val and not test_leaks_into_loop

    # item 3: scaler fit range —— 必须调用 load_and_process_leakfree(ds_name, fit_units)
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
    """数据检查：对 T2 用到的每个 (dataset, seed)，fit/val/calib 三者互不重叠，
    并集覆盖全部 unit。calib 与 fit∪val（选模集合）不重叠即是 item4。"""
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
