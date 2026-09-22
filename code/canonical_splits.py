"""
STEP A：全项目唯一的发动机级三向切分（fit/val/calib），一次算好、落盘，
STEP0/STEP1/STEP3 全部从这份文件读取，不各自重新切分（保证三条训练线
的 train/val 划分逐发动机一致，方法间可比；calib 与三者也互不重叠）。
"""
import os
import json

import pandas as pd

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')

if __name__ == '__main__':
    out = {}
    report_lines = []
    for ds in C.DATASETS:
        col_names = C.INDEX_NAMES + C.SETTING_NAMES + C.SENSOR_NAMES
        train_df = pd.read_csv(os.path.join(C.DATA_DIR, f'train_{ds}.txt'),
                                sep=r'\s+', header=None, names=col_names)
        all_units = sorted(train_df['unit_nr'].unique().tolist())
        out[ds] = {}
        report_lines.append(f"\n## {ds}  (total train engines = {len(all_units)})\n")
        for seed in C.SEEDS:
            fit_units, val_units, calib_units = C.compute_canonical_split(all_units, seed)
            assert set(fit_units) | set(val_units) | set(calib_units) == set(all_units)
            assert not (set(fit_units) & set(val_units))
            assert not (set(fit_units) & set(calib_units))
            assert not (set(val_units) & set(calib_units))
            out[ds][str(seed)] = {'fit_units': fit_units, 'val_units': val_units, 'calib_units': calib_units}
            report_lines.append(
                f"- seed={seed}: fit(n={len(fit_units)})={fit_units}\n"
                f"  val(n={len(val_units)})={val_units}\n"
                f"  calib(n={len(calib_units)})={calib_units}\n"
            )
            print(f"{ds} seed={seed}: fit={len(fit_units)} val={len(val_units)} calib={len(calib_units)} "
                  f"(sum={len(fit_units)+len(val_units)+len(calib_units)}, total={len(all_units)})")

    out_path = os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved -> {out_path}")

    notes_path = os.path.join(RESULTS_DIR, 'CANONICAL_SPLITS_NOTES.md')
    with open(notes_path, 'w') as fp:
        fp.write("# 全项目唯一的发动机级三向切分（fit/val/calib），逐发动机编号\n\n")
        fp.write("生成脚本：`canonical_splits.py`，数据：`canonical_splits.json`。\n")
        fp.write("STEP0/STEP1/STEP3 全部从这份文件读取 fit_units/val_units（+STEP3 额外读 "
                  "calib_units），不各自重新切分。三者互不重叠（assert 已在生成时验证），"
                  "并集等于该数据集全部 train 发动机。\n")
        fp.write("".join(report_lines))
    print(f"Saved -> {notes_path}")
    print("STEP A complete.")
