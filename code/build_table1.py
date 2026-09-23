"""
R8-B3 + R9-Part1：Table I（2x2 协议实验）四格在新协议下重新推理。

R9-Part1（本轮）：T/W、V/W 两格换成"受控"版本 T/W'、V/W'——旧版 T/W 在
全部train发动机上拟合（100%），旧版 V/W 用 80% fit，与 V/F、T/F 用的
canonical_splits.json 60% fit_units 不是同一拟合数据量，混进了一个额外
变量。T/W'、V/W' 由 protocol_2x2_controlled_retrain.py 重新训练，
fit_units 与 V/F、T/F 完全相同，只保留"选模判据"和"scaler拟合范围"两个
变量。旧版 T/W（全部发动机）、V/W（80%）的 checkpoint 已移入
code/superseded/ + results/superseded/，不再是 Table I 的数据来源，仅存
档供溯源。

  T/W': results/generated/checkpoints_r9_2x2_controlled/{ds}_LSTM_testselect_wholefilescaler_seed{seed}.pt
        （protocol_2x2_controlled_retrain.py 训出，fit_units=60%canonical，
        test-select + whole-file scaler）
  V/W': results/generated/checkpoints_r9_2x2_controlled/{ds}_LSTM_valselect_wholefilescaler_seed{seed}.pt
        （同上，val-select + whole-file scaler）
  V/F : results/generated/checkpoints_leakfree/{ds}_LSTM_seed{seed}.pt
        （train_lstm.py 训出，val-select + fit-only scaler，
        本项目主协议）
  T/F : results/generated/checkpoints_leakfree_r2/{ds}_LSTM_testselect_fitonlyscaler_seed{seed}.pt
        （protocol_2x2_quadrant4.py 训出，test-select + fit-only scaler）

T/W'、V/W' 用"整份训练文件"拟合的 scaler（C.load_and_process，泄漏版）；
V/F、T/F 用"只在 fit_units 上拟合"的 scaler（C.load_and_process_leakfree，
本项目主协议）——这是每格自己训练时用的同一个 scaler，不是本脚本另外
选的。四格的 fit_units（训练数据本身，60% canonical）现在完全相同。

数据集范围：FD001/FD002/FD004（与 protocol_2x2_quadrant4.py 一致，不含
FD003——FD003 本来就没有泄漏基线）。5 seeds，每格每个(ds,seed)一条记录。

只做推理，不重训，不改 main.tex。
"""
import os
import json

import numpy as np
import torch

import common as C

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R8_DIR = os.path.join(RESULTS_DIR, 'leakfree_r8')
os.makedirs(R8_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD004']
QUADRANTS = ['T_W', 'V_W', 'V_F', 'T_F']


def ckpt_path_for(quadrant, ds, seed):
    if quadrant == 'T_W':
        return os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm_2x2_controlled', f"{ds}_LSTM_testselect_wholefilescaler_seed{seed}.pt")
    if quadrant == 'V_W':
        return os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm_2x2_controlled', f"{ds}_LSTM_valselect_wholefilescaler_seed{seed}.pt")
    if quadrant == 'V_F':
        return os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_LSTM_seed{seed}.pt")
    if quadrant == 'T_F':
        return os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm_drift_controls', f"{ds}_LSTM_testselect_fitonlyscaler_seed{seed}.pt")
    raise ValueError(quadrant)


def test_data_for(quadrant, ds, seed, canon):
    """returns (X_test, y_test) under the SAME scaler used to train this quadrant's
    checkpoint, and the NEW (B2, RUL-1) test-truth convention (already baked into
    C.create_sequences)."""
    if quadrant in ('T_W', 'V_W'):
        train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)
    else:
        fit_units = canon[ds][str(seed)]['fit_units']
        train_df, test_df, true_ruls, feat_cols, _ = C.load_and_process_leakfree(ds, fit_units)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    return X_test, y_test


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {q: {} for q in QUADRANTS}
    for quadrant in QUADRANTS:
        for ds in DATASETS:
            print(f"\n{'=' * 20} {quadrant} / {ds} {'=' * 20}")
            result[quadrant][ds] = []
            for seed in C.SEEDS:
                ckpt = ckpt_path_for(quadrant, ds, seed)
                model = C.load_checkpoint_model(ckpt, device)
                X_test, y_test = test_data_for(quadrant, ds, seed, canon)
                X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                with torch.no_grad():
                    mu_out, log_sigma_out = model(X_t)
                mu = np.clip(mu_out.cpu().numpy().flatten() * 125.0, 0, 125)
                sigma = np.exp(log_sigma_out.cpu().numpy().flatten()) * 125.0
                rmse, score = C.rmse_score(y_test, mu)
                picp, mpiw = C.picp_mpiw(y_test, mu, sigma, z=C.Z_SCORE)
                ece = C.compute_ece(mu, sigma, y_test, np.arange(0.05, 1.00, 0.05))
                result[quadrant][ds].append({'seed': seed, 'rmse': rmse, 'score': score,
                                              'picp': picp, 'mpiw': mpiw, 'ece': ece})
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f"  seed={seed}: RMSE={rmse:.3f} Score={score:.1f} PICP={picp:.3f} "
                      f"MPIW={mpiw:.2f} ECE={ece:.4f}")

    per_seed_path = os.path.join(R8_DIR, 'table1_2x2_per_seed.json')
    with open(per_seed_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {per_seed_path}")

    # ---- Table I summary: mean over 5 seeds per (quadrant, ds), plus the
    # Sel./Norm./Int. contrasts the paper's Table I reports ----
    def mean_of(cells, key):
        return float(np.mean([c[key] for c in cells]))

    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for q in QUADRANTS:
            cells = result[q][ds]
            summary[ds][q] = {k: mean_of(cells, k) for k in ['rmse', 'score', 'picp', 'mpiw', 'ece']}

        # Sel. = mean over normaliser levels of (T-V) on RMSE/score
        # Norm. = mean over selection levels of (W-F) on RMSE/score
        # Int. = (T/W - V/W) - (T/F - V/F)
        for metric in ['rmse', 'score']:
            TW, VW, VF, TF = (summary[ds][q][metric] for q in ['T_W', 'V_W', 'V_F', 'T_F'])
            sel = ((TW - VW) + (TF - VF)) / 2.0
            norm = ((TW - TF) + (VW - VF)) / 2.0
            inter = (TW - VW) - (TF - VF)
            summary[ds].setdefault('contrasts', {})[metric] = {'Sel': sel, 'Norm': norm, 'Int': inter}

    summary_path = os.path.join(R8_DIR, 'table1_2x2_summary.json')
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=float)
    print(f"Saved -> {summary_path}")

    print("\n=== Table I summary (mean of 5 seeds) ===")
    for ds in DATASETS:
        print(f"\n{ds}:")
        for q in QUADRANTS:
            s = summary[ds][q]
            print(f"  {q}: RMSE={s['rmse']:.3f} Score={s['score']:.1f} PICP={s['picp']:.3f} "
                  f"MPIW={s['mpiw']:.2f} ECE={s['ece']:.4f}")
        print(f"  Sel/Norm/Int (RMSE): {summary[ds]['contrasts']['rmse']}")
        print(f"  Sel/Norm/Int (Score): {summary[ds]['contrasts']['score']}")
