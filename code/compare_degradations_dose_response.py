"""
Core question: plotting the three deterministic degradation types
(bias/drift/gain) plus Gaussian noise (arm C) on the same f_oob axis --
do the dose-response curves coincide?

Method: for each (backbone, dataset, method), take each perturbation
type's (feat_oob, PICP) point set, interpolate PICP on a common feat_oob
grid, and measure "coincidence" via max absolute deviation and mean
absolute deviation. No curve fitting -- only the measured separation is
reported.

Gaussian-noise baseline: for LSTM, uses the existing arm-C data (the arm-C
portion of `noise_sensitivity_leakfree_armC*.json` /
`noise_sensitivity_leakfree*.json`, or the already-pooled points in
`dose_response_feat_oob_leakfree.json`; to align strictly with
bias/drift/gain's shared levels {0.1,0.5,1,2,5}% FS, this pulls the arm-C
portion directly from the raw arm-C sweep file, not the main SNR arm's
points). For Transformer, uses its own arm-C sweep
(`t2_transformer_armC_sweep_leakfree.json`, which only has arm C anyway).
"""
import os
import json

import numpy as np

import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
PCT_LEVELS = [0.1, 0.5, 1, 2, 5]


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def points_from_block(block, method):
    return sorted([(block['feat_oob'][k], picp_of(block[method][k], method)) for k in block['feat_oob']],
                  key=lambda p: p[0])


def interp(sorted_pts, x):
    xs = [p[0] for p in sorted_pts]; ys = [p[1] for p in sorted_pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = 0 if xs[i + 1] == xs[i] else (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])
    return ys[-1]


def lstm_gaussian_armc_points(ds, method):
    if ds == 'FD003':
        with open(os.path.join(T2.RESULTS_DIR, 'noise_sensitivity_leakfree_armC_FD003.json')) as f:
            block = json.load(f)['FD003']
    else:
        with open(os.path.join(T2.RESULTS_DIR, 'noise_sensitivity_leakfree_armC.json')) as f:
            block = json.load(f)[ds]
    return points_from_block(block, method)


def transformer_gaussian_armc_points(ds, method):
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_armC_sweep_leakfree.json')) as f:
        block = json.load(f)[ds]
    return points_from_block(block, method)


if __name__ == '__main__':
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_degradation_sweep_leakfree.json')) as f:
        degr = json.load(f)

    grid = [str(p) for p in PCT_LEVELS]
    summary = {}
    for backbone in ['LSTM', 'Transformer']:
        summary[backbone] = {}
        for ds in DATASETS:
            summary[backbone][ds] = {}
            print(f"\n{'=' * 15} {backbone} / {ds} {'=' * 15}")
            for method in METHODS:
                gaussian = (transformer_gaussian_armc_points(ds, method) if backbone == 'Transformer'
                            else lstm_gaussian_armc_points(ds, method))
                bias_pts = points_from_block(degr[backbone]['bias'][ds], method)
                drift_pts = points_from_block(degr[backbone]['drift'][ds], method)
                gain_pts = points_from_block(degr[backbone]['gain'][ds], method)

                # common comparison grid: use the Gaussian-noise (arm C) feat_oob values as
                # reference points, interpolating the other three curves at the same feat_oob values
                ref_fo = [p[0] for p in gaussian]
                gaussian_picp = [p[1] for p in gaussian]
                bias_picp = [interp(bias_pts, x) for x in ref_fo]
                drift_picp = [interp(drift_pts, x) for x in ref_fo]
                gain_picp = [interp(gain_pts, x) for x in ref_fo]

                diffs = {
                    'bias_vs_gaussian': np.array(bias_picp) - np.array(gaussian_picp),
                    'drift_vs_gaussian': np.array(drift_picp) - np.array(gaussian_picp),
                    'gain_vs_gaussian': np.array(gain_picp) - np.array(gaussian_picp),
                }
                max_abs_dev = float(max(np.max(np.abs(v)) for v in diffs.values()))
                mean_abs_dev = float(np.mean([np.mean(np.abs(v)) for v in diffs.values()]))
                summary[backbone][ds][method] = {
                    'ref_feat_oob': ref_fo,
                    'gaussian_picp': gaussian_picp, 'bias_picp': bias_picp,
                    'drift_picp': drift_picp, 'gain_picp': gain_picp,
                    'max_abs_deviation_picp': max_abs_dev, 'mean_abs_deviation_picp': mean_abs_dev,
                }
                print(f"  {method:16}: max|Δpicp| across degradation types = {max_abs_dev:.3f}  "
                      f"mean|Δpicp| = {mean_abs_dev:.3f}")

    overall_max = max(summary[b][d][m]['max_abs_deviation_picp']
                       for b in summary for d in summary[b] for m in summary[b][d])
    overall_mean = float(np.mean([summary[b][d][m]['mean_abs_deviation_picp']
                                   for b in summary for d in summary[b] for m in summary[b][d]]))
    print(f"\n{'=' * 60}\nOverall across all (backbone,dataset,method): "
          f"max|Δpicp|={overall_max:.3f}  mean|Δpicp|={overall_mean:.3f}")
    print("Dose-response curves are NOT treated as overlapping if max|Δpicp| is large relative to "
          "the PICP range of interest (~0.10-0.90); reported as measured, no curve-fitting applied.")

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_partB_dose_response_overlap_summary.json')
    with open(out_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
