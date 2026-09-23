"""
Part B supplement: the absolute/relative half-life f_oob for each of the
three degradation types (bias/drift/gain), 2 backbones x 4 datasets x 5
mechanisms. Previously only a "how much does PICP deviate compared to the
Gaussian-noise arm C" overlap metric was computed, without giving each
degradation type its own half-life point -- this script produces that
table.

Each degradation type uses only its own 5 levels (not pooled across
types), with logic matching mechanism_transformer.py's compute_half_life
(abs threshold 0.80, rel threshold picp_clean-0.10).
"""
import os
import json

import transformer_common as T2

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['MSE_fixed', 'NLL', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
DEGRADATIONS = ['bias', 'drift', 'gain']
BACKBONES = ['LSTM', 'Transformer']


def picp_of(cell, method):
    return cell['picp']['mean'] if method == 'Deep_Ensemble' else cell['picp']['grand_mean']


def crossover(pts, threshold):
    pts = sorted(pts, key=lambda p: p[0])
    for i in range(len(pts) - 1):
        fo0, p0 = pts[i][0], pts[i][1]
        fo1, p1 = pts[i + 1][0], pts[i + 1][1]
        if p0 >= threshold and p1 < threshold:
            frac = 0 if p1 == p0 else (threshold - p0) / (p1 - p0)
            return fo0 + frac * (fo1 - fo0)
    if pts and pts[0][1] < threshold:
        return pts[0][0]
    return None


def half_life_for_block(block):
    out = {}
    for method in METHODS:
        pts = sorted([(block['feat_oob'][k], picp_of(block[method][k], method)) for k in block['feat_oob']],
                     key=lambda p: p[0])
        picp_clean = pts[0][1]
        out[method] = {'picp_clean': picp_clean, 'abs_co': crossover(pts, 0.80),
                        'rel_co': crossover(pts, picp_clean - 0.10)}
    return out


if __name__ == '__main__':
    with open(os.path.join(T2.TRANSFORMER_DIR, 't2_degradation_sweep_leakfree.json')) as f:
        degr = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for degradation in DEGRADATIONS:
            result[backbone][degradation] = {}
            for ds in DATASETS:
                result[backbone][degradation][ds] = half_life_for_block(degr[backbone][degradation][ds])

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_degradation_half_life_feat_oob.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"Saved -> {out_path}")

    for backbone in BACKBONES:
        for degradation in DEGRADATIONS:
            print(f"\n=== {backbone} / {degradation} (rel_co) ===")
            for ds in DATASETS:
                row = result[backbone][degradation][ds]
                print(f"  {ds}: " + "  ".join(f"{m}={row[m]['rel_co']}" for m in METHODS))
