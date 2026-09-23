"""
Pure post-processing, no new inference, no PYTHONHASHSEED requirement.

attribution_bootstrap_order_averaged.py's |Delta_mu|-|Delta_sigma| bootstrap
has two known caveats (already documented honestly in that script's own
docstring):
  (1) It uses the "nearest measured grid point" C values from
      B_attribution_raw_per_seed.json, not the "exact interpolated
      crossover" that G/H use -- an operating-point mismatch.
  (2) Order A (freeze sigma first, then release mu) and order B (freeze mu
      first, then release sigma) have their mu_effect/sigma_effect averaged
      into a single avg_mu/avg_sigma before bootstrapping, which masks
      whether each order individually crosses zero robustly.

This script instead uses the C00_per_seed/C10_per_seed/C01_per_seed/
C11_per_seed fields added to G_transformer_exact_interp_attribution.json /
H_lstm_exact_interp_attribution.json (the four combinations computed per
seed, at G/H's own exact interpolated crossover point; C01's two-point new
inference is kept per seed, see those scripts' mu_sigma_pert_replicates /
per_seed_C_at_point) -- order A and order B each get their own 5 per-seed
values of |mu_effect|-|sigma_effect|, bootstrapped separately (n=2000,
resampling the 5 seeds with replacement, 95% interval), instead of
averaging the two orders together.

The old "order-averaged + nearest grid point" version is kept as-is under
the output's `_order_averaged` field (reusing
attribution_bootstrap_order_averaged's bootstrap_abs_mu_minus_abs_sigma
verbatim, still sourced from B_attribution_raw_per_seed.json), for
old-vs-new comparison -- not deleted or overwritten.

Also runs the interaction sign correction on G/H first (after
regenerating those files, the interaction field needs recomputing under
the (C11-C10)-(C01-C00) convention, reusing
attribution_bootstrap_order_averaged.fix_interaction verbatim).
"""
import os
import json

import numpy as np

from attribution_bootstrap_order_averaged import fix_interaction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
METHODS = ['NLL', 'CP_norm']
N_BOOT = 2000
RNG = np.random.RandomState(2026)


def bootstrap_5(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        boot[i] = values[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'per_seed_values': values.tolist(),
        'mean': float(values.mean()),
        'bootstrap_mean': float(boot.mean()),
        'ci95_lo': float(lo), 'ci95_hi': float(hi),
        'excludes_zero': bool(lo > 0 or hi < 0),
        'dominant_direction': 'mu' if lo > 0 else ('sigma' if hi < 0 else None),
    }


def bootstrap_by_order_from_cell(cell):
    """cell: one G/H result[ds][method] dict, with C00_per_seed/.../C11_per_seed
    already populated at the exact-interpolation crossover point."""
    c00 = np.array(cell['C00_per_seed']); c10 = np.array(cell['C10_per_seed'])
    c01 = np.array(cell['C01_per_seed']); c11 = np.array(cell['C11_per_seed'])
    orderA_mu = c10 - c00; orderA_sigma = c11 - c10
    orderB_sigma = c01 - c00; orderB_mu = c11 - c01

    diffA = np.abs(orderA_mu) - np.abs(orderA_sigma)
    diffB = np.abs(orderB_mu) - np.abs(orderB_sigma)

    rA = bootstrap_5(diffA)
    rB = bootstrap_5(diffB)
    rA['seed_order'] = cell.get('per_seed_seed_order')
    rB['seed_order'] = cell.get('per_seed_seed_order')
    return {'order_A_freeze_sigma_first': rA, 'order_B_freeze_mu_first': rB,
            'evaluation_point': 'G/H exact-interpolation crossover (crossover_feat_oob_exact)'}


if __name__ == '__main__':
    g_path = os.path.join(ATTR_DECISION_DIR, 'G_transformer_exact_interp_attribution.json')
    h_path = os.path.join(ATTR_DECISION_DIR, 'H_lstm_exact_interp_attribution.json')
    g_data = fix_interaction(g_path)
    h_data = fix_interaction(h_path)
    print("Interaction sign fixed in G and H (verified via algebraic identity check).")

    # old order-averaged version, unchanged method, nearest-grid-point data source
    with open(os.path.join(ATTR_DECISION_DIR, 'B_attribution_raw_per_seed.json')) as f:
        raw_b = json.load(f)

    def old_order_averaged(cell_b):
        c00 = np.array(cell_b['C00_per_seed']); c10 = np.array(cell_b['C10_per_seed'])
        c01 = np.array(cell_b['C01_per_seed']); c11 = np.array(cell_b['C11_per_seed'])
        orderA_mu = c10 - c00; orderA_sigma = c11 - c10
        orderB_sigma = c01 - c00; orderB_mu = c11 - c01
        avg_mu = (orderA_mu + orderB_mu) / 2.0
        avg_sigma = (orderA_sigma + orderB_sigma) / 2.0
        diff = np.abs(avg_mu) - np.abs(avg_sigma)
        r = bootstrap_5(diff)
        r['caveat'] = ("nearest-grid-point operating point (frozen_decomposition_2x2.json / "
                        "B_attribution_raw_per_seed.json), order-A/order-B averaged before "
                        "bootstrap -- NOT the same operating point or order-separated methodology "
                        "as the sibling fields in this file; kept only for before/after comparison.")
        return r

    result = {'LSTM': {}, 'Transformer': {}}
    counts = {'LSTM': {'A_excl_zero': 0, 'B_excl_zero': 0, 'total_cells': 0},
              'Transformer': {'A_excl_zero': 0, 'B_excl_zero': 0, 'total_cells': 0}}

    for backbone, g_or_h in (('Transformer', g_data), ('LSTM', h_data)):
        for ds in DATASETS:
            result[backbone][ds] = {}
            for m in METHODS:
                cell = g_or_h[ds][m]
                cell_b = raw_b[backbone][ds][m]
                if 'note' in cell or 'note' in cell_b:
                    result[backbone][ds][m] = {'note': cell.get('note') or cell_b.get('note')}
                    continue
                r = bootstrap_by_order_from_cell(cell)
                r['_order_averaged'] = old_order_averaged(cell_b)
                result[backbone][ds][m] = r
                counts[backbone]['total_cells'] += 1
                if r['order_A_freeze_sigma_first']['excludes_zero']:
                    counts[backbone]['A_excl_zero'] += 1
                if r['order_B_freeze_mu_first']['excludes_zero']:
                    counts[backbone]['B_excl_zero'] += 1
                print(f"  {backbone}/{ds}/{m}: "
                      f"A mean={r['order_A_freeze_sigma_first']['mean']:.4f} "
                      f"CI=[{r['order_A_freeze_sigma_first']['ci95_lo']:.4f},"
                      f"{r['order_A_freeze_sigma_first']['ci95_hi']:.4f}] "
                      f"excl0={r['order_A_freeze_sigma_first']['excludes_zero']}  |  "
                      f"B mean={r['order_B_freeze_mu_first']['mean']:.4f} "
                      f"CI=[{r['order_B_freeze_mu_first']['ci95_lo']:.4f},"
                      f"{r['order_B_freeze_mu_first']['ci95_hi']:.4f}] "
                      f"excl0={r['order_B_freeze_mu_first']['excludes_zero']}")

    result['_summary'] = counts
    print(f"\nCounts (CI excludes zero): {json.dumps(counts, indent=2)}")

    out_path = os.path.join(ATTR_DECISION_DIR, 'bootstrap_by_order_exact_interp.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
