"""
Diagnostic, inference-level, no retraining: three-way contrast of what
gets perturbed.

Background: every injection function in noise_injection.py perturbs
whatever feature_cols it's given uniformly -- FD002/FD004's feature_cols =
SETTING_NAMES(3) + 15 sensors, meaning every degradation experiment
(Gaussian/bias/gain/drift) on these two datasets adds "sensor noise/drift/
bias/gain" equally to the 3 operating-condition setting columns. Settings
are a commanded regime, not a sensor reading, and physically should not be
corrupted by "sensor degradation" -- this is what this diagnostic checks.

Implementation: doesn't modify any injection function itself, just zeroes
the full_scale_range (the per-channel noise/drift/bias/gain magnitude
reference) passed to them on the columns that shouldn't be perturbed --
those columns' delta/noise_std become 0, equivalent to "not perturbed",
while window shape/column order stay unchanged (model input dimension
unchanged, only some columns' values stop changing).

Three variants:
  sensors  -- perturb only the 15 sensor columns, 3 setting columns stay clean
  settings -- perturb only the 3 setting columns, 15 sensor columns stay clean
  joint    -- the current convention, all 18 columns perturbed together
              (control, equivalent to the existing pipeline)

Tests only FD002/FD004 (the only datasets with setting columns), both
backbones, NLL mechanism, 5 seeds, drift 5% and Gaussian scheme C
(armC, fixed-pct) 1%.

Decision metric (L=20) matches maintenance_decision_one_sided.py's
convention: triggered = (mu - Z_SCORE*sigma) <= L; at_risk = y<=L;
premature = triggered & (y > L+20); unrecognised = at_risk & ~triggered.
(NLL's lower bound is already mu-1.645*sigma, equivalent to a one-sided
95% lower bound, no extra CP recalibration needed.)

Read-only diagnostic; does not modify main.tex or write any tex.
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
DIAG_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'protocol_diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

DATASETS = ['FD002', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
VARIANTS = ['sensors', 'settings', 'joint']
CONDITIONS = ['drift5pct', 'gauss1pct_armC']
N_TRIALS = V4.N_TRIALS
L = 20
Z = C.Z_SCORE


def variant_scale_and_mask(feat_cols, full_scale, variant):
    is_setting = np.array([c in C.SETTING_NAMES for c in feat_cols])
    if variant == 'sensors':
        keep = ~is_setting
    elif variant == 'settings':
        keep = is_setting
    elif variant == 'joint':
        keep = np.ones(len(feat_cols), dtype=bool)
    else:
        raise ValueError(variant)
    scale = full_scale * keep.astype(np.float64)
    return scale, keep


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for ds in DATASETS:
        train_df_raw, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
        full_scale = V4.fit_fullscale_range(train_df_raw, feat_cols)
        X_raw_clean, y_ref, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')
        result[ds] = {}

        for backbone in BACKBONES:
            print(f"\n{'=' * 20} {ds} / {backbone} {'=' * 20}")
            scalers_by_seed = {}
            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                scalers_by_seed[seed] = scaler
            models_by_seed = {seed: T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                               for seed in C.SEEDS}

            result[ds][backbone] = {}
            for variant in VARIANTS:
                scale_v, keep_mask = variant_scale_and_mask(feat_cols, full_scale, variant)
                result[ds][backbone][variant] = {}

                for condition in CONDITIONS:
                    picp_cells, foob_cells, premature_cells, unrecog_cells, atrisk_n_cells = [], [], [], [], []
                    trials = [0] if condition == 'drift5pct' else range(N_TRIALS)

                    for trial in trials:
                        if condition == 'drift5pct':
                            X_raw = V4.inject_drift_fixed_pct_windows(X_raw_clean, 5.0, scale_v)
                            y_this = y_ref
                        else:
                            rng = np.random.RandomState(C.stable_seed(ds, 'r8a1', variant, condition, trial))
                            raw_noisy = V4.inject_noise_fixed_pct_raw(test_df_raw, feat_cols, 1.0, rng, scale_v)
                            df_tmp = test_df_raw.copy(); df_tmp[feat_cols] = raw_noisy
                            X_raw, y_this, _ = V4.extract_raw_windows(df_tmp, feat_cols, true_ruls, mode='test')

                        for seed in C.SEEDS:
                            X_scaled = V4.scale_raw_windows(X_raw, scalers_by_seed[seed])
                            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                            mu, ls = E.infer_nll(models_by_seed[seed], X_t)
                            sigma = np.exp(ls) * 125.0
                            lo, hi = mu - Z * sigma, mu + Z * sigma
                            picp_cells.append(float(np.mean((y_this >= lo) & (y_this <= hi))))
                            # keep_mask is this script's own "columns actually
                            # perturbed under this variant" (differs across
                            # sensors/settings/joint) -- V4.feat_oob is only
                            # the shared reduction formula, mask semantics unchanged.
                            foob_cells.append(V4.feat_oob(X_scaled, keep_mask))

                            lb = mu - Z * sigma
                            triggered = lb <= L
                            at_risk = y_this <= L
                            premature = triggered & (y_this > L + 20)
                            unrecognised = at_risk & (~triggered)
                            premature_cells.append(float(premature.mean()))
                            unrecog_cells.append(float(unrecognised.mean()))
                            atrisk_n_cells.append(int(at_risk.sum()))

                    result[ds][backbone][variant][condition] = {
                        'picp_grand_mean': float(np.mean(picp_cells)),
                        'f_oob_grand_mean': float(np.mean(foob_cells)),
                        'premature_rate_L20_grand_mean': float(np.mean(premature_cells)),
                        'unrecognised_rate_L20_grand_mean': float(np.mean(unrecog_cells)),
                        'n_at_risk_total_pooled': int(np.sum(atrisk_n_cells)),
                        'n_cells': len(picp_cells),
                    }
                    r = result[ds][backbone][variant][condition]
                    print(f"  [{variant:8s}/{condition:14s}] PICP={r['picp_grand_mean']:.4f}  "
                          f"f_oob={r['f_oob_grand_mean']*100:.4f}%  "
                          f"premature@L20={r['premature_rate_L20_grand_mean']:.4f}  "
                          f"unrecognised@L20={r['unrecognised_rate_L20_grand_mean']:.4f}")

            del models_by_seed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(DIAG_DIR, 'A1_perturbation_target_diagnostic.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    # ---- stop-condition check ----
    print("\n" + "=" * 70)
    print("STOP-CONDITION CHECK: sensors-only drift premature_rate vs joint (current)")
    print("=" * 70)
    any_trigger = False
    for ds in DATASETS:
        for backbone in BACKBONES:
            p_sensors = result[ds][backbone]['sensors']['drift5pct']['premature_rate_L20_grand_mean']
            p_joint = result[ds][backbone]['joint']['drift5pct']['premature_rate_L20_grand_mean']
            ratio = (p_sensors / p_joint) if p_joint > 0 else float('nan')
            triggers = p_joint > 0 and p_sensors < 0.5 * p_joint
            any_trigger = any_trigger or triggers
            print(f"  {ds}/{backbone}: sensors-only={p_sensors:.4f}  joint={p_joint:.4f}  "
                  f"ratio={ratio:.3f}  STOP-CONDITION-MET={triggers}")
    print(f"\nANY CELL TRIGGERS STOP CONDITION: {any_trigger}")
