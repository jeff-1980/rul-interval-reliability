"""
Maintenance decision recomputation ("terminal-instant decision stress
test", not "time to first maintenance").

1. Lower bound unified to a one-sided 95%:
   - Gaussian-family mechanisms (NLL/MSE-fixed/MC-Dropout/Ensemble):
     mu - 1.645*sigma is already a one-sided 95% normal quantile, unchanged.
   - CP-norm: previously used a two-sided alpha=0.10 conformal quantile
     (q_norm, symmetric interval [mu-q*sigma, mu+q*sigma] covering 90% both
     sides). Now uses a one-sided alpha=0.05 conformal lower bound instead:
     the nonconformity score is the **signed** s=(mu-y)/sigma (not absolute
     value), lower bound = mu - q_lower*sigma, q_lower =
     ceil((n+1)(1-0.05))/n quantile (of s, not |s|), recomputed on the
     calibration set (same calib_units as Split-CP).

2. Terminology and quantities:
   - "missed failure" renamed to "unrecognised maintenance need within
     lead time", and reports a conditional miss rate = (true RUL<=L AND
     not triggered) / (engines with true RUL<=L) (denominator restricted
     to engines actually at risk, not the whole fleet).
   - "wasted life" split into two columns: "true RUL at trigger" (mean
     true RUL when triggered) and "excess over L" (= true RUL - L, the
     same quantity computed all along, definition unchanged, just now
     explicitly noted as "counted under immediate maintenance" -- a
     trigger is treated as the engine going out of service at that
     instant, and excess is the life wasted by that decision).

3. Full cost table: 5 mechanisms x 2 backbones x 4 datasets x 4 conditions
   (clean/noise1%/bias5%/drift5%) x L in {10,20,30} x cost ratio in
   {5,20,100}, reporting cost, ranking, and bootstrap intervals for
   pairwise cross-mechanism differences (resampled by engine, n=2000),
   exhaustively checking whether rankings reverse across cost ratios.

4. Fully renamed to "terminal-instant decision stress test".

Inference only, no retraining. Reuses maintenance_decision_two_sided.py's
perturbation/loading logic.
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import sweep_engine as E
from interval_score import get_cp_q
from maintenance_decision_two_sided import calib_sigma_fixed, get_raw_test_window, sequences_for_units

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
ATTR_DECISION_DIR = os.path.join(RESULTS_DIR, 'intermediate', 'attribution_and_decision')
os.makedirs(ATTR_DECISION_DIR, exist_ok=True)

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']
METHODS = ['NLL', 'MSE_fixed', 'MC_Dropout_fixed', 'Deep_Ensemble', 'CP_norm']
CONDITIONS = ['clean', 'gaussian1pct', 'drift5pct', 'bias5pct']
L_LEVELS = [10, 20, 30]
COST_RATIOS = [5, 20, 100]
N_TRIALS = V4.N_TRIALS
ALPHA_ONESIDED = 0.05
N_BOOT = 2000


def conformal_quantile_signed(scores_signed, alpha, n):
    """One-sided conformal quantile: the signed score s=(mu-y)/sigma at the
    ceil((n+1)(1-alpha))/n quantile (not absolute value), lower bound =
    mu - q*sigma."""
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    level = k / n
    return float(np.quantile(scores_signed, level, method='higher'))


def onesided_cp_q(ds, backbone, seed, canon, device):
    """On calib_units, recompute the one-sided 95% (alpha=0.05) conformal
    lower-bound quantile q_lower from the NLL checkpoint's mu/sigma."""
    fit_units = canon[ds][str(seed)]['fit_units']
    calib_units = canon[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_calib, y_calib_raw = sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)], feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
    mu_calib, ls_calib = E.infer_nll(nll_model, X_calib_t)
    sigma_calib = np.exp(ls_calib) * 125.0
    s_signed = (mu_calib - y_calib) / np.clip(sigma_calib, 1e-6, None)
    q_lower = conformal_quantile_signed(s_signed, ALPHA_ONESIDED, len(y_calib))
    del nll_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return q_lower


def cost_from_indicators(unrecognised, premature, ratio):
    return ratio * unrecognised.mean() + 1.0 * premature.mean()


def bootstrap_pairwise_ci(unrecog_a, prem_a, unrecog_b, prem_b, ratio, n_boot=N_BOOT, rng=None):
    """unrecog_*/prem_*: (n_cells, n_engines) boolean. Resamples by engine
    (columns), keeping the cells structure fixed, recomputing the
    bootstrap distribution of cost_a-cost_b. Returns
    (mean_diff, ci_lo, ci_hi). Positive = a costs more (worse) than b."""
    n_engines = unrecog_a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n_engines, n_engines)
        ca = cost_from_indicators(unrecog_a[:, idx], prem_a[:, idx], ratio)
        cb = cost_from_indicators(unrecog_b[:, idx], prem_b[:, idx], ratio)
        diffs[i] = ca - cb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    rng_global = np.random.RandomState(42)

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        canon = json.load(f)

    result = {}
    for backbone in BACKBONES:
        result[backbone] = {}
        for ds in DATASETS:
            print(f"\n{'=' * 20} {backbone} / {ds} {'=' * 20}")
            train_df_raw_probe, _, _, feat_cols_probe, _ = V4.load_raw_train_test_and_scaler(ds)
            full_scale = V4.fit_fullscale_range(train_df_raw_probe, feat_cols_probe)
            full_scale = V4.sensor_only_scale(feat_cols_probe, full_scale)

            scalers_by_seed = {}
            aleatory_var_by_seed = {}
            mc_model_by_seed = {}
            nll_model_by_seed = {}
            cp_q_onesided_by_seed = {}
            for seed in C.SEEDS:
                fit_units = canon[ds][str(seed)]['fit_units']
                _, _, _, _, scaler = V4.load_raw_train_test_and_scaler_leakfree(ds, fit_units)
                scalers_by_seed[seed] = scaler
                av, mc_model = calib_sigma_fixed(ds, backbone, seed, canon, device)
                aleatory_var_by_seed[seed] = av
                mc_model_by_seed[seed] = mc_model
                nll_model_by_seed[seed] = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
                cp_q_onesided_by_seed[seed] = onesided_cp_q(ds, backbone, seed, canon, device)

            result[backbone][ds] = {}
            # per-engine indicator arrays, keyed by (condition, method, L) -> dict with 'triggered','y'
            engine_level_data = {}

            for condition in CONDITIONS:
                trials = range(N_TRIALS) if condition in ('gaussian1pct', 'bias5pct') else [0]
                nll_mu_by_seed_trial = {}
                nll_sigma_by_seed_trial = {}
                lower_bounds = {m: [] for m in METHODS}  # list over (seed,trial) cells of per-engine lower-bound arrays
                y_cells = []

                for trial in trials:
                    X_raw, y_ref, feat_cols, test_df_raw, true_ruls = get_raw_test_window(
                        ds, condition, trial, full_scale, None)
                    y_cells.append(y_ref)

                    for seed in C.SEEDS:
                        scaler = scalers_by_seed[seed]
                        X_scaled = V4.scale_raw_windows(X_raw, scaler)
                        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

                        mu_n, ls_n = E.infer_nll(nll_model_by_seed[seed], X_t)
                        sigma_n = np.exp(ls_n) * 125.0
                        nll_mu_by_seed_trial[(seed, trial)] = mu_n
                        nll_sigma_by_seed_trial[(seed, trial)] = sigma_n

                        aleatory_var = aleatory_var_by_seed[seed]
                        mc_seed = C.stable_seed(ds, backbone, condition, trial, seed, 'mc_dropout')
                        mu_mc, sigma_mc = E.infer_mc_dropout(mc_model_by_seed[seed], X_t, T=50,
                                                              aleatory_var=aleatory_var, seed=mc_seed)
                        mu_mse, sigma_mse = E.infer_mse_fixed(mc_model_by_seed[seed], X_t,
                                                               sigma_fixed=float(np.sqrt(aleatory_var)))
                        q_lower = cp_q_onesided_by_seed[seed]

                        lower_bounds['NLL'].append(mu_n - C.Z_SCORE * sigma_n)
                        lower_bounds['MSE_fixed'].append(mu_mse - C.Z_SCORE * sigma_mse)
                        lower_bounds['MC_Dropout_fixed'].append(mu_mc - C.Z_SCORE * sigma_mc)
                        lower_bounds['CP_norm'].append(mu_n - q_lower * sigma_n)

                # Deep_Ensemble: combine per (seed,trial)-> per trial across 5 seeds
                for ti, trial in enumerate(trials):
                    mu_mem = np.stack([nll_mu_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    sigma_mem = np.stack([nll_sigma_by_seed_trial[(s, trial)] for s in C.SEEDS])
                    mu_ens = mu_mem.mean(0)
                    sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
                    sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
                    mu_ens = np.clip(mu_ens, 0, 125)
                    lower_bounds.setdefault('Deep_Ensemble', []).append(mu_ens - C.Z_SCORE * sigma_ens)

                # pool: for NLL/MSE/MC/CP there are len(trials)*5 cells; for Ensemble there are len(trials) cells
                n_engines = len(y_cells[0])
                for m in METHODS:
                    lb_stack = np.stack(lower_bounds[m])  # (n_cells, n_engines)
                    y_stack = np.tile(np.array(y_cells), (lb_stack.shape[0] // len(y_cells), 1)) \
                        if m != 'Deep_Ensemble' else np.array(y_cells)
                    engine_level_data[(condition, m)] = {'lower_bounds': lb_stack, 'y': y_stack}

            # ---- compute rates/costs/bootstrap per (condition, method, L) ----
            result[backbone][ds] = {}
            indicator_cache = {}  # (condition, L, method) -> (unrecognised, premature) boolean arrays
            for condition in CONDITIONS:
                result[backbone][ds][condition] = {}
                for m in METHODS:
                    d = engine_level_data[(condition, m)]
                    lb = d['lower_bounds']; y = d['y']
                    result[backbone][ds][condition][m] = {}
                    for L in L_LEVELS:
                        triggered = lb <= L  # (n_cells, n_engines)
                        at_risk = y <= L
                        recognised = triggered & at_risk
                        unrecognised = at_risk & (~triggered)
                        premature = triggered & (y > L + 20)
                        indicator_cache[(condition, L, m)] = (unrecognised, premature)

                        n_at_risk_total = at_risk.sum()
                        conditional_unrecognised_rate = float(unrecognised.sum() / n_at_risk_total) if n_at_risk_total > 0 else None
                        overall_unrecognised_rate = float(unrecognised.mean())
                        premature_rate = float(premature.mean())
                        if premature.any():
                            true_rul_at_trigger = float(y[premature].mean())
                            excess_over_L = float((y[premature] - L).mean())
                        else:
                            true_rul_at_trigger = None
                            excess_over_L = None

                        costs = {str(r): r * overall_unrecognised_rate + 1.0 * premature_rate for r in COST_RATIOS}
                        result[backbone][ds][condition][m][str(L)] = {
                            'overall_unrecognised_rate': overall_unrecognised_rate,
                            'conditional_unrecognised_rate': conditional_unrecognised_rate,
                            'n_at_risk': int(n_at_risk_total),
                            'premature_rate': premature_rate,
                            'true_rul_at_trigger': true_rul_at_trigger,
                            'excess_over_L': excess_over_L,
                            'cost_by_ratio': costs,
                        }
                print(f"  [{condition}] L=20 unrecognised(overall): " +
                      "  ".join(f"{m}={result[backbone][ds][condition][m]['20']['overall_unrecognised_rate']:.3f}"
                                 for m in METHODS))

            # ---- exhaustive rank-reversal check across ratios, for every (condition,L) ----
            reversal_log = []
            for condition in CONDITIONS:
                for L in L_LEVELS:
                    rankings = {}
                    for r in COST_RATIOS:
                        costs = {m: result[backbone][ds][condition][m][str(L)]['cost_by_ratio'][str(r)] for m in METHODS}
                        rankings[r] = tuple(sorted(costs, key=lambda m: costs[m]))
                    if len(set(rankings.values())) > 1:
                        reversal_log.append({'condition': condition, 'L': L, 'rankings_by_ratio':
                                              {str(r): list(rankings[r]) for r in COST_RATIOS}})
            result[backbone][ds]['_rank_reversals'] = reversal_log
            print(f"  Rank reversals found: {len(reversal_log)} / {len(CONDITIONS)*len(L_LEVELS)} (condition,L) groups")

            # ---- bootstrap CI: top-ranked method vs each other, per (condition, L, ratio) ----
            boot_rng = np.random.RandomState(123)
            bootstrap_results = {}
            for condition in CONDITIONS:
                for L in L_LEVELS:
                    for r in COST_RATIOS:
                        costs = {m: result[backbone][ds][condition][m][str(L)]['cost_by_ratio'][str(r)] for m in METHODS}
                        top = min(costs, key=lambda m: costs[m])
                        unrecog_top, prem_top = indicator_cache[(condition, L, top)]
                        for m in METHODS:
                            if m == top:
                                continue
                            unrecog_m, prem_m = indicator_cache[(condition, L, m)]
                            mean_diff, lo, hi = bootstrap_pairwise_ci(unrecog_m, prem_m, unrecog_top, prem_top,
                                                                       r, rng=boot_rng)
                            key = f"{condition}|L={L}|ratio={r}|{m}_minus_{top}"
                            bootstrap_results[key] = {'mean_diff': mean_diff, 'ci95_lo': lo, 'ci95_hi': hi,
                                                       'significant': lo > 0 or hi < 0}
            result[backbone][ds]['_bootstrap_top_vs_others'] = bootstrap_results
            n_sig = sum(1 for v in bootstrap_results.values() if v['significant'])
            print(f"  Bootstrap: {n_sig}/{len(bootstrap_results)} top-vs-other comparisons significant at 95% CI")

            for seed in C.SEEDS:
                del mc_model_by_seed[seed], nll_model_by_seed[seed]
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    out_path = os.path.join(ATTR_DECISION_DIR, 'C_maintenance_full_onesided.json')
    with open(out_path, 'w') as fp:
        json.dump(result, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")

    total_reversals = sum(len(result[b][d]['_rank_reversals']) for b in BACKBONES for d in DATASETS)
    total_groups = len(BACKBONES) * len(DATASETS) * len(CONDITIONS) * len(L_LEVELS)
    print(f"\nTOTAL rank reversals across ALL {total_groups} (backbone,dataset,condition,L) groups: {total_reversals}")
