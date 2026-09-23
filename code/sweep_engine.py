"""
Generic perturbation-sweep engine, matching run_sweep_noise_lstm_fd003.py's
run_arm() protocol (5-seed models x N_TRIALS shared noise stream, five
mechanisms NLL/MC-Dropout/Ensemble/CP-norm/MSE-fixed, plus NLL/CP-norm
frozen-sigma counterfactuals and feat_oob logging), generalised along two
axes:
  1. backbone in {LSTM, Transformer} -- model loading goes through
     transformer_common's checkpoint dispatch (Transformer's CP-norm reuses
     the NLL checkpoint, see that file's header).
  2. Perturbation type -- two call patterns:
     (a) run_df_perturb_sweep: perturbation injected once over the whole
         raw df (noise/bias/gain all fall here, with a unified injection
         signature inject_fn(test_df_raw, feat_cols, level, rng,
         full_scale) -> raw_noisy).
     (b) run_drift_sweep: drift needs a position-dependent ramp inside each
         window, so the raw df must be windowed (unscaled) before
         injection; see noise_injection.extract_raw_windows /
         inject_drift_fixed_pct_windows.

LSTM side reuses existing checkpoints (no retraining); Transformer side
reuses checkpoints from T2 training. CP-norm's q / q_by_level and
MC-Dropout's aleatory_var are read from each backbone's existing result
files (adapter: load_mc_cp_for_seed).
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2

CONF_LEVELS = np.arange(0.05, 1.00, 0.05)
Z_SCORE = 1.645
N_TRIALS = V4.N_TRIALS

with open(os.path.join(T2.PROJ_DIR, 'results', 'canonical_splits.json')) as f:
    CANON = json.load(f)

_MC_CP_CACHE = {}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_mc_cp_for_seed(ds, backbone):
    """Returns (mc_by_seed, cp_by_seed), two dict[seed_str] -> record,
    unifying the differing source formats across LSTM (FD001/2/4's
    dataset-keyed mcdropout_fixed_leakfree.json + split_cp_leakfree.json;
    FD003's flat-list stepFD003_*_leakfree_results.json) and Transformer's
    dataset-keyed leakfree_t2/t2_transformer_*_results.json."""
    key = (ds, backbone)
    if key in _MC_CP_CACHE:
        return _MC_CP_CACHE[key]

    if backbone == 'Transformer':
        mc_all = _load_json(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json'))
        cp_all = _load_json(os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_splitcp_leakfree_results.json'))
        mc_list, cp_list = mc_all[ds], cp_all[ds]
    else:  # LSTM
        if ds == 'FD003':
            mc_list = _load_json(os.path.join(T2.RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json'))
            cp_list = _load_json(os.path.join(T2.RESULTS_DIR, 'stepFD003_splitcp_leakfree_results.json'))
        else:
            mc_all = _load_json(os.path.join(T2.RESULTS_DIR, 'mcdropout_fixed_leakfree.json'))
            cp_all = _load_json(os.path.join(T2.RESULTS_DIR, 'split_cp_leakfree.json'))
            mc_list, cp_list = mc_all[ds], cp_all[ds]

    mc_by_seed = {str(r['seed']): r for r in mc_list}
    cp_by_seed = {str(r['seed']): r for r in cp_list}
    _MC_CP_CACHE[key] = (mc_by_seed, cp_by_seed)
    return mc_by_seed, cp_by_seed


_CALIB_ALEATORY_CACHE = {}


def _sequences_for_units(df, feature_cols, unit_set):
    X_list, y_list = [], []
    for unit in sorted(unit_set):
        unit_data = df[df['unit_nr'] == unit][feature_cols].values
        rul_arr = df[df['unit_nr'] == unit]['RUL'].values
        for i in range(len(unit_data) - C.SEQUENCE_LENGTH):
            X_list.append(unit_data[i: i + C.SEQUENCE_LENGTH])
            y_list.append(rul_arr[i + C.SEQUENCE_LENGTH])
    return np.array(X_list), np.array(y_list)


def calib_aleatory_var(ds, backbone, seed, device, mc_model=None):
    """Fair-calibration fix: MSE-fixed/MC-Dropout's aleatory_var across every
    degradation/noise sweep used to come from load_mc_cp_for_seed()'s
    result files -- training-residual variance, not the same thing as
    Table II's fair-calibration version (calibration-set residual variance,
    same definition as maintenance_decision_two_sided.calib_sigma_fixed /
    fair_calibration_main_table). This recomputes it on calib_units
    directly, used throughout every degradation sweep instead of reading
    the training-residual file. Cached by (ds,backbone,seed) to avoid
    recomputing across levels within the same sweep."""
    key = (ds, backbone, seed)
    if key in _CALIB_ALEATORY_CACHE:
        return _CALIB_ALEATORY_CACHE[key]
    fit_units = CANON[ds][str(seed)]['fit_units']
    calib_units = CANON[ds][str(seed)]['calib_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_calib, y_calib_raw = _sequences_for_units(train_df[train_df['unit_nr'].isin(calib_units)],
                                                 feat_cols, calib_units)
    y_calib = np.clip(y_calib_raw, 0, C.MAX_RUL)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32).to(device)
    owns_model = mc_model is None
    if owns_model:
        mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    mc_model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X_calib_t.shape[0], 8192):
            mus.append(mc_model(X_calib_t[i:i + 8192]).cpu().numpy().flatten())
    yhat_calib = np.clip(np.concatenate(mus) * 125.0, 0, 125)
    aleatory_var = float(np.var(y_calib - yhat_calib, ddof=1))
    if owns_model:
        del mc_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    _CALIB_ALEATORY_CACHE[key] = aleatory_var
    return aleatory_var


def load_models_for_seed(ds, backbone, seed, device):
    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
    mc_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    if backbone == 'LSTM':
        cp_ckpt = os.path.join(T2.PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_SplitCP_seed{seed}.pt")
        cp_model = C.load_checkpoint_model(cp_ckpt, device)
    else:
        cp_model = nll_model  # Transformer: CP reuses the NLL checkpoint, see file header
    return nll_model, mc_model, cp_model


def picp_mpiw(y_true, mu, sigma, z=Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi))), float(np.mean(hi - lo))


def metrics_basic(y_true, mu, sigma):
    rmse, score = C.rmse_score(y_true, mu)
    picp, mpiw = picp_mpiw(y_true, mu, sigma)
    ece = C.compute_ece(mu, sigma, y_true, CONF_LEVELS)
    return {'rmse': rmse, 'score': score, 'picp': picp, 'mpiw': mpiw, 'ece': ece, 'sigma_mean': float(sigma.mean())}


def grand_and_marginal_stats(cells, n_trials=N_TRIALS):
    keys = ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']
    out = {}
    arr = {k: np.array([[cells[i][t][k] for t in range(n_trials)] for i in range(5)]) for k in keys}
    for k in keys:
        a = arr[k]
        model_marg = a.mean(axis=1); trial_marg = a.mean(axis=0)
        out[k] = {'grand_mean': float(a.mean()), 'grand_std': float(a.std(ddof=1)),
                   'model_marginal_mean': float(model_marg.mean()), 'model_marginal_std': float(model_marg.std(ddof=1)),
                   'trial_marginal_mean': float(trial_marg.mean()), 'trial_marginal_std': float(trial_marg.std(ddof=1))}
    return out


def infer_nll(model, X_t, batch=8192):
    mus, ls = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            m, s = model(X_t[i:i + batch])
            mus.append(m.cpu().numpy().flatten()); ls.append(s.cpu().numpy().flatten())
    return np.concatenate(mus) * 125.0, np.concatenate(ls)


def infer_mc_dropout(mc_model, X_t, T, aleatory_var, batch=4096, seed=None):
    """seed: T=50 dropout sampling (mc_model.train()) is live randomness,
    previously never seeded, so it wasn't reproducible across processes.
    Passing a deterministic seed (derived via C.stable_seed) and calling
    torch.manual_seed before sampling makes it reproducible going forward
    -- this is a "new" deterministic sample, not the same as any
    pre-existing result's unrecorded sampling, so those older
    MC_Dropout_fixed numbers can't be recovered bit-for-bit, only future
    reruns are mutually reproducible."""
    if seed is not None:
        torch.manual_seed(seed)
    mc_model.train()
    all_samples = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            xb = X_t[i:i + batch]
            samples = [mc_model(xb).cpu().numpy().flatten() * 125.0 for _ in range(T)]
            all_samples.append(np.stack(samples))
    samples = np.concatenate(all_samples, axis=1)
    mu = np.clip(samples.mean(0), 0, 125)
    sigma = np.sqrt(aleatory_var + samples.var(0))
    return mu, sigma


def infer_mse_fixed(mc_model, X_t, sigma_fixed, batch=8192):
    mc_model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            mus.append(mc_model(X_t[i:i + batch]).cpu().numpy().flatten())
    mu = np.clip(np.concatenate(mus) * 125.0, 0, 125)
    sigma = np.full_like(mu, sigma_fixed)
    return mu, sigma


def _empty_out():
    return {'NLL': {}, 'MC_Dropout_fixed': {}, 'Deep_Ensemble': {}, 'CP_norm': {}, 'MSE_fixed': {}, 'feat_oob': {},
            'NLL_frozen_sigma': {}, 'CP_norm_frozen_sigma': {}}


def _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                     device, aleatory_var_calib_by_seed):
    nll_cells = [[None] * N_TRIALS for _ in range(5)]
    mc_cells = [[None] * N_TRIALS for _ in range(5)]
    cp_cells = [[None] * N_TRIALS for _ in range(5)]
    mse_cells = [[None] * N_TRIALS for _ in range(5)]
    nll_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
    cp_frozen_cells = [[None] * N_TRIALS for _ in range(5)]
    nll_mu_grid = [[None] * N_TRIALS for _ in range(5)]
    nll_sigma_grid = [[None] * N_TRIALS for _ in range(5)]

    _, cp_by_seed = load_mc_cp_for_seed(ds, backbone)

    for i, seed in enumerate(C.SEEDS):
        nll_model, mc_model, cp_model = models_by_seed[seed]
        # Fair-calibration fix: aleatory_var now uses calibration-set
        # residual variance (matching Table II), not the training-residual
        # result file.
        aleatory_var = aleatory_var_calib_by_seed[seed]
        q_norm = cp_by_seed[str(seed)]['cp_norm']['q']
        q_norm_by_level = cp_by_seed[str(seed)]['cp_norm']['q_by_level']

        for t in range(N_TRIALS):
            X_t = trial_X[t][seed]

            mu_n, ls_n = infer_nll(nll_model, X_t)
            sigma_n = np.exp(ls_n) * 125.0
            nll_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n)
            nll_mu_grid[i][t] = mu_n; nll_sigma_grid[i][t] = sigma_n
            sigma_n_frozen = clean_sigma_nll[seed]
            nll_frozen_cells[i][t] = metrics_basic(trial_y, mu_n, sigma_n_frozen)

            mc_seed = C.stable_seed(ds, backbone, level_key, t, seed, 'mc_dropout')
            mu_m, sigma_m = infer_mc_dropout(mc_model, X_t, T=50, aleatory_var=aleatory_var, seed=mc_seed)
            mc_cells[i][t] = metrics_basic(trial_y, mu_m, sigma_m)

            mu_mse, sigma_mse = infer_mse_fixed(mc_model, X_t, sigma_fixed=float(np.sqrt(aleatory_var)))
            mse_cells[i][t] = metrics_basic(trial_y, mu_mse, sigma_mse)

            mu_c, ls_c = infer_nll(cp_model, X_t)
            sigma_c = np.exp(ls_c) * 125.0
            picp_c, mpiw_c = picp_mpiw(trial_y, mu_c, sigma_c, z=q_norm)
            rmse_c, score_c = C.rmse_score(trial_y, mu_c)
            emp = []
            for p in CONF_LEVELS:
                q = q_norm_by_level[f"{p:.2f}"]
                lo, hi = mu_c - q * sigma_c, mu_c + q * sigma_c
                emp.append(np.mean((trial_y >= lo) & (trial_y <= hi)))
            ece_c = float(np.mean(np.abs(np.array(emp) - CONF_LEVELS)))
            cp_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_c, 'mpiw': mpiw_c,
                               'ece': ece_c, 'sigma_mean': float(sigma_c.mean())}
            sigma_c_frozen = clean_sigma_cp[seed]
            picp_cf, mpiw_cf = picp_mpiw(trial_y, mu_c, sigma_c_frozen, z=q_norm)
            cp_frozen_cells[i][t] = {'rmse': rmse_c, 'score': score_c, 'picp': picp_cf, 'mpiw': mpiw_cf,
                                      'ece': float('nan'), 'sigma_mean': float(sigma_c_frozen.mean())}

    out['NLL'][level_key] = grand_and_marginal_stats(nll_cells)
    out['MC_Dropout_fixed'][level_key] = grand_and_marginal_stats(mc_cells)
    out['CP_norm'][level_key] = grand_and_marginal_stats(cp_cells)
    out['MSE_fixed'][level_key] = grand_and_marginal_stats(mse_cells)
    out['NLL_frozen_sigma'][level_key] = grand_and_marginal_stats(nll_frozen_cells)
    out['CP_norm_frozen_sigma'][level_key] = grand_and_marginal_stats(cp_frozen_cells)

    ens_trial_metrics = []
    for t in range(N_TRIALS):
        mu_mem = np.stack([nll_mu_grid[i][t] for i in range(5)])
        sigma_mem = np.stack([nll_sigma_grid[i][t] for i in range(5)])
        mu_ens = mu_mem.mean(0)
        sigma2_ens = (sigma_mem ** 2 + mu_mem ** 2).mean(0) - mu_ens ** 2
        sigma_ens = np.sqrt(np.clip(sigma2_ens, 0, None))
        mu_ens = np.clip(mu_ens, 0, 125)
        ens_trial_metrics.append(metrics_basic(trial_y, mu_ens, sigma_ens))
    ens_arr = {k: np.array([m[k] for m in ens_trial_metrics]) for k in ['rmse', 'picp', 'mpiw', 'ece', 'sigma_mean']}
    out['Deep_Ensemble'][level_key] = {k: {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'n_trials': N_TRIALS}
                                        for k, v in ens_arr.items()}

    print(f"    {level_key:>6}: feat_oob={out['feat_oob'][level_key]:.4f}  "
          f"NLL_PICP={out['NLL'][level_key]['picp']['grand_mean']:.3f}  "
          f"MC_PICP={out['MC_Dropout_fixed'][level_key]['picp']['grand_mean']:.3f}  "
          f"Ens_PICP={out['Deep_Ensemble'][level_key]['picp']['mean']:.3f}  "
          f"CP_PICP={out['CP_norm'][level_key]['picp']['grand_mean']:.3f}  "
          f"MSE_PICP={out['MSE_fixed'][level_key]['picp']['grand_mean']:.3f}")


def _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls, scalers_by_seed, models_by_seed):
    clean_sigma_nll, clean_sigma_cp = {}, {}
    for seed in C.SEEDS:
        nll_model, mc_model, cp_model = models_by_seed[seed]
        scaler = scalers_by_seed[seed]
        scaled_clean = scaler.transform(test_df_raw[feat_cols].values.astype(np.float64))
        df_clean = test_df_raw.copy(); df_clean[feat_cols] = scaled_clean
        X_clean, _ = C.create_sequences(df_clean, feat_cols, mode='test', true_ruls=true_ruls)
        X_clean_t = torch.tensor(X_clean, dtype=torch.float32).to(device)
        _, ls_nll = infer_nll(nll_model, X_clean_t)
        clean_sigma_nll[seed] = np.exp(ls_nll) * 125.0
        _, ls_cp = infer_nll(cp_model, X_clean_t)
        clean_sigma_cp[seed] = np.exp(ls_cp) * 125.0
    return clean_sigma_nll, clean_sigma_cp


def run_df_perturb_sweep(ds, backbone, perturb_name, inject_fn, levels, is_pct, device,
                          scalers_by_seed, full_scale=None, global_std=None):
    """levels: list of values; inject_fn(test_df_raw, feat_cols, level, rng,
    full_scale) -> raw_noisy (bias/gain/armC-gaussian all share this path,
    same signature, different inject_fn). level_key: str(level) when
    is_pct=True (e.g. armC/bias/gain's 0.1/0.5/1/2/5); 'inf' for is_pct=False
    (noise SNR levels; this function currently only serves pct-type
    perturbations, SNR-type still uses the older run_arm)."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    for level in levels:
        level_key = str(level)
        trial_X, trial_y, trial_feat_oob = {}, None, []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(ds, backbone, perturb_name, level_key, t)))
            raw_noisy = inject_fn(test_df_raw, feat_cols, level, rng, full_scale)
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                # f_oob computed on the windowed X_test actually fed to the
                # model (not the full-trajectory scaled_feat scale_and_package
                # returns), matching the drift branch and PICP's own 5x5
                # aggregation. Denominator restricted to sensor columns via
                # the single project-wide V4.feat_oob implementation.
                trial_feat_oob.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def run_snr_sweep(ds, backbone, scheme, levels, device, scalers_by_seed, global_std=None, km=None, cond_std=None):
    """Main SNR arm (Gaussian noise, per_condition or global scheme),
    matching the LSTM side's run_sweep_noise_lstm.py /
    run_sweep_noise_lstm_fd003.py main-arm protocol exactly -- added so
    that the Transformer and LSTM half-life computations are based on the
    same (main arm + arm C) pooled data across the same feat_oob range
    (arm C alone doesn't cover the <2% feat_oob region where the LSTM
    half-life mostly falls, so a main-arm-only vs. arm-C-only comparison
    would not be over the same feat_oob range). scheme='per_condition' for
    FD002/FD004 (multi-condition); scheme='global' for FD001/FD003
    (single-condition, equivalent to A_percondition degenerating to
    B_pooled)."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    for level in levels:
        level_key = 'inf' if np.isinf(level) else str(level)
        trial_X, trial_y, trial_feat_oob = {}, None, []
        for t in range(N_TRIALS):
            rng = np.random.RandomState((C.stable_seed(ds, backbone, 'mainarm', scheme, level_key, t)))
            raw_noisy = V4.inject_noise_raw(test_df_raw, feat_cols, level, rng, scheme,
                                             global_std=global_std, km=km, cond_std=cond_std)
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                df_noisy, scaled_feat = V4.scale_and_package(test_df_raw, feat_cols, raw_noisy, scaler)
                X_test, y_test = C.create_sequences(df_noisy, feat_cols, mode='test', true_ruls=true_ruls)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_test, dtype=torch.float32).to(device)
                trial_y = y_test
                # See run_df_perturb_sweep's comment on the same f_oob convention.
                trial_feat_oob.append(V4.feat_oob(X_test, V4.sensor_mask_for(feat_cols)))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def run_drift_sweep(ds, backbone, levels, device, scalers_by_seed, full_scale):
    """drift: window-level injection (each window independently re-zeroes
    its 0->k*FS ramp), via extract_raw_windows(mode='test') +
    inject_drift_fixed_pct_windows + scale_raw_windows; everything else
    (5-model x 5-trial shared stream, five mechanisms, frozen sigma,
    feat_oob) matches run_df_perturb_sweep exactly."""
    _, test_df_raw, true_ruls, feat_cols, _ = V4.load_raw_train_test_and_scaler(ds)
    out = _empty_out()

    models_by_seed = {seed: load_models_for_seed(ds, backbone, seed, device) for seed in C.SEEDS}
    aleatory_var_calib_by_seed = {seed: calib_aleatory_var(ds, backbone, seed, device, mc_model=models_by_seed[seed][1])
                                  for seed in C.SEEDS}
    clean_sigma_nll, clean_sigma_cp = _clean_sigmas(ds, backbone, device, test_df_raw, feat_cols, true_ruls,
                                                     scalers_by_seed, models_by_seed)

    X_raw_clean, y_test_fixed, _ = V4.extract_raw_windows(test_df_raw, feat_cols, true_ruls, mode='test')

    for level in levels:
        level_key = str(level)
        X_raw_drifted = V4.inject_drift_fixed_pct_windows(X_raw_clean, level, full_scale)
        trial_X, trial_y, trial_feat_oob = {}, y_test_fixed, []
        # drift is a deterministic ramp (no randomness), so all N_TRIALS
        # trials degenerate to the identical injection; the N_TRIALS axis
        # is kept only to share grand_and_marginal_stats' aggregation code
        # path with the other perturbation types (zero trial-to-trial
        # variance here is expected).
        for t in range(N_TRIALS):
            for seed in C.SEEDS:
                scaler = scalers_by_seed[seed]
                X_scaled = V4.scale_raw_windows(X_raw_drifted, scaler)
                trial_X.setdefault(t, {})[seed] = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                trial_feat_oob.append(V4.feat_oob(X_scaled, V4.sensor_mask_for(feat_cols)))
        out['feat_oob'][level_key] = float(np.mean(trial_feat_oob))
        _eval_one_level(level_key, trial_X, trial_y, ds, backbone, models_by_seed, clean_sigma_nll, clean_sigma_cp, out,
                         device, aleatory_var_calib_by_seed)

    for seed in C.SEEDS:
        for m in models_by_seed[seed]:
            del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out
