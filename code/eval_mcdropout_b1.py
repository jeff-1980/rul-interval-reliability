"""
R8-B1（推理级，不重训）：重新评估 MC-Dropout/MSE 三个文件。checkpoint 权重
不变；aleatory_var 用 fit_units 残差（训练标签，不读官方RUL文件，B2不影响，
为代码简单也一并重算）。T=50/100 的 dropout 采样本身是活随机数——原始
训练脚本从未播种过（与本项目更早发现的同类问题同源），这里用
C.stable_seed 播种，保证本轮自己的两次独立重跑逐比特一致，但不assert
与本轮之前的旧基线逐比特一致（旧基线的具体采样序列本来就没有留下可
回放的随机性凭证）。

覆盖：
  mcdropout_fixed_leakfree.json               (LSTM, FD001/FD002/FD004)
  stepFD003_mcdropout_mse_leakfree_results.json (LSTM, FD003, 含 mse_row)
  t2_transformer_msemcd_leakfree_results.json  (Transformer, 全部4个数据集, 含 mse_row)
"""
import os
import json

import numpy as np
import torch

import common as C
import noise_injection as V4
import transformer_common as T2
import mc_dropout_model as S1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
MC_T_MAIN, MC_T_EXTRA = 50, 100
CONF_LEVELS = np.arange(0.05, 1.00, 0.05)


def picp_mpiw(yt, mu, sigma, z=C.Z_SCORE):
    lo = mu - z * sigma; hi = mu + z * sigma
    return float(np.mean((yt >= lo) & (yt <= hi))), float(np.mean(hi - lo))


def batched_forward(model, X_t, batch=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch):
            outs.append(model(X_t[i:i + batch]).cpu().numpy())
    return np.concatenate(outs).flatten()


def eval_one(model, ds, seed, canon, device, tag, with_mse_row):
    fit_units = canon[ds][str(seed)]['fit_units']
    train_df, test_df, true_ruls, feat_cols, scaler = C.load_and_process_leakfree(ds, fit_units)
    X_fit, y_fit = C.create_sequences(train_df[train_df['unit_nr'].isin(fit_units)], feat_cols, mode='train')
    X_fit_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    yhat_fit = np.clip(batched_forward(model, X_fit_t) * 125.0, 0, 125)
    aleatory_var = float(np.var(y_fit * 125.0 - yhat_fit, ddof=1))
    del X_fit_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    mc_seed = C.stable_seed(ds, tag, seed, 'r8b1_mcdropout_eval')
    torch.manual_seed(mc_seed)
    model.train()
    samples = []
    with torch.no_grad():
        for _ in range(MC_T_EXTRA):
            samples.append(model(X_test_t).cpu().numpy().flatten() * 125.0)
    samples = np.stack(samples)

    def eval_variant(T):
        s = samples[:T]
        mu = np.clip(s.mean(0), 0, 125)
        eps_var = s.var(0)
        sigma_sampling = np.sqrt(eps_var)
        rmse, score = C.rmse_score(y_test, mu)
        p_s, w_s = picp_mpiw(y_test, mu, sigma_sampling)
        ece_s = C.compute_ece(mu, sigma_sampling, y_test, CONF_LEVELS)
        sigma_full = np.sqrt(aleatory_var + eps_var)
        p_f, w_f = picp_mpiw(y_test, mu, sigma_full)
        ece_f = C.compute_ece(mu, sigma_full, y_test, CONF_LEVELS)
        return {'T': T, 'rmse': rmse, 'score': score,
                'sampling_only': {'picp': p_s, 'mpiw': w_s, 'ece': ece_s, 'sigma_mean': float(sigma_sampling.mean())},
                'kendall_gal_full': {'picp': p_f, 'mpiw': w_f, 'ece': ece_f, 'sigma_mean': float(sigma_full.mean()),
                                      'aleatory_var': aleatory_var, 'epistemic_var_mean': float(eps_var.mean())}}

    res_T50 = eval_variant(MC_T_MAIN)
    res_T100 = eval_variant(MC_T_EXTRA)
    out = {'seed': seed, 'T50': res_T50, 'T100': res_T100}

    if with_mse_row:
        model.eval()
        mu_mse = np.clip(batched_forward(model, X_test_t) * 125.0, 0, 125)
        sigma_fixed = float(np.sqrt(aleatory_var))
        sigma_arr = np.full_like(mu_mse, sigma_fixed)
        rmse_mse, score_mse = C.rmse_score(y_test, mu_mse)
        picp_mse, mpiw_mse = picp_mpiw(y_test, mu_mse, sigma_arr)
        ece_mse = C.compute_ece(mu_mse, sigma_arr, y_test, CONF_LEVELS)
        out['mse_row'] = {'rmse': rmse_mse, 'score': score_mse, 'picp': picp_mse, 'mpiw': mpiw_mse,
                           'ece': ece_mse, 'sigma_fixed': sigma_fixed}
    return out


if __name__ == '__main__':
    C.require_fixed_hashseed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(os.path.join(PROJ_DIR, 'results', 'canonical_splits.json')) as f:
        CANON = json.load(f)
    ckpt_dir_leakfree = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

    # ---- 1. LSTM FD001/FD002/FD004 (no mse_row here -- separate file) ----
    out1 = {}
    for ds in C.DATASETS:
        print(f"\n{'=' * 10} LSTM MC-Dropout/MSE {ds} {'=' * 10}")
        out1[ds] = []
        for seed in C.SEEDS:
            ck = torch.load(os.path.join(ckpt_dir_leakfree, f"{ds}_MCDropoutMSE_seed{seed}.pt"),
                             map_location=device, weights_only=False)
            model = S1.MC_LSTM(ck['input_dim'], ck['hidden_dim'], ck['dropout']).to(device)
            model.load_state_dict(ck['state_dict'])
            r = eval_one(model, ds, seed, CANON, device, 'lstm', with_mse_row=False)
            out1[ds].append(r)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  seed={seed}: T50 kendall_gal_full PICP={r['T50']['kendall_gal_full']['picp']:.3f}")
    p1 = os.path.join(RESULTS_DIR, 'mcdropout_fixed_leakfree.json')
    with open(p1, 'w') as fp:
        json.dump(out1, fp, indent=2, default=float)
    print(f"Saved -> {p1}")

    # ---- 2. LSTM FD003 (with mse_row) ----
    print(f"\n{'=' * 10} LSTM MC-Dropout/MSE FD003 {'=' * 10}")
    out2 = []
    for seed in C.SEEDS:
        ck = torch.load(os.path.join(ckpt_dir_leakfree, f"FD003_MCDropoutMSE_seed{seed}.pt"),
                         map_location=device, weights_only=False)
        model = S1.MC_LSTM(ck['input_dim'], ck['hidden_dim'], ck['dropout']).to(device)
        model.load_state_dict(ck['state_dict'])
        r = eval_one(model, 'FD003', seed, CANON, device, 'lstm_fd003', with_mse_row=True)
        out2.append(r)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  seed={seed}: T50 kendall_gal_full PICP={r['T50']['kendall_gal_full']['picp']:.3f}")
    p2 = os.path.join(RESULTS_DIR, 'stepFD003_mcdropout_mse_leakfree_results.json')
    with open(p2, 'w') as fp:
        json.dump(out2, fp, indent=2, default=float)
    print(f"Saved -> {p2}")

    # ---- 3. Transformer, all 4 datasets (with mse_row) ----
    out3 = {}
    for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"\n{'=' * 10} Transformer MC-Dropout/MSE {ds} {'=' * 10}")
        out3[ds] = []
        for seed in C.SEEDS:
            model = T2.load_checkpoint_mc_model_t2('Transformer', T2.mc_ckpt_path('Transformer', ds, seed), device)
            r = eval_one(model, ds, seed, CANON, device, 'transformer', with_mse_row=True)
            out3[ds].append(r)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  seed={seed}: T50 kendall_gal_full PICP={r['T50']['kendall_gal_full']['picp']:.3f}")
    p3 = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_msemcd_leakfree_results.json')
    with open(p3, 'w') as fp:
        json.dump(out3, fp, indent=2, default=float)
    print(f"Saved -> {p3}")

    print("\nR8-B1 MC-Dropout/MSE re-eval complete (no retraining).")
