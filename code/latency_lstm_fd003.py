"""
FD003 completion (6/6): latency, platform batch=512 protocol, identical
protocol to an earlier unified-platform latency script (not included;
superseded) (priming pass + 2s clock warmup + no empty_cache between
methods + 50 warmup + 11-round median), FD003 only. Latency doesn't
depend on the weights, so it's measured with the leakfree checkpoints
(same architecture, so the numbers should match the old checkpoints --
this isn't redefining the latency convention).

Usage: `python3 latency_lstm_fd003.py <run_tag>`, run independently
multiple times (e.g. run1/run2/run3) with different run_tags, then pool
the raw rounds across all independent runs and take the median -- don't
cherry-pick a single run.
"""
import os
import sys
import json
import time

import numpy as np
import torch

import common as C
import mc_dropout_model as S1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
CKPT_DIR = os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm')

N_WARMUP = 50
N_ROUNDS = 11
PLATFORM_BATCH = 512
DS = 'FD003'


def time_one_round_cuda(fn):
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record(); fn(); ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e)


def measure(fn, n_warmup=N_WARMUP, n_rounds=N_ROUNDS):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    round_ms = [time_one_round_cuda(fn) for _ in range(n_rounds)]
    round_ms = np.array(round_ms)
    return {'round_ms_raw': round_ms.tolist(), 'median_ms': float(np.median(round_ms)),
            'std_ms': float(np.std(round_ms, ddof=1)), 'n_warmup': n_warmup, 'n_rounds': n_rounds}


def gpu_snapshot():
    import subprocess
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    procs = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                            capture_output=True, text=True).stdout.strip()
    return {'gpu_util_mem': out, 'compute_processes': procs if procs else 'none'}


def clock_warmup(nll_model, X_t, seconds=2.0):
    start = time.time()
    with torch.no_grad():
        while time.time() - start < seconds:
            for _ in range(20):
                nll_model(X_t)
    torch.cuda.synchronize()


def make_batch(X_test_np, batch_size):
    n_real = X_test_np.shape[0]
    reps = int(np.ceil(batch_size / n_real))
    return np.tile(X_test_np, (reps, 1, 1))[:batch_size]


if __name__ == '__main__':
    run_tag = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU snapshot before: {gpu_snapshot()}")

    train_df, test_df, true_ruls, feat_cols = C.load_and_process(DS)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    N_real = len(y_test)
    print(f"Real {DS} test windows: N={N_real}, tiled to {PLATFORM_BATCH}")
    X_np = make_batch(X_test, PLATFORM_BATCH)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)

    seed = C.SEEDS[0]
    nll_model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{seed}.pt"), device)
    ck = torch.load(os.path.join(CKPT_DIR, f"{DS}_MCDropoutMSE_seed{seed}.pt"), map_location=device, weights_only=False)
    mse_model = S1.MC_LSTM(ck['input_dim'], ck['hidden_dim'], ck['dropout']).to(device)
    mse_model.load_state_dict(ck['state_dict'])
    ens_models = [C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_LSTM_seed{s}.pt"), device) for s in C.SEEDS]
    cp_model = C.load_checkpoint_model(os.path.join(CKPT_DIR, f"{DS}_SplitCP_seed{seed}.pt"), device)

    print("Priming pass (untimed)...")
    with torch.no_grad():
        nll_model(X_t)
        mse_model.eval(); mse_model(X_t)
        mse_model.train(); mse_model(X_t)
        for mdl in ens_models:
            mdl(X_t)
        cp_model(X_t)
    torch.cuda.synchronize()
    mse_model.eval()

    print("GPU clock warm-up (~2s)...")
    clock_warmup(nll_model, X_t, seconds=2.0)

    results = {'dataset': DS, 'platform_batch': PLATFORM_BATCH, 'real_N': N_real, 'run_tag': run_tag,
               'gpu_snapshot_before': gpu_snapshot()}

    with torch.no_grad():
        m = measure(lambda: nll_model(X_t))
    results['NLL'] = m
    print(f"NLL: median={m['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    with torch.no_grad():
        m = measure(lambda: cp_model(X_t))
    results['CP_norm'] = m
    print(f"CP-norm: median={m['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    with torch.no_grad():
        m = measure(lambda: mse_model(X_t))
    results['MSE'] = m
    print(f"MSE: median={m['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    mse_model.train()
    T = 50
    def mc_unit():
        with torch.no_grad():
            for _ in range(T):
                mse_model(X_t)
    m = measure(mc_unit)
    results['MC_Dropout_T50'] = m
    print(f"MC-Dropout(T=50): median={m['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    def ens_unit():
        with torch.no_grad():
            for mdl in ens_models:
                mdl(X_t)
    m = measure(ens_unit)
    results['Deep_Ensemble_M5'] = m
    print(f"Deep Ensemble(M=5): median={m['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    results['gpu_snapshot_after'] = gpu_snapshot()

    nll_med = results['NLL']['median_ms']; mse_med = results['MSE']['median_ms']
    mc_med = results['MC_Dropout_T50']['median_ms']; ens_med = results['Deep_Ensemble_M5']['median_ms']
    cp_med = results['CP_norm']['median_ms']
    checks = {
        'CP_over_NLL_within_2pct': (abs(cp_med / nll_med - 1.0) < 0.02, cp_med / nll_med),
        'Ensemble_over_NLL_in_4.5_5.5': (4.5 <= ens_med / nll_med <= 5.5, ens_med / nll_med),
        'MCDropout_over_NLL_in_45_55': (45 <= mc_med / nll_med <= 55, mc_med / nll_med),
        'MSE_over_NLL_within_5pct': (abs(mse_med / nll_med - 1.0) < 0.05, mse_med / nll_med),
    }
    results['acceptance_checks'] = {k: {'pass': bool(v[0]), 'ratio': float(v[1])} for k, v in checks.items()}
    n_pass = sum(1 for v in checks.values() if v[0])
    results['n_pass_of_4'] = n_pass
    print(f"\n{run_tag}: {n_pass}/4 checks pass ({', '.join(f'{k}={v[1]:.3f}' for k,v in checks.items())})")

    out_path = os.path.join(RESULTS_DIR, f'latency_FD003_platform_batch512_{run_tag}.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"Saved -> {out_path}")
