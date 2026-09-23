"""
T2-A9: unified latency re-measurement, 10 rows (5 LSTM + 5 Transformer)
x 4 datasets, same session, batch=512 protocol, using the exact same
protocol as an earlier unified-platform latency script (not included;
superseded) / latency_lstm_fd003.py (priming pass + 2s GPU clock warm-up +
no empty_cache between methods + 50 warm-up iterations + median over 11
CUDA-event rounds). The 5 LSTM rows are also re-measured in this same
session (not copied from old numbers), so that all 10 rows' latency
ratios are comparable under the same GPU state -- an explicit requirement
for this task ("re-measure the 5 LSTM rows together to keep them in the
same session").

The four hard criteria are each checked once inside LSTM and once inside
Transformer (CP~=NLL, Ensemble~=5xNLL, MCDropout~=50xNLL, MSE~=NLL), 4x2=8
items in total.
"""
import os
import sys
import json
import time

import numpy as np
import torch

import common as C
import mc_dropout_model as S1
import transformer_common as T2

N_WARMUP = 50
N_ROUNDS = 11
PLATFORM_BATCH = 512
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']


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
    return {'gpu_util_mem': out}


def clock_warmup(model, X_t, seconds=2.0):
    start = time.time()
    with torch.no_grad():
        while time.time() - start < seconds:
            for _ in range(20):
                model(X_t)
    torch.cuda.synchronize()


def make_batch(X_test_np, batch_size):
    n_real = X_test_np.shape[0]
    reps = int(np.ceil(batch_size / n_real))
    return np.tile(X_test_np, (reps, 1, 1))[:batch_size]


def load_models(ds, backbone, device):
    seed = C.SEEDS[0]
    nll_model = T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, seed), device)
    mse_model = T2.load_checkpoint_mc_model_t2(backbone, T2.mc_ckpt_path(backbone, ds, seed), device)
    ens_models = [T2.load_checkpoint_model_t2(backbone, T2.nll_ckpt_path(backbone, ds, s), device) for s in C.SEEDS]
    if backbone == 'LSTM':
        cp_model = C.load_checkpoint_model(
            os.path.join(T2.PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_SplitCP_seed{seed}.pt"), device)
    else:
        cp_model = nll_model
    return nll_model, mse_model, ens_models, cp_model


def measure_backbone(ds, backbone, X_t, device):
    nll_model, mse_model, ens_models, cp_model = load_models(ds, backbone, device)

    print(f"  Priming pass ({backbone})...")
    with torch.no_grad():
        nll_model(X_t)
        mse_model.eval(); mse_model(X_t)
        mse_model.train(); mse_model(X_t)
        for mdl in ens_models:
            mdl(X_t)
        cp_model(X_t)
    torch.cuda.synchronize()
    mse_model.eval()

    print(f"  GPU clock warm-up ({backbone}, ~2s)...")
    clock_warmup(nll_model, X_t, seconds=2.0)

    results = {}
    with torch.no_grad():
        results['NLL'] = measure(lambda: nll_model(X_t))
    print(f"  [{backbone}] NLL: median={results['NLL']['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    with torch.no_grad():
        results['CP_norm'] = measure(lambda: cp_model(X_t))
    print(f"  [{backbone}] CP-norm: median={results['CP_norm']['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    with torch.no_grad():
        results['MSE'] = measure(lambda: mse_model(X_t))
    print(f"  [{backbone}] MSE: median={results['MSE']['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    mse_model.train()
    T = 50
    def mc_unit():
        with torch.no_grad():
            for _ in range(T):
                mse_model(X_t)
    results['MC_Dropout_T50'] = measure(mc_unit)
    print(f"  [{backbone}] MC-Dropout(T=50): median={results['MC_Dropout_T50']['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

    def ens_unit():
        with torch.no_grad():
            for mdl in ens_models:
                mdl(X_t)
    results['Deep_Ensemble_M5'] = measure(ens_unit)
    print(f"  [{backbone}] Deep Ensemble(M=5): median={results['Deep_Ensemble_M5']['median_ms']/PLATFORM_BATCH:.6f}ms/smp")

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
    print(f"  [{backbone}] {n_pass}/4 checks pass ({', '.join(f'{k}={v[1]:.3f}' for k, v in checks.items())})")

    del nll_model, mse_model, cp_model
    for m in ens_models:
        del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    run_tag = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU snapshot before: {gpu_snapshot()}")

    all_out = {'run_tag': run_tag, 'platform_batch': PLATFORM_BATCH, 'gpu_snapshot_before': gpu_snapshot()}

    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)
        X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
        N_real = len(y_test)
        X_np = make_batch(X_test, PLATFORM_BATCH)
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        print(f"Real {ds} test windows: N={N_real}, tiled to {PLATFORM_BATCH}")

        ds_out = {'real_N': N_real}
        for backbone in ['LSTM', 'Transformer']:
            print(f"\n--- {ds} / {backbone} ---")
            ds_out[backbone] = measure_backbone(ds, backbone, X_t, device)
        all_out[ds] = ds_out

    all_out['gpu_snapshot_after'] = gpu_snapshot()

    out_path = os.path.join(T2.TRANSFORMER_DIR, f'latency_T2_unified_platform_batch512_{run_tag}.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
