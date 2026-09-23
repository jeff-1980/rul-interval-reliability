"""
T2 supplementary measurement: Transformer latency batch-size plateau sweep
{512,1024,2048,4096}, using the same priming + 2s clock warm-up + 50
warm-up iterations + 11-round CUDA-event median method as the old
batch=512 protocol, measuring only NLL (locate the plateau first with the
cheapest single mechanism, then re-measure all 5 rows at the plateau
batch size once found). Measured separately for each of the 4 datasets
(different input_dim/sequence counts).
"""
import os
import json
import time

import numpy as np
import torch

import common as C
import transformer_common as T2

N_WARMUP = 50
N_ROUNDS = 11
BATCH_LEVELS = [512, 1024, 2048, 4096]
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
    round_ms = np.array([time_one_round_cuda(fn) for _ in range(n_rounds)])
    return {'median_ms': float(np.median(round_ms)), 'std_ms': float(np.std(round_ms, ddof=1))}


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


if __name__ == '__main__':
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_out = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)
        X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)

        seed = C.SEEDS[0]
        nll_model = T2.load_checkpoint_model_t2('Transformer', T2.nll_ckpt_path('Transformer', ds, seed), device)

        ds_out = {}
        for batch in BATCH_LEVELS:
            X_np = make_batch(X_test, batch)
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                nll_model(X_t)
            torch.cuda.synchronize()
            clock_warmup(nll_model, X_t, seconds=1.0)
            with torch.no_grad():
                m = measure(lambda: nll_model(X_t))
            per_smp_us = m['median_ms'] / batch * 1000.0
            ds_out[str(batch)] = {'median_ms_total': m['median_ms'], 'per_sample_us': per_smp_us}
            print(f"  batch={batch:5d}: median={m['median_ms']:.4f}ms total, {per_smp_us:.4f}us/sample")
            del X_t
            torch.cuda.empty_cache()

        # plateau criterion: per-sample latency change <5% from one level to the next counts as plateaued
        plateau_batch = None
        levels = [str(b) for b in BATCH_LEVELS]
        for i in range(len(levels) - 1):
            v0 = ds_out[levels[i]]['per_sample_us']
            v1 = ds_out[levels[i + 1]]['per_sample_us']
            rel_change = abs(v1 - v0) / v0
            if rel_change < 0.05:
                plateau_batch = int(levels[i])
                break
        ds_out['plateau_batch'] = plateau_batch
        print(f"  -> plateau batch: {plateau_batch}")

        del nll_model
        torch.cuda.empty_cache()
        all_out[ds] = ds_out

    out_path = os.path.join(T2.TRANSFORMER_DIR, 't2_transformer_latency_batchsweep.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
