"""
R2-4：延迟重新设计。
- 去掉"期望倍数(M/T)作为接受/拒绝判据"——不再有PASS/FAIL，倍数只作诊断
  注记随数字一起报告。
- 报告全部重复测量的分布（中位数+IQR），不是单一中位数。
- 补 batch=1 和 batch=32 两档（原有 batch=512 保留）。
- CUDA-event（纯GPU核函数时间）与端到端 wall-clock（含Python/CPU调度、
  H2D/D2H等）分开报告——两者此前一直混用同一套 CUDA-event 计时，这次
  额外加一层 time.perf_counter() 包裹的端到端计时对照。

两骨干×4数据集×5机制×3个batch档，每个组合仍是50热身+11轮，报中位数和
IQR，不做通过/不通过判定。
"""
import os
import sys
import json
import time

import numpy as np
import torch

import common as C
import transformer_common as T2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'generated')
R2_DIR = os.path.join(RESULTS_DIR, 'leakfree_r2')

N_WARMUP = 50
N_ROUNDS = 11
BATCH_LEVELS = [1, 32, 512]
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
BACKBONES = ['LSTM', 'Transformer']


def time_one_round_cuda_event(fn):
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record(); fn(); ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e)


def time_one_round_wallclock(fn):
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0  # ms


def measure_both(fn, n_warmup=N_WARMUP, n_rounds=N_ROUNDS):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    cuda_ms = np.array([time_one_round_cuda_event(fn) for _ in range(n_rounds)])
    wall_ms = np.array([time_one_round_wallclock(fn) for _ in range(n_rounds)])
    def stats(a):
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        return {'median_ms': float(med), 'iqr_ms': float(q3 - q1), 'q1_ms': float(q1), 'q3_ms': float(q3),
                'all_rounds_ms': a.tolist()}
    return {'cuda_event': stats(cuda_ms), 'wallclock': stats(wall_ms)}


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
            os.path.join(PROJ_DIR, 'results', 'checkpoints', 'lstm', f"{ds}_SplitCP_seed{seed}.pt"), device)
    else:
        cp_model = nll_model
    return nll_model, mse_model, ens_models, cp_model


def measure_backbone_batch(ds, backbone, batch, device):
    train_df, test_df, true_ruls, feat_cols = C.load_and_process(ds)
    X_test, y_test = C.create_sequences(test_df, feat_cols, mode='test', true_ruls=true_ruls)
    X_np = make_batch(X_test, batch)
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)

    nll_model, mse_model, ens_models, cp_model = load_models(ds, backbone, device)
    with torch.no_grad():
        nll_model(X_t); mse_model.eval(); mse_model(X_t); mse_model.train(); mse_model(X_t)
        for m in ens_models:
            m(X_t)
        cp_model(X_t)
    torch.cuda.synchronize()
    mse_model.eval()
    clock_warmup(nll_model, X_t, seconds=1.0)

    results = {}
    with torch.no_grad():
        results['NLL'] = measure_both(lambda: nll_model(X_t))
        results['CP_norm'] = measure_both(lambda: cp_model(X_t))
        results['MSE'] = measure_both(lambda: mse_model(X_t))

    mse_model.train()
    T = 50
    def mc_unit():
        with torch.no_grad():
            for _ in range(T):
                mse_model(X_t)
    results['MC_Dropout_T50'] = measure_both(mc_unit)

    def ens_unit():
        with torch.no_grad():
            for m in ens_models:
                m(X_t)
    results['Deep_Ensemble_M5'] = measure_both(ens_unit)

    # 倍数诊断注记（不作判据）
    nll_med = results['NLL']['cuda_event']['median_ms']
    results['diagnostic_ratios'] = {
        'CP_over_NLL': results['CP_norm']['cuda_event']['median_ms'] / nll_med,
        'Ensemble_over_NLL': results['Deep_Ensemble_M5']['cuda_event']['median_ms'] / nll_med,
        'MCDropout_over_NLL': results['MC_Dropout_T50']['cuda_event']['median_ms'] / nll_med,
        'MSE_over_NLL': results['MSE']['cuda_event']['median_ms'] / nll_med,
    }
    print(f"    batch={batch}: NLL(cuda)={nll_med/batch*1000:.4f}us/smp  "
          f"NLL(wall)={results['NLL']['wallclock']['median_ms']/batch*1000:.4f}us/smp  "
          f"ratios(cuda)={ {k: round(v,2) for k,v in results['diagnostic_ratios'].items()} }")

    del nll_model, mse_model, cp_model
    for m in ens_models:
        del m
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    run_tag = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}  run_tag={run_tag}")

    all_out = {}
    for ds in DATASETS:
        print(f"\n{'=' * 20} {ds} {'=' * 20}")
        all_out[ds] = {}
        for backbone in BACKBONES:
            print(f"  --- {backbone} ---")
            all_out[ds][backbone] = {}
            for batch in BATCH_LEVELS:
                all_out[ds][backbone][str(batch)] = measure_backbone_batch(ds, backbone, batch, device)

    out_path = os.path.join(R2_DIR, f'latency_redesign_{run_tag}.json')
    with open(out_path, 'w') as fp:
        json.dump(all_out, fp, indent=2, default=float)
    print(f"\nSaved -> {out_path}")
